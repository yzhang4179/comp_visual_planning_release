import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from cvp.cosmos import CICosmos
from cvp.dataset import SapienDataset, h5_worker_init_fn
from cvp.diffusion import DDPM
from cvp.nn_diffusion import DiT
from cvp.utils import count_parameters, seed_everything
from cvp.utils.hub import resolve_file


@hydra.main(config_path="configs", config_name="config", version_base=None)
def pipeline(args):
    # ---------------------- DDP Setup ----------------------#
    # torchrun sets RANK, WORLD_SIZE and LOCAL_RANK
    rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    is_master = rank == 0
    device = f"cuda:{local_rank}"

    seed_everything(args.seed + rank, deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    ckpt_dir = Path(args.train_dir) / "planner"
    if is_master:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Create Dataset ----------------------#
    dataset = SapienDataset(dataset_path=args.dataset_path, actions_stats_path=resolve_file(args.actions_stats_path, args.hf_repo, args.ckpt_root),
                            horizon=args.input_size[0], pad_before=args.pad_before, pad_after=args.pad_after)
    if is_master:
        print(dataset)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                                num_workers=args.num_workers, drop_last=True, worker_init_fn=h5_worker_init_fn)

    # ---------------------- Create Diffusion Model ----------------------#
    cosmos = CICosmos(
        encoder_path=resolve_file(f"{args.cosmos_dir}/encoder.jit", args.hf_repo, args.ckpt_root),
        decoder_path=resolve_file(f"{args.cosmos_dir}/decoder.jit", args.hf_repo, args.ckpt_root),
        device=device,
    )

    nn_diffusion = DiT(
        in_dim=args.in_dim,
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        input_size=tuple(args.input_size),
        patch_size=tuple(args.patch_size),
        dropout=args.dropout,
    ).to(device)
    if is_master:
        print(f"DiT parameters: {count_parameters(nn_diffusion) / 1e6:.2f}M")

    # wrap with DDP before creating the agent, so that its optimizer holds the DDP parameters
    ddp_diffusion = DDP(nn_diffusion, device_ids=[local_rank], output_device=local_rank)

    agent = DDPM(inst_horizon=args.input_size[0],
        nn_diffusion=ddp_diffusion, nn_condition=None, diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate,
        predict_noise=args.predict_noise, beta_schedule=args.beta_schedule,
        optim_params={"lr": args.lr, "weight_decay": args.weight_decay}, device=device)
    lr_scheduler = CosineAnnealingLR(agent.optimizer, args.gradient_steps)
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []

    # ---------------------- Training ----------------------#
    pbar = tqdm(total=args.gradient_steps, desc="Training", disable=not is_master)
    epoch = 0
    while n_gradient_step < args.gradient_steps:
        train_sampler.set_epoch(epoch)
        for batch in dataloader:
            # (batch, horizon, 3, 256, 256) images -> (batch, horizon, 16, 16, 16) latents
            nlatents = cosmos.encode(batch["images"].to(device, non_blocking=True)).to(torch.float32)

            # update diffusion
            diffusion_loss = agent.update(x0=nlatents, condition=None)["loss"]
            lr_scheduler.step()
            diffusion_loss_list.append(diffusion_loss)
            n_gradient_step += 1
            pbar.update(1)

            if is_master and n_gradient_step % args.log_interval == 0:
                tqdm.write(f"[step {n_gradient_step}] avg_diffusion_loss: {np.mean(diffusion_loss_list):.6f}")
                diffusion_loss_list = []

            if is_master and n_gradient_step % args.save_interval == 0:
                agent.save(ckpt_dir / f"diffusion_{n_gradient_step}.ckpt")

            if n_gradient_step >= args.gradient_steps:
                break
        epoch += 1
    pbar.close()

    if is_master:
        agent.save(ckpt_dir / "diffusion_final.ckpt")
        print("done")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    pipeline()
