from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cvp.cosmos import CICosmos
from cvp.dataset import SapienDataset, h5_worker_init_fn
from cvp.invdyn import InverseDynamicsModel
from cvp.utils import loop_dataloader, seed_everything
from cvp.utils.hub import resolve_file


@hydra.main(config_path="configs", config_name="config", version_base=None)
def pipeline(args):
    seed_everything(args.seed, deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = args.device
    ckpt_dir = Path(args.train_dir) / "invdyn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Create Dataset ----------------------#
    dataset = SapienDataset(dataset_path=args.dataset_path, actions_stats_path=resolve_file(args.actions_stats_path, args.hf_repo, args.ckpt_root),
                            horizon=args.input_size[0], pad_before=args.pad_before, pad_after=args.pad_after)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.invdyn_batch_size, num_workers=args.num_workers,
                                shuffle=True, worker_init_fn=h5_worker_init_fn)

    # ---------------------- Create Inverse Dynamics Model ----------------------#
    cosmos = CICosmos(
        encoder_path=resolve_file(f"{args.cosmos_dir}/encoder.jit", args.hf_repo, args.ckpt_root),
        decoder_path=resolve_file(f"{args.cosmos_dir}/decoder.jit", args.hf_repo, args.ckpt_root),
        device=device,
    )
    invdyn = InverseDynamicsModel(embed_dim=args.invdyn_embed_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(invdyn.parameters(), lr=args.lr)
    invdyn.train()
    n_gradient_step = 0
    invdyn_loss_list = []

    # ---------------------- Training ----------------------#
    for batch in tqdm(loop_dataloader(dataloader), desc="Training", total=args.gradient_steps):
        nimgs = batch["images"].to(device)
        nactions = batch["actions"].to(device)

        # (batch, horizon, 3, 256, 256) images -> (batch, horizon, 16, 16, 16) latents
        nlatents = cosmos.encode(nimgs).to(torch.float32)

        # predict the action between every two consecutive frames
        nactions_pred = invdyn(nlatents[:, :-1], nlatents[:, 1:])
        invdyn_loss = nn.MSELoss()(nactions_pred, nactions[:, :-1])
        optimizer.zero_grad()
        invdyn_loss.backward()
        optimizer.step()
        invdyn_loss_list.append(invdyn_loss.item())
        n_gradient_step += 1

        if n_gradient_step % args.log_interval == 0:
            tqdm.write(f"[step {n_gradient_step}] avg_invdyn_loss: {np.mean(invdyn_loss_list):.6f}")
            invdyn_loss_list = []

        if n_gradient_step % args.save_interval == 0:
            torch.save(invdyn.state_dict(), ckpt_dir / f"invdyn_{n_gradient_step}.ckpt")

        if n_gradient_step >= args.gradient_steps:
            break
    torch.save(invdyn.state_dict(), ckpt_dir / "invdyn_final.ckpt")
    print("done")


if __name__ == "__main__":
    pipeline()
