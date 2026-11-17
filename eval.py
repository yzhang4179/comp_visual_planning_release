import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from cvp.cosmos import CICosmos, latent2video, video2latent
from cvp.diffusion import CVP
from cvp.invdyn import InverseDynamicsModel
from cvp.nn_diffusion import DiT
from cvp.utils import save_mp4, seed_everything
from cvp.utils.hub import resolve_file
from env import ENVS, load_eval_states


@hydra.main(config_path="configs", config_name="config", version_base=None)
def pipeline(args):
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    (output_dir / "plans").mkdir(parents=True, exist_ok=True)
    (output_dir / "rollouts").mkdir(parents=True, exist_ok=True)

    # ---------------------- Create Diffusion Model ----------------------#
    nn_diffusion = DiT(
        in_dim=args.in_dim,
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        input_size=tuple(args.input_size),
        patch_size=tuple(args.patch_size),
        dropout=args.dropout,
    )

    agent = CVP(
        inst_horizon=args.input_size[0],
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise,
        beta_schedule=args.beta_schedule,
        x_min=args.x_min,
        x_max=args.x_max,
        device=args.device,
    )
    agent.load(resolve_file(args.planner_ckpt, args.hf_repo, args.ckpt_root))
    agent.eval()

    # ---------------------- Create Inverse Dynamics Model ----------------------#
    invdyn = InverseDynamicsModel(embed_dim=args.invdyn_embed_dim, dropout=args.dropout).to(args.device)
    invdyn.load_state_dict(torch.load(resolve_file(args.invdyn_ckpt, args.hf_repo, args.ckpt_root), map_location=args.device, weights_only=True))
    invdyn.eval()

    with open(resolve_file(args.actions_stats_path, args.hf_repo, args.ckpt_root)) as f:
        stats = json.load(f)
    action_min = np.array(stats["min"])
    action_max = np.array(stats["max"])

    cosmos = CICosmos(
        encoder_path=resolve_file(f"{args.cosmos_dir}/encoder.jit", args.hf_repo, args.ckpt_root),
        decoder_path=resolve_file(f"{args.cosmos_dir}/decoder.jit", args.hf_repo, args.ckpt_root),
        device=args.device,
    )

    # ---------------------- Create Environment ----------------------#
    # raster renderer, since the planner was trained on raster images and plans from the rendered start and goal
    env = ENVS[args.env_name](start=args.start, goal=args.goal, recording=True, headless=True)
    eval_states = load_eval_states(args.env_name)

    episode_success = []
    episode_times = []
    for i in range(args.num_episodes):
        # ---------------------- Start and Goal Images ----------------------#
        goal_entry = eval_states[f"ep{i}"][f"goal_{args.goal}"]
        start_state = eval_states[f"ep{i}"][f"start_{args.start}"]["state"]
        goal_img = env.render_state(goal_entry["state"], start=goal_entry["render_start"], goal=args.goal)
        start_img = env.render_state(start_state, start=args.start, goal=args.goal)
        env.reset(start=args.start, goal=args.goal, initial_states=start_state)

        # ---------------------- Plan ----------------------#
        start_goal_latents = video2latent(np.stack([start_img, goal_img], axis=0), cosmos)  # (1, 2, C, h, w)
        prior = torch.zeros((1, args.canon_horizon, *start_goal_latents.shape[2:]), device=args.device)
        prior[0, 0] = start_goal_latents[0, 0]
        prior[0, -1] = start_goal_latents[0, -1]

        start_time = time.time()
        if args.method == "cvp":
            zt, _ = agent.cvp_sample(
                priors=prior,
                # ----------------- sampling ----------------- #
                temperature=args.temperature,
                sample_steps=args.sample_steps,
                # ------------------ chunks ------------------ #
                canon_horizon=args.canon_horizon,
                stride=args.stride,
                horizon_ctx=OmegaConf.to_container(args.horizon_ctx),
                # ------------------ guidance ----------------- #
                wg=args.wg,
                discount=args.discount,
                w_sync=args.w_sync,
                sync_loss=args.sync_loss,
            )
        elif args.method == "gsc":
            zt, _ = agent.gsc_sample(
                priors=prior,
                # ----------------- sampling ----------------- #
                temperature=args.temperature,
                sample_steps=args.sample_steps,
                # ------------------ chunks ------------------ #
                canon_horizon=args.canon_horizon,
                stride=args.stride,
                horizon_ctx=OmegaConf.to_container(args.horizon_ctx),
            )
        else:
            raise ValueError(f"Unknown method {args.method}, expected cvp or gsc")
        plan_time = time.time() - start_time

        # ---------------------- Inverse Dynamics ----------------------#
        with torch.no_grad():
            nactions = invdyn(zt[0, :-1], zt[0, 1:]).cpu().numpy()
        actions = nactions.clip(-1.0, 1.0)
        # [-1, 1] -> [action_min, action_max]
        actions = (actions + 1.0) / 2.0 * np.clip(action_max - action_min, 1e-8, None) + action_min
        actions[:, -1] = np.sign(actions[:, -1]).astype(actions.dtype)

        # ---------------------- Execute ----------------------#
        reward = 0
        for action in actions:
            _, reward, _, _ = env.step(action)
            if reward == 1:
                break

        episode_success.append(float(reward))
        episode_times.append(plan_time)
        save_mp4(latent2video(cosmos, zt), output_dir / "plans" / f"ep{i}_success_{int(reward)}.mp4", fps=args.video_fps)
        save_mp4(env.get_recordings(), output_dir / "rollouts" / f"ep{i}_success_{int(reward)}.mp4", fps=args.video_fps)
        print(f"Episode {i}, success: {reward}", flush=True)
        env.clear_cache(clear=(i % 3 == 0))  # free SAPIEN's render cache every 3 episodes

    # ---------------------- Metrics ----------------------#
    print(f"Success rate: {np.mean(episode_success):.3%}")
    metrics = {
        "avg_success_rate (%)": round(np.mean(episode_success) * 100, 3),
        "avg_plan_time (s)": round(np.mean(episode_times), 3),
        "success": episode_success,
        "plan_times (s)": np.round(np.array(episode_times), 3).tolist(),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    pipeline()
