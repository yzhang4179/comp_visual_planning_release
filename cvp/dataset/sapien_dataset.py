import json
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch


def build_indices_per_episode(
    episode_lengths: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """
    Build sampling indices per episode without concatenating arrays in memory.
    Returns an int64 array of shape [N, 5]: (ep_id, buffer_start, buffer_end, sample_start, sample_end)
    """
    assert sequence_length >= 1
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    idxs = []
    for ep_id, ep_len in enumerate(episode_lengths):
        # inclusive start range with padding
        min_start = -pad_before
        max_start = ep_len - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start = max(idx, 0)
            buffer_end = min(idx + sequence_length, ep_len)
            start_offset = buffer_start - idx
            end_offset = (idx + sequence_length) - buffer_end
            sample_start = 0 + start_offset
            sample_end = sequence_length - end_offset
            # (ep_id, where to slice from episode arrays, where to place in the fixed-length sample)
            idxs.append((ep_id, buffer_start, buffer_end, sample_start, sample_end))
    return np.asarray(idxs, dtype=np.int64)


class SapienDataset(torch.utils.data.Dataset):
    """
    Clips of `horizon` frames from an HDF5 file of trajectories, `ep<k>/<trajectory>/{images, actions}`, with uint8
    images of shape (T, 256, 256, 3) and 8-dim actions.
    The file is opened lazily in each worker. Images are scaled to [-1, 1], actions are min-max normalized
    to [-1, 1] with the stats in `actions_stats_path`. Clips at the episode boundaries repeat the edge frames.
    """

    def __init__(
        self,
        dataset_path: str,
        actions_stats_path: str,
        horizon: int = 16,
        pad_before: int = 2,
        pad_after: int = 2,
        keys: Tuple[str, str] = ("images", "actions"),
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.keys = keys
        self._h5: Optional[h5py.File] = None  # opened lazily per worker

        # ---- Build episode list without reading arrays ----
        episodes: List[Tuple[str, int]] = []  # (group_path, length)
        with h5py.File(dataset_path, "r") as f:
            eps = sorted(f.keys(), key=lambda d: int(d[2:]))
            for ep in eps:
                grp = f[ep]
                for sub in grp.keys():
                    path = f"{ep}/{sub}"
                    length = grp[sub]["actions"].shape[0]
                    episodes.append((path, int(length)))

        self.episodes = episodes
        self.episode_lengths = np.asarray([L for _, L in episodes], dtype=np.int64)
        self.indices = build_indices_per_episode(
            self.episode_lengths, self.horizon, self.pad_before, self.pad_after
        )

        with open(actions_stats_path, "r") as fh:
            stats = json.load(fh)
        self.act_min = torch.as_tensor(stats["min"], dtype=torch.float32)
        self.act_max = torch.as_tensor(stats["max"], dtype=torch.float32)

    def __repr__(self):
        return f"SapienDataset({self.dataset_path}): {len(self.episodes)} trajectories, {len(self)} clips of {self.horizon} frames"

    def _ensure_open(self):
        if self._h5 is None:
            # swmr for safe concurrent reads (read-only)
            self._h5 = h5py.File(self.dataset_path, "r", swmr=True)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        self._ensure_open()
        ep_id, b_start, b_end, s_start, s_end = self.indices[i]
        path, ep_len = self.episodes[int(ep_id)]

        out = {}
        grp = self._h5[path]

        # Slice and pad per key
        for key in self.keys:
            arr = grp[key]  # HDF5 dataset
            sample = arr[int(b_start):int(b_end)]  # (L, ...)
            # Pad to horizon when window hits episode boundaries
            if int(s_start) > 0 or int(s_end) < self.horizon:
                shape = (self.horizon,) + sample.shape[1:]
                data = np.zeros(shape, dtype=arr.dtype)
                if int(s_start) > 0:
                    data[:int(s_start)] = sample[0]
                if int(s_end) < self.horizon:
                    data[int(s_end):] = sample[-1]
                data[int(s_start):int(s_end)] = sample
            else:
                data = sample
            out[key] = data

        # Images: uint8 [T,H,W,3] -> float32 [T,3,H,W] in [-1,1]
        imgs = out["images"]
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
        imgs = imgs.to(dtype=torch.float32)
        imgs = imgs.div_(127.5).sub_(1.0)
        out["images"] = imgs

        # Actions: min-max normalized to [-1, 1]
        acts = torch.from_numpy(out["actions"]).to(torch.float32)
        denom = (self.act_max - self.act_min).clamp_min_(1e-8)
        acts = (acts - self.act_min) / denom
        acts = acts.clamp_(0.0, 1.0)
        acts = acts * 2.0 - 1.0
        out["actions"] = acts

        return out

    def __del__(self):
        try:
            if getattr(self, "_h5", None) is not None:
                self._h5.close()
        except Exception:
            pass


def h5_worker_init_fn(worker_id: int):
    """
    Ensure HDF5 handle isn't inherited from the parent (forked).
    Each worker opens its own handle lazily in __getitem__.
    """
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if getattr(ds, "_h5", None) is not None:
        try:
            ds._h5.close()
        except Exception:
            pass
        ds._h5 = None
