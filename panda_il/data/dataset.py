"""A torch Dataset of observation / action windows over the synced 20 Hz arrays.

Everything is loaded into RAM once (220 episodes of two 96 x 96 cameras are a
few GB of uint8), which on Windows beats spawned DataLoader workers. Windows
follow Diffusion Policy: for an episode with T actions and T + 1 observations,
a window starting at step ``i`` holds the ``obs_horizon`` observations ending at
``i`` and the ``pred_horizon`` actions starting at ``i``; starts range from
``-(obs_horizon - 1)`` to ``T - pred_horizon + action_horizon - 1`` and
out-of-range frames repeat the first or last one (``pad_before`` / ``pad_after``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from panda_il.data.schema import Dataset


@dataclass
class WindowConfig:
    obs_mode: str = "state"  # "state" (29-D) or "image" (22-D proprio + cameras)
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    cameras: tuple[str, ...] = ("front", "wrist")

    @property
    def pad_before(self) -> int:
        return self.obs_horizon - 1

    @property
    def pad_after(self) -> int:
        return self.action_horizon - 1


@dataclass
class Episode:
    lowdim: np.ndarray  # (T + 1, D) float32: state or proprio
    action: np.ndarray  # (T, 7) float32
    images: dict[str, np.ndarray]  # camera -> (T + 1, H, W, 3) uint8, delivered frames per tick


def load_episodes(dataset: Dataset, episode_ids: list[int], cfg: WindowConfig) -> list[Episode]:
    episodes = []
    for episode_id in episode_ids:
        record = dataset.read_episode(episode_id, load_images=cfg.obs_mode == "image")
        lowdim = record.synced["state" if cfg.obs_mode == "state" else "proprio"].astype(np.float32)
        action = record.synced["action"].astype(np.float32)
        images = {}
        if cfg.obs_mode == "image":
            for cam in cfg.cameras:
                index = record.synced[f"image_index_{cam}"]
                images[cam] = np.asarray(record.raw[f"cam_{cam}"]["image"])[index]
        episodes.append(Episode(lowdim, action, images))
    return episodes


class WindowDataset(TorchDataset):
    def __init__(self, episodes: list[Episode], cfg: WindowConfig):
        self.episodes = episodes
        self.cfg = cfg
        self.index: list[tuple[int, int]] = []  # (episode, window start)
        for e, episode in enumerate(episodes):
            n_actions = len(episode.action)
            for start in range(-cfg.pad_before, n_actions - cfg.pred_horizon + cfg.pad_after + 1):
                self.index.append((e, start))

    def __len__(self) -> int:
        return len(self.index)

    @property
    def lowdim_dim(self) -> int:
        return self.episodes[0].lowdim.shape[1]

    def _obs_slice(self, length: int, start: int) -> np.ndarray:
        idx = np.arange(start - self.cfg.pad_before, start + 1)
        return np.clip(idx, 0, length - 1)

    def _action_slice(self, n_actions: int, start: int) -> np.ndarray:
        idx = np.arange(start, start + self.cfg.pred_horizon)
        return np.clip(idx, 0, n_actions - 1)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, start = self.index[i]
        episode = self.episodes[e]
        obs_idx = self._obs_slice(len(episode.lowdim), start)
        act_idx = self._action_slice(len(episode.action), start)
        item = {
            "lowdim": torch.from_numpy(episode.lowdim[obs_idx]),  # (To, D)
            "action": torch.from_numpy(episode.action[act_idx]),  # (Tp, 7)
        }
        for cam, frames in episode.images.items():
            item[f"image_{cam}"] = torch.from_numpy(frames[obs_idx])  # (To, H, W, 3) uint8
        return item

    def all_lowdim(self) -> np.ndarray:
        return np.concatenate([e.lowdim for e in self.episodes])

    def all_actions(self) -> np.ndarray:
        return np.concatenate([e.action for e in self.episodes])
