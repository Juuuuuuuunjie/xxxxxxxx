import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_eeg
from targets import compute_channel_time_patch_targets


class EEGPretrainDataset(Dataset):
    """
    EEG 预训练数据集。

    输入 sample:
        {
            "signal": np.ndarray [C_in, T],
            "channel_names": list[str],
            "sfreq": float,
        }

    输出:
        token_inputs: [S, L]
        targets: [S, F, K]
        token_channel_indices: [S]
        token_time_indices: [S]
        token_valid_mask: [S]
        channel_valid_mask: [C]
    """

    def __init__(self, samples, cfg):
        self.samples = samples
        self.cfg = cfg

        self.target_num_points = int(round(
            cfg.data.clip_seconds * cfg.data.target_sfreq
        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        signal = item["signal"]
        channel_names = item["channel_names"]
        sfreq = item["sfreq"]

        # 1) 预处理到固定 64 通道、固定采样率、固定长度
        signal_64, channel_valid_mask = preprocess_eeg(
            signal=signal,
            channel_names=channel_names,
            orig_sfreq=sfreq,
            target_sfreq=self.cfg.data.target_sfreq,
            target_num_points=self.target_num_points,
        )

        # signal_64: [64, T]

        # 2) 切 channel-time patch，并生成时频 target
        patches, targets = compute_channel_time_patch_targets(
            signal=signal_64,
            sfreq=self.cfg.data.target_sfreq,
            patch_seconds=self.cfg.data.patch_seconds,
            patch_stride_seconds=self.cfg.data.patch_stride_seconds,
            target_frame_seconds=self.cfg.data.target_frame_seconds,
            stft_window_seconds=self.cfg.data.stft_window_seconds,
            stft_hop_seconds=self.cfg.data.stft_hop_seconds,
            band_defs=self.cfg.data.band_defs,
            eps=self.cfg.data.eps,
        )

        # patches: [C, N, L]
        # targets: [C, N, F, K]

        C, N, L = patches.shape
        _, _, F, K = targets.shape

        # 3) flatten 成 token 序列
        token_inputs = patches.reshape(C * N, L).astype(np.float32)
        token_targets = targets.reshape(C * N, F, K).astype(np.float32)

        # token 顺序:
        # c0_t0, c0_t1, ..., c0_tN,
        # c1_t0, c1_t1, ..., c1_tN,
        # ...
        token_channel_indices = np.repeat(np.arange(C), N).astype(np.int64)
        token_time_indices = np.tile(np.arange(N), C).astype(np.int64)

        # 4) 根据通道有效性生成 token_valid_mask
        token_valid_mask = channel_valid_mask[token_channel_indices].astype(np.float32)

        return {
            "token_inputs": torch.tensor(token_inputs, dtype=torch.float32),
            "targets": torch.tensor(token_targets, dtype=torch.float32),
            "token_channel_indices": torch.tensor(token_channel_indices, dtype=torch.long),
            "token_time_indices": torch.tensor(token_time_indices, dtype=torch.long),
            "token_valid_mask": torch.tensor(token_valid_mask, dtype=torch.float32),
            "channel_valid_mask": torch.tensor(channel_valid_mask, dtype=torch.float32),
        }


def collate_fn(batch):
    token_inputs = torch.stack([x["token_inputs"] for x in batch], dim=0)
    targets = torch.stack([x["targets"] for x in batch], dim=0)
    token_channel_indices = torch.stack([x["token_channel_indices"] for x in batch], dim=0)
    token_time_indices = torch.stack([x["token_time_indices"] for x in batch], dim=0)
    token_valid_mask = torch.stack([x["token_valid_mask"] for x in batch], dim=0)
    channel_valid_mask = torch.stack([x["channel_valid_mask"] for x in batch], dim=0)

    return {
        "token_inputs": token_inputs,
        "targets": targets,
        "token_channel_indices": token_channel_indices,
        "token_time_indices": token_time_indices,
        "token_valid_mask": token_valid_mask,
        "channel_valid_mask": channel_valid_mask,
    }
