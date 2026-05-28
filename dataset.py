import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_eeg
from targets import compute_channel_time_patch_targets


class EEGPretrainDataset(Dataset):
    """
    EEG 预训练数据集。

    输出:
        token_inputs: [S, L]
        targets: [S, F, K]
        token_channel_indices: [S]
        token_time_indices: [S]
        token_valid_mask: [S]
    """
    def __init__(self, samples, cfg):
        self.samples = samples
        self.cfg = cfg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        signal = item["signal"]
        channel_names = item["channel_names"]
        sfreq = item["sfreq"]

        signal_64, channel_valid_mask = preprocess_eeg(
            signal=signal,
            channel_names=channel_names,
            orig_sfreq=sfreq,
            target_sfreq=self.cfg.data.target_sfreq,
        )

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

        C, N, L = patches.shape
        F = len(self.cfg.data.band_defs)
        K = int(round(self.cfg.data.patch_seconds / self.cfg.data.target_frame_seconds))

        token_inputs = patches.reshape(C * N, L).astype(np.float32)
        token_targets = targets.reshape(C * N, F, K).astype(np.float32)

        # token 顺序:
        # c0_t0, c0_t1, ..., c0_tN,
        # c1_t0, c1_t1, ..., c1_tN,
        # ...
        token_channel_indices = np.repeat(np.arange(C), N).astype(np.int64)

        token_time_indices = np.tile(np.arange(N), C).astype(np.int64)

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


def build_mock_samples(num_samples=32, clip_seconds=10, sfreq=100):
    samples = []

    T = int(round(clip_seconds * sfreq))

    base_channels = [
        "Fp1", "Fp2",
        "F3", "F4",
        "C3", "C4",
        "P3", "P4",
        "O1", "O2",
        "Fz", "Cz", "Pz",
        "T7", "T8", "Oz",
    ]

    for _ in range(num_samples):
        c_in = len(base_channels)
        t = np.arange(T) / sfreq

        signal = []

        for ch in range(c_in):
            alpha_amp = 0.8 + 0.4 * np.sin(2 * np.pi * 0.3 * t + np.random.rand())
            theta_amp = 0.4 + 0.2 * np.sin(2 * np.pi * 0.2 * t + np.random.rand())
            beta_amp = 0.2 + 0.1 * np.sin(2 * np.pi * 0.5 * t + np.random.rand())

            x = (
                alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand()) +
                theta_amp * np.sin(2 * np.pi * 6 * t + np.random.rand()) +
                beta_amp * np.sin(2 * np.pi * 20 * t + np.random.rand()) +
                0.3 * np.random.randn(T)
            )

            signal.append(x)

        signal = np.stack(signal, axis=0).astype(np.float32)

        samples.append({
            "signal": signal,
            "channel_names": base_channels,
            "sfreq": sfreq,
        })

    return samples
