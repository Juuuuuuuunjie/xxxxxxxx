import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_eeg
from targets import compute_token_bandpower_trajectory_targets


class EEGPretrainDataset(Dataset):
    """
    EEG 预训练数据集。

    samples 中每个样本格式:

    {
        "signal": np.ndarray [C_in, T],
        "channel_names": list[str],
        "sfreq": int
    }

    返回:
        token_inputs: [N, 64 * L]
        targets: [N, 64, B, K]
        channel_valid_mask: [64]
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

        signal_64, valid_mask = preprocess_eeg(
            signal=signal,
            channel_names=channel_names,
            orig_sfreq=sfreq,
            target_sfreq=self.cfg.data.target_sfreq,
        )

        windows, targets = compute_token_bandpower_trajectory_targets(
            signal=signal_64,
            sfreq=self.cfg.data.target_sfreq,
            window_seconds=self.cfg.data.window_seconds,
            stride_seconds=self.cfg.data.window_stride_seconds,
            target_frame_seconds=self.cfg.data.target_frame_seconds,
            stft_window_seconds=self.cfg.data.stft_window_seconds,
            stft_hop_seconds=self.cfg.data.stft_hop_seconds,
            band_defs=self.cfg.data.band_defs,
            eps=self.cfg.data.eps,
        )

        # windows: [N, 64, L]
        # targets: [N, 64, B, K]
        N, C, L = windows.shape
        token_inputs = windows.reshape(N, C * L).astype(np.float32)

        return {
            "token_inputs": torch.tensor(token_inputs, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "channel_valid_mask": torch.tensor(valid_mask, dtype=torch.float32),
        }


def collate_fn(batch):
    """
    当前最小版本假设 batch 内所有样本 token 数量一致。

    Returns:
        token_inputs: [B, N, D]
        targets: [B, N, C, BANDS, K]
        channel_valid_mask: [B, C]
    """
    token_inputs = torch.stack([x["token_inputs"] for x in batch], dim=0)
    targets = torch.stack([x["targets"] for x in batch], dim=0)
    channel_valid_mask = torch.stack([x["channel_valid_mask"] for x in batch], dim=0)

    return {
        "token_inputs": token_inputs,
        "targets": targets,
        "channel_valid_mask": channel_valid_mask,
    }


def build_mock_samples(num_samples=32, clip_seconds=10, sfreq=100):
    """
    生成可跑通流程的模拟 EEG 数据。

    真实使用时，可以替换成从 EDF / FIF / MAT / NPY 中读取的真实 EEG。
    """
    samples = []
    T = clip_seconds * sfreq

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
            # 构造带有时间变化的模拟 EEG
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
