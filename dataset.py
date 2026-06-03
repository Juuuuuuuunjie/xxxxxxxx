import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import spectrogram

from preprocessing import preprocess_eeg


def _next_power_of_two(x: int):
    n = 1
    while n < x:
        n *= 2
    return n


def cut_signal_patches(
    signal: np.ndarray,
    sfreq: int,
    patch_seconds: float,
    patch_stride_seconds: float,
):
    """
    从原始信号切 patch。

    Args:
        signal: [C, T]

    Returns:
        patches: [C, N, L]
    """
    C, T = signal.shape

    patch_len = int(round(patch_seconds * sfreq))
    patch_stride = int(round(patch_stride_seconds * sfreq))

    N = compute_num_patches(
        total_points=T,
        patch_len=patch_len,
        patch_stride=patch_stride,
    )

    if N == 0:
        return np.zeros((C, 0, patch_len), dtype=np.float32)

    patches = np.zeros((C, N, patch_len), dtype=np.float32)

    for n in range(N):
        start = n * patch_stride
        end = start + patch_len
        patches[:, n, :] = signal[:, start:end]

    return patches.astype(np.float32)


def generate_targets_from_timefreq(
    timefreq_repr: np.ndarray,
    sfreq: int,
    patch_seconds: float,
    patch_stride_seconds: float,
    target_frame_seconds: float,
):
    """
    从时频表示生成 target。

    Args:
        timefreq_repr: [C, F, T]

    Returns:
        targets: [C, N, F, K]
    """
    C, F, T = timefreq_repr.shape

    patch_len = int(round(patch_seconds * sfreq))
    patch_stride = int(round(patch_stride_seconds * sfreq))

    N = compute_num_patches(
        total_points=T,
        patch_len=patch_len,
        patch_stride=patch_stride,
    )

    K = int(round(patch_seconds / target_frame_seconds))

    if N == 0:
        return np.zeros((C, 0, F, K), dtype=np.float32)

    targets = np.zeros((C, N, F, K), dtype=np.float32)

    for n in range(N):
        start = n * patch_stride
        end = start + patch_len

        target_indices = np.linspace(start, end - 1, K).astype(int)
        targets[:, n, :, :] = timefreq_repr[:, :, target_indices]

    return targets.astype(np.float32)


def compute_num_patches(
    total_points: int,
    patch_len: int,
    patch_stride: int,
):
    if total_points < patch_len:
        return 0
    return int(np.floor((total_points - patch_len) / patch_stride) + 1)


def compute_continuous_bandpower_trajectory(
    signal: np.ndarray,
    sfreq: int,
    band_defs,
    stft_window_seconds: float,
    stft_hop_seconds: float,
    eps: float = 1e-6,
):
    """
    对整段 EEG 计算连续的频段能量轨迹。

    Args:
        signal: [C, T]

    Returns:
        band_power: [C, F, S]
        stft_times: [S]
    """
    nperseg = int(round(stft_window_seconds * sfreq))
    hop = int(round(stft_hop_seconds * sfreq))
    noverlap = max(0, nperseg - hop)
    nfft = _next_power_of_two(nperseg)

    freqs, stft_times, psd = spectrogram(
        signal,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="density",
        mode="psd",
        axis=1,
    )

    band_list = []

    for _, f_low, f_high in band_defs:
        freq_idx = (freqs >= f_low) & (freqs < f_high)

        if freq_idx.sum() == 0:
            one_band = np.zeros((signal.shape[0], psd.shape[-1]), dtype=np.float32)
        else:
            one_band = psd[:, freq_idx, :].mean(axis=1)

        one_band = np.log(one_band + eps).astype(np.float32)
        band_list.append(one_band)

    band_power = np.stack(band_list, axis=0)
    band_power = np.transpose(band_power, (1, 0, 2)).astype(np.float32)

    return band_power, stft_times.astype(np.float32)


def compute_continuous_timefreq_representation(
    signal: np.ndarray,
    sfreq: int,
    band_defs,
    stft_window_seconds: float,
    stft_hop_seconds: float,
    target_sfreq: int,
    eps: float = 1e-6,
):
    """
    计算连续的时频表示，并插值回原始时间分辨率。

    Args:
        signal: [C, T]

    Returns:
        timefreq: [C, F, T]
    """
    band_power, stft_times = compute_continuous_bandpower_trajectory(
        signal=signal,
        sfreq=sfreq,
        band_defs=band_defs,
        stft_window_seconds=stft_window_seconds,
        stft_hop_seconds=stft_hop_seconds,
        eps=eps,
    )

    C, F, S = band_power.shape
    T = signal.shape[1]

    original_times = np.arange(T) / target_sfreq

    timefreq = np.zeros((C, F, T), dtype=np.float32)

    for c in range(C):
        for f in range(F):
            timefreq[c, f, :] = np.interp(
                original_times,
                stft_times,
                band_power[c, f, :],
                left=band_power[c, f, 0],
                right=band_power[c, f, -1],
            )

    return timefreq.astype(np.float32)


def standardize_signal(signal: np.ndarray, eps: float = 1e-8):
    """
    对原始信号做标准化。

    Args:
        signal: [C, T]

    Returns:
        standardized: [C, T]
    """
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return ((signal - mean) / (std + eps)).astype(np.float32)


def standardize_timefreq(timefreq: np.ndarray, eps: float = 1e-8):
    """
    对时频表示做标准化。

    Args:
        timefreq: [C, F, T]

    Returns:
        standardized: [C, F, T]
    """
    mean = timefreq.mean(axis=2, keepdims=True)
    std = timefreq.std(axis=2, keepdims=True)
    return ((timefreq - mean) / (std + eps)).astype(np.float32)


def preprocess_all_samples(raw_samples, cfg):
    """
    离线预处理所有原始样本，返回可直接喂给 Dataset 的列表。

    每个原始 sample 是一个 clip（已经按 clip_seconds 切好的片段）：
        {
            "signal": np.ndarray [C_in, T_in],
            "channel_names": list[str],
            "sfreq": float,
        }

    处理流程：
        1. preprocess_eeg → [64, T]（对齐通道、重采样、padding）
        2. standardize_signal
        3. compute_continuous_timefreq_representation → [64, F, T]
        4. standardize_timefreq
        5. cut_signal_patches → [64, N, L]
        6. generate_targets_from_timefreq → [64, N, F, K]
        7. flatten (通道 × 时间patch) → token 序列

    返回：
        processed: list of dict，每个 dict 包含：
            token_inputs:          np.ndarray [S, L]        S = C*N
            targets:               np.ndarray [S, F, K]
            token_channel_indices: np.ndarray [S]           int64
            token_time_indices:    np.ndarray [S]           int64
            token_valid_mask:      np.ndarray [S]           float32
            channel_valid_mask:    np.ndarray [C]           float32
    """
    target_num_points = int(round(cfg.data.clip_seconds * cfg.data.target_sfreq))
    processed = []

    for i, item in enumerate(raw_samples):
        signal = item["signal"]
        channel_names = item["channel_names"]
        sfreq = item["sfreq"]

        # 1) 对齐通道、重采样、padding → [64, T]
        signal_64, channel_valid_mask = preprocess_eeg(
            signal=signal,
            channel_names=channel_names,
            orig_sfreq=sfreq,
            target_sfreq=cfg.data.target_sfreq,
            target_num_points=target_num_points,
        )

        # 2) 标准化原始信号
        signal_64 = standardize_signal(signal_64)

        # 3) 计算连续时频表示 → [64, F, T]
        timefreq_repr = compute_continuous_timefreq_representation(
            signal=signal_64,
            sfreq=cfg.data.target_sfreq,
            band_defs=cfg.data.band_defs,
            stft_window_seconds=cfg.data.stft_window_seconds,
            stft_hop_seconds=cfg.data.stft_hop_seconds,
            target_sfreq=cfg.data.target_sfreq,
            eps=cfg.data.eps,
        )

        # 4) 标准化时频表示
        timefreq_repr = standardize_timefreq(timefreq_repr)

        # 5) 切 patch → [C, N, L]
        signal_patches = cut_signal_patches(
            signal=signal_64,
            sfreq=cfg.data.target_sfreq,
            patch_seconds=cfg.data.patch_seconds,
            patch_stride_seconds=cfg.data.patch_stride_seconds,
        )

        # 6) 生成 target → [C, N, F, K]
        targets = generate_targets_from_timefreq(
            timefreq_repr=timefreq_repr,
            sfreq=cfg.data.target_sfreq,
            patch_seconds=cfg.data.patch_seconds,
            patch_stride_seconds=cfg.data.patch_stride_seconds,
            target_frame_seconds=cfg.data.target_frame_seconds,
        )

        C, N, L = signal_patches.shape
        _, _, F, K = targets.shape

        # 7) flatten → token 序列 [S, L], S = C*N
        token_inputs = signal_patches.reshape(C * N, L).astype(np.float32)
        token_targets = targets.reshape(C * N, F, K).astype(np.float32)

        token_channel_indices = np.repeat(np.arange(C), N).astype(np.int64)
        token_time_indices = np.tile(np.arange(N), C).astype(np.int64)
        token_valid_mask = channel_valid_mask[token_channel_indices].astype(np.float32)

        processed.append({
            "token_inputs": token_inputs,
            "targets": token_targets,
            "token_channel_indices": token_channel_indices,
            "token_time_indices": token_time_indices,
            "token_valid_mask": token_valid_mask,
            "channel_valid_mask": channel_valid_mask.astype(np.float32),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(raw_samples):
            print(f"  preprocessed {i + 1}/{len(raw_samples)} clips")

    return processed


class EEGPretrainDataset(Dataset):
    """
    EEG 预训练数据集。

    接收已经离线预处理好的样本列表（由 preprocess_all_samples 生成）。
    __getitem__ 只做 numpy → tensor 转换，不做任何计算。

    输出 batch（经 collate_fn）:
        token_inputs:          [B, S, L]
        targets:               [B, S, F, K]
        token_channel_indices: [B, S]
        token_time_indices:    [B, S]
        token_valid_mask:      [B, S]
        channel_valid_mask:    [B, C]
    """

    def __init__(self, processed_samples):
        self.samples = processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        return {
            "token_inputs":          torch.tensor(item["token_inputs"],          dtype=torch.float32),
            "targets":               torch.tensor(item["targets"],               dtype=torch.float32),
            "token_channel_indices": torch.tensor(item["token_channel_indices"], dtype=torch.long),
            "token_time_indices":    torch.tensor(item["token_time_indices"],    dtype=torch.long),
            "token_valid_mask":      torch.tensor(item["token_valid_mask"],      dtype=torch.float32),
            "channel_valid_mask":    torch.tensor(item["channel_valid_mask"],    dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "token_inputs":          torch.stack([x["token_inputs"]          for x in batch], dim=0),
        "targets":               torch.stack([x["targets"]               for x in batch], dim=0),
        "token_channel_indices": torch.stack([x["token_channel_indices"] for x in batch], dim=0),
        "token_time_indices":    torch.stack([x["token_time_indices"]    for x in batch], dim=0),
        "token_valid_mask":      torch.stack([x["token_valid_mask"]      for x in batch], dim=0),
        "channel_valid_mask":    torch.stack([x["channel_valid_mask"]    for x in batch], dim=0),
    }
