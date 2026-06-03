import numpy as np
from scipy.signal import spectrogram


def _next_power_of_two(x: int):
    n = 1
    while n < x:
        n *= 2
    return n


def compute_num_patches(
    total_points: int,
    patch_len: int,
    patch_stride: int,
):
    if total_points < patch_len:
        return 0
    return int(np.floor((total_points - patch_len) / patch_stride) + 1)


def make_channel_time_patches(
    signal: np.ndarray,
    sfreq: int,
    patch_seconds: float,
    patch_stride_seconds: float,
):
    """
    把 EEG 按通道和时间切成 ViT-style tokens。

    Args:
        signal: [C, T]
        sfreq: 采样率
        patch_seconds: 每个 token 的时间长度
        patch_stride_seconds: token 时间步长

    Returns:
        patches: [C, N, L]

    其中:
        C = 通道数
        N = 每个通道上的时间 patch 数
        L = 每个 patch 的采样点数
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
            C = 通道数
            F = 频段数
            S = STFT 时间帧数

        stft_times: [S]
            每个 STFT 帧对应的时间中心，单位秒
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

    # psd: [C, freq_bins, stft_frames]
    band_list = []

    for _, f_low, f_high in band_defs:
        freq_idx = (freqs >= f_low) & (freqs < f_high)

        if freq_idx.sum() == 0:
            one_band = np.zeros((signal.shape[0], psd.shape[-1]), dtype=np.float32)
        else:
            # 对频段内部频率 bin 平均，但保留时间维度
            one_band = psd[:, freq_idx, :].mean(axis=1)

        # log power，减小不同频段数值尺度差异
        one_band = np.log(one_band + eps).astype(np.float32)

        band_list.append(one_band)

    # [F, C, S] -> [C, F, S]
    band_power = np.stack(band_list, axis=0)
    band_power = np.transpose(band_power, (1, 0, 2)).astype(np.float32)

    return band_power, stft_times.astype(np.float32)


