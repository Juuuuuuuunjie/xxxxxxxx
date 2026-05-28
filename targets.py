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


def compute_channel_time_patch_targets(
    signal: np.ndarray,
    sfreq: int,
    patch_seconds: float,
    patch_stride_seconds: float,
    target_frame_seconds: float,
    stft_window_seconds: float,
    stft_hop_seconds: float,
    band_defs,
    eps: float = 1e-6,
):
    """
    生成 channel-time patch token 和对应时频轨迹 target。

    输入:
        signal: [C, T]

    输出:
        patches: [C, N, L]
            每个通道上的时间 patch。

        targets: [C, N, F, K]
            每个 channel-time patch 对应的频段能量时间轨迹。

    例如:
        signal: [64, 1024]
        patch_len: 128
        N = 8

        patches: [64, 8, 128]
        flatten 后就是 [512, 128]
    """
    C, T = signal.shape

    patch_len = int(round(patch_seconds * sfreq))
    patch_stride = int(round(patch_stride_seconds * sfreq))

    patches = make_channel_time_patches(
        signal=signal,
        sfreq=sfreq,
        patch_seconds=patch_seconds,
        patch_stride_seconds=patch_stride_seconds,
    )

    C, N, L = patches.shape
    F = len(band_defs)
    K = int(round(patch_seconds / target_frame_seconds))

    if N == 0:
        return patches, np.zeros((C, 0, F, K), dtype=np.float32)

    continuous_band_power, stft_times = compute_continuous_bandpower_trajectory(
        signal=signal,
        sfreq=sfreq,
        band_defs=band_defs,
        stft_window_seconds=stft_window_seconds,
        stft_hop_seconds=stft_hop_seconds,
        eps=eps,
    )

    # continuous_band_power: [C, F, S]
    # stft_times: [S]

    targets = np.zeros((C, N, F, K), dtype=np.float32)

    for c in range(C):
        for n in range(N):
            patch_start_time = n * patch_stride_seconds

            # patch 内部的 K 个目标时间点
            target_times = patch_start_time + (np.arange(K) + 0.5) * target_frame_seconds

            for f in range(F):
                targets[c, n, f, :] = np.interp(
                    target_times,
                    stft_times,
                    continuous_band_power[c, f, :],
                    left=continuous_band_power[c, f, 0],
                    right=continuous_band_power[c, f, -1],
                )

    return patches.astype(np.float32), targets.astype(np.float32)


class RunningTargetNormalizer:
    """
    target shape:
        [C, N, F, K]

    统计:
        mean: [C, F]
        std:  [C, F]

    也就是对:
        - 时间 patch 维 N
        - patch 内部时间帧 K

    做统计，保留通道和频段维度。
    """
    def __init__(self, n_channels: int, n_bands: int, eps: float = 1e-6):
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.eps = eps

        self.mean = np.zeros((n_channels, n_bands), dtype=np.float64)
        self.var = np.ones((n_channels, n_bands), dtype=np.float64)
        self.count = 0

    def fit_batch(self, targets: np.ndarray):
        """
        Args:
            targets: [C, N, F, K]
        """
        if targets.shape[1] == 0:
            return

        # [C, N, F, K] -> [C, F]
        batch_mean = targets.mean(axis=(1, 3))
        batch_var = targets.var(axis=(1, 3))
        batch_count = targets.shape[1] * targets.shape[3]

        if self.count == 0:
            self.mean = batch_mean.astype(np.float64)
            self.var = batch_var.astype(np.float64)
            self.count = batch_count
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / total

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total

        new_var = m2 / total

        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, targets: np.ndarray):
        """
        Args:
            targets: [C, N, F, K]

        Returns:
            normalized: [C, N, F, K]
        """
        mean = self.mean.astype(np.float32)[:, None, :, None]
        std = np.sqrt(self.var.astype(np.float32) + self.eps)[:, None, :, None]

        return (targets - mean) / std
