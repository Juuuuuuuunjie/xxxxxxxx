import numpy as np
from scipy.signal import spectrogram


def window_signal(signal: np.ndarray, sfreq: int, window_seconds: float, stride_seconds: float):
    """
    按时间窗切分 EEG。

    Args:
        signal: [C, T]
        sfreq: 采样率
        window_seconds: 每个 token 的长度，单位秒
        stride_seconds: token 步长，单位秒

    Returns:
        windows: [N, C, L]
    """
    C, T = signal.shape
    win_len = int(round(window_seconds * sfreq))
    stride = int(round(stride_seconds * sfreq))

    windows = []
    for start in range(0, T - win_len + 1, stride):
        end = start + win_len
        windows.append(signal[:, start:end])

    if len(windows) == 0:
        return np.zeros((0, C, win_len), dtype=np.float32)

    return np.stack(windows, axis=0).astype(np.float32)


def _next_power_of_two(x: int):
    """
    返回不小于 x 的 2 的幂，用作 nfft。
    """
    n = 1
    while n < x:
        n *= 2
    return n


def compute_continuous_bandpower_trajectory(
    signal: np.ndarray,
    sfreq: int,
    band_defs,
    stft_window_seconds: float,
    stft_hop_seconds: float,
    eps: float = 1e-6,
):
    """
    对整个 EEG clip 计算连续的频段能量轨迹。

    这里的目标是保留频段能量随时间变化的信息，而不是只求平均能量。

    Args:
        signal: [C, T]
        sfreq: 采样率
        band_defs: 频段定义，例如 [("alpha", 8, 13), ...]
        stft_window_seconds: STFT 窗长
        stft_hop_seconds: STFT hop
        eps: 防止 log(0)

    Returns:
        band_power: [C, B, S]
            C = 通道数
            B = 频段数
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

    # psd: [C, F, S]
    band_features = []

    for _, f_low, f_high in band_defs:
        freq_idx = (freqs >= f_low) & (freqs < f_high)

        if freq_idx.sum() == 0:
            band_power = np.zeros((signal.shape[0], psd.shape[-1]), dtype=np.float32)
        else:
            # 对频段内部的频率 bin 求平均，但保留时间维度
            band_power = psd[:, freq_idx, :].mean(axis=1)

        # log-power，更稳定，也能减轻不同频段尺度差异
        band_power = np.log(band_power + eps)
        band_features.append(band_power.astype(np.float32))

    # [B, C, S] -> [C, B, S]
    band_features = np.stack(band_features, axis=0)
    band_features = np.transpose(band_features, (1, 0, 2)).astype(np.float32)

    return band_features, stft_times.astype(np.float32)


def compute_token_bandpower_trajectory_targets(
    signal: np.ndarray,
    sfreq: int,
    window_seconds: float,
    stride_seconds: float,
    target_frame_seconds: float,
    stft_window_seconds: float,
    stft_hop_seconds: float,
    band_defs,
    eps: float = 1e-6,
):
    """
    生成每个 token 对应的频段能量时间轨迹目标。

    输入:
        signal: [C, T]

    输出:
        windows: [N, C, L]
            模型输入用的时域 EEG token。

        targets: [N, C, B, K]
            每个 token 内部的频段能量时间轨迹。
            N = token 数量
            C = 通道数
            B = 频段数
            K = token 内部目标时间点数量

    设计重点:
        - 频率维度在每个频段内部平均
        - 时间维度保留 K 个点
        - 因此模型必须重构频段能量随时间变化的轨迹
    """
    windows = window_signal(
        signal=signal,
        sfreq=sfreq,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )

    N = windows.shape[0]
    C = signal.shape[0]
    B = len(band_defs)
    K = int(round(window_seconds / target_frame_seconds))

    if N == 0:
        return windows, np.zeros((0, C, B, K), dtype=np.float32)

    continuous_band_power, stft_times = compute_continuous_bandpower_trajectory(
        signal=signal,
        sfreq=sfreq,
        band_defs=band_defs,
        stft_window_seconds=stft_window_seconds,
        stft_hop_seconds=stft_hop_seconds,
        eps=eps,
    )

    # continuous_band_power: [C, B, S]
    # stft_times: [S]

    targets = np.zeros((N, C, B, K), dtype=np.float32)

    for token_idx in range(N):
        token_start_time = token_idx * stride_seconds

        # token 内部的 K 个目标时间点
        # 使用中心点，例如 0.05, 0.15, ..., 0.95
        target_times = token_start_time + (np.arange(K) + 0.5) * target_frame_seconds

        for c in range(C):
            for b in range(B):
                targets[token_idx, c, b, :] = np.interp(
                    target_times,
                    stft_times,
                    continuous_band_power[c, b, :],
                    left=continuous_band_power[c, b, 0],
                    right=continuous_band_power[c, b, -1],
                )

    return windows.astype(np.float32), targets.astype(np.float32)


class RunningTargetNormalizer:
    """
    用训练集 target 统计均值和方差。

    当前 target shape:
        [N, C, B, K]

    这里统计:
        mean: [C, B]
        std:  [C, B]

    也就是说:
        - 对所有 token 求平均
        - 对 token 内部的所有时间点 K 求平均
        - 保留 channel 和 band 维度

    这样做可以:
        - 保留频段能量随时间的变化
        - 避免每个时间点被单独归一化掉
        - 缓解低频功率主导 loss 的问题
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
            targets: [N, C, B, K]
        """
        if targets.shape[0] == 0:
            return

        # [N, C, B, K] -> 对 N 和 K 统计
        batch_mean = targets.mean(axis=(0, 3))  # [C, B]
        batch_var = targets.var(axis=(0, 3))    # [C, B]
        batch_count = targets.shape[0] * targets.shape[3]

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
            targets: [N, C, B, K]

        Returns:
            normalized_targets: [N, C, B, K]
        """
        mean = self.mean.astype(np.float32)[None, :, :, None]
        std = np.sqrt(self.var.astype(np.float32) + self.eps)[None, :, :, None]
        return (targets - mean) / std
