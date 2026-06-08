import os
import json
import math
import shutil
from typing import Dict, List, Tuple, Any

import numpy as np
import pyarrow as pa
import lance

from scipy.signal import spectrogram, resample_poly

from configs import Config


# =============================================================================
# 0. 从 configs.py 读取配置
# =============================================================================

CFG = Config()
DATA_CFG = CFG.data


# =============================================================================
# 1. 路径配置
# =============================================================================

# 原始 Lance 路径。
# 这个是服务器原始数据 Lance，不是 processed Lance。
RAW_LANCE_PATH = "./serverData/eeg_openneuro_76ch_val.lance"

# 输出 processed Lance 路径。
# 这里使用 configs.py 里的 data.lance_path。
# 确保 train_real_set.py 里也读取同一个 cfg.data.lance_path。
OUTPUT_LANCE_PATH = DATA_CFG.lance_path

# electrode id -> channel name 的 json 路径。
ELECTRODE_VOCAB_PATH = "./electrode_vocab.json"


# =============================================================================
# 2. 原始 Lance 相关配置
# =============================================================================

# 如果原始 Lance signals 是除以 200 后的值，则这里设 True。
DENORM_CLIP_DIVIDE_200 = True

# 原始服务器数据采样率。
# 注意：你的 configs.py 里目前没有 orig_sfreq 字段，所以这里仍然保留本地配置。
# 如果你想完全统一，也可以在 DataConfig 里加：
# orig_sfreq: float = 200.0
ORIG_SFREQ = 200.0

# 离线转换 batch size。
BATCH_SIZE = 64

# 是否覆盖已有 output lance。
OVERWRITE = True

# 遇到 bad_channel_mask=True 的通道，是否跳过。
DROP_BAD_CHANNELS = True


# =============================================================================
# 3. 从 configs.py 读取数据处理参数
# =============================================================================

CONVERT_V_TO_UV = DATA_CFG.convert_v_to_uv

CLIP_SECONDS = DATA_CFG.clip_seconds
CLIP_STRIDE_SECONDS = DATA_CFG.clip_stride_seconds

TARGET_SFREQ = DATA_CFG.target_sfreq

N_CHANNELS = DATA_CFG.n_channels

PATCH_SECONDS = DATA_CFG.patch_seconds
PATCH_STRIDE_SECONDS = DATA_CFG.patch_stride_seconds

TARGET_FRAME_SECONDS = DATA_CFG.target_frame_seconds

STFT_WINDOW_SECONDS = DATA_CFG.stft_window_seconds
STFT_HOP_SECONDS = DATA_CFG.stft_hop_seconds

BAND_DEFS = DATA_CFG.band_defs
N_BANDS = len(BAND_DEFS)

EPS = DATA_CFG.eps


# =============================================================================
# 4. 派生 shape 参数
# =============================================================================

TOTAL_POINTS = int(round(CLIP_SECONDS * TARGET_SFREQ))

PATCH_LEN = int(round(PATCH_SECONDS * TARGET_SFREQ))
PATCH_STRIDE = int(round(PATCH_STRIDE_SECONDS * TARGET_SFREQ))

NUM_TIME_PATCHES = int(np.floor((TOTAL_POINTS - PATCH_LEN) / PATCH_STRIDE) + 1)
NUM_TOKENS = N_CHANNELS * NUM_TIME_PATCHES

# 关键：target 的最后一维 K 来自 TARGET_FRAME_SECONDS，而不是 STFT_HOP_SECONDS。
FRAMES_PER_PATCH = int(round(PATCH_SECONDS / TARGET_FRAME_SECONDS))

STFT_WINDOW_POINTS = int(round(STFT_WINDOW_SECONDS * TARGET_SFREQ))
STFT_HOP_POINTS = int(round(STFT_HOP_SECONDS * TARGET_SFREQ))


print("\n===== Build Processed Lance Config From configs.py =====")
print("RAW_LANCE_PATH:", RAW_LANCE_PATH)
print("OUTPUT_LANCE_PATH:", OUTPUT_LANCE_PATH)
print("ELECTRODE_VOCAB_PATH:", ELECTRODE_VOCAB_PATH)

print("\n===== Raw Lance Config =====")
print("ORIG_SFREQ:", ORIG_SFREQ)
print("DENORM_CLIP_DIVIDE_200:", DENORM_CLIP_DIVIDE_200)
print("DROP_BAD_CHANNELS:", DROP_BAD_CHANNELS)

print("\n===== DataConfig =====")
print("CONVERT_V_TO_UV:", CONVERT_V_TO_UV)
print("CLIP_SECONDS:", CLIP_SECONDS)
print("CLIP_STRIDE_SECONDS:", CLIP_STRIDE_SECONDS)
print("TARGET_SFREQ:", TARGET_SFREQ)
print("N_CHANNELS:", N_CHANNELS)
print("PATCH_SECONDS:", PATCH_SECONDS)
print("PATCH_STRIDE_SECONDS:", PATCH_STRIDE_SECONDS)
print("TARGET_FRAME_SECONDS:", TARGET_FRAME_SECONDS)
print("STFT_WINDOW_SECONDS:", STFT_WINDOW_SECONDS)
print("STFT_HOP_SECONDS:", STFT_HOP_SECONDS)
print("BAND_DEFS:", BAND_DEFS)
print("N_BANDS:", N_BANDS)
print("EPS:", EPS)

print("\n===== Derived Shape Params =====")
print("TOTAL_POINTS:", TOTAL_POINTS)
print("PATCH_LEN:", PATCH_LEN)
print("PATCH_STRIDE:", PATCH_STRIDE)
print("NUM_TIME_PATCHES:", NUM_TIME_PATCHES)
print("NUM_TOKENS:", NUM_TOKENS)
print("FRAMES_PER_PATCH:", FRAMES_PER_PATCH)
print("STFT_WINDOW_POINTS:", STFT_WINDOW_POINTS)
print("STFT_HOP_POINTS:", STFT_HOP_POINTS)


assert N_CHANNELS == 64, f"Current script expects 64 standard channels, got {N_CHANNELS}"
assert TOTAL_POINTS > 0
assert PATCH_LEN > 0
assert PATCH_STRIDE > 0
assert TOTAL_POINTS >= PATCH_LEN
assert NUM_TIME_PATCHES > 0
assert NUM_TOKENS == N_CHANNELS * NUM_TIME_PATCHES
assert FRAMES_PER_PATCH > 0
assert N_BANDS == len(BAND_DEFS)
assert abs(FRAMES_PER_PATCH * TARGET_FRAME_SECONDS - PATCH_SECONDS) < 1e-6, (
    f"PATCH_SECONDS={PATCH_SECONDS} cannot be evenly divided by "
    f"TARGET_FRAME_SECONDS={TARGET_FRAME_SECONDS}"
)


# =============================================================================
# 5. 标准 64 通道模板
# =============================================================================

STANDARD_64_CHANNELS = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
    "A1", "A2",
]

assert len(STANDARD_64_CHANNELS) == 64, (
    f"STANDARD_64_CHANNELS length = {len(STANDARD_64_CHANNELS)}, not 64"
)


# =============================================================================
# 6. 通道名处理
# =============================================================================

def load_electrode_vocab(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    id2name = {}
    for k, v in raw.items():
        id2name[int(k)] = str(v)

    return id2name


def normalize_channel_name(name: str) -> str:
    if name is None:
        return "UNK_EEG"

    name = str(name).strip()

    if name.startswith("EEG "):
        name = name[4:].strip()

    for suffix in ["-REF", "-LE", "-A1", "-A2", "-M1", "-M2"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    upper_alias = {
        "FP1": "Fp1",
        "FPZ": "Fpz",
        "FP2": "Fp2",

        "AF7": "AF7",
        "AF3": "AF3",
        "AFZ": "AFz",
        "AF4": "AF4",
        "AF8": "AF8",

        "F7": "F7",
        "F5": "F5",
        "F3": "F3",
        "F1": "F1",
        "FZ": "Fz",
        "F2": "F2",
        "F4": "F4",
        "F6": "F6",
        "F8": "F8",

        "FT7": "FT7",
        "FC5": "FC5",
        "FC3": "FC3",
        "FC1": "FC1",
        "FCZ": "FCz",
        "FC2": "FC2",
        "FC4": "FC4",
        "FC6": "FC6",
        "FT8": "FT8",

        "T7": "T7",
        "C5": "C5",
        "C3": "C3",
        "C1": "C1",
        "CZ": "Cz",
        "C2": "C2",
        "C4": "C4",
        "C6": "C6",
        "T8": "T8",

        "TP7": "TP7",
        "CP5": "CP5",
        "CP3": "CP3",
        "CP1": "CP1",
        "CPZ": "CPz",
        "CP2": "CP2",
        "CP4": "CP4",
        "CP6": "CP6",
        "TP8": "TP8",

        "P7": "P7",
        "P5": "P5",
        "P3": "P3",
        "P1": "P1",
        "PZ": "Pz",
        "P2": "P2",
        "P4": "P4",
        "P6": "P6",
        "P8": "P8",

        "PO7": "PO7",
        "PO3": "PO3",
        "POZ": "POz",
        "PO4": "PO4",
        "PO8": "PO8",

        "O1": "O1",
        "OZ": "Oz",
        "O2": "O2",
        "IZ": "Iz",

        "A1": "A1",
        "A2": "A2",
        "M1": "A1",
        "M2": "A2",
    }

    old_1020_alias = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }

    name_upper = name.upper()

    if name_upper in old_1020_alias:
        return old_1020_alias[name_upper]

    if name_upper in upper_alias:
        return upper_alias[name_upper]

    return name


def electrode_ids_to_channel_names(
    electrode_ids: np.ndarray,
    id2name: Dict[int, str],
) -> List[str]:
    channel_names = []

    for eid in electrode_ids.tolist():
        eid = int(eid)

        if eid == 0:
            name = "PAD"
        else:
            name = id2name.get(eid, "UNK_EEG")

        name = normalize_channel_name(name)
        channel_names.append(name)

    return channel_names


# =============================================================================
# 7. 原始 signals 解析与通道映射
# =============================================================================

def parse_raw_lance_row(
    row: Dict[str, Any],
    id2name: Dict[int, str],
    orig_sfreq: float,
    denorm_clip_divide_200: bool = True,
    drop_bad_channels: bool = True,
) -> Dict[str, Any]:

    n_channels = int(row["channel_counts"])

    signals_flat = np.asarray(row["signals"], dtype=np.float32)

    if signals_flat.size % n_channels != 0:
        raise ValueError(
            f"signals length {signals_flat.size} cannot be divided by "
            f"channel_counts {n_channels}"
        )

    seq_len = signals_flat.size // n_channels
    signal = signals_flat.reshape(n_channels, seq_len).astype(np.float32)

    if denorm_clip_divide_200:
        signal = signal * 200.0

    electrode_ids = np.asarray(row["electrode_ids"], dtype=np.int64)[:n_channels]
    channel_names = electrode_ids_to_channel_names(electrode_ids, id2name)

    if "bad_channel_mask" in row and row["bad_channel_mask"] is not None:
        bad_channel_mask = np.asarray(row["bad_channel_mask"], dtype=bool)[:n_channels]
    else:
        bad_channel_mask = np.zeros(n_channels, dtype=bool)

    if drop_bad_channels:
        good_mask = ~bad_channel_mask
        signal = signal[good_mask]
        channel_names = [
            name for name, good in zip(channel_names, good_mask.tolist())
            if good
        ]

    return {
        "signal": signal,
        "channel_names": channel_names,
        "sfreq": float(orig_sfreq),

        "sample_id": row.get("sample_id", -1),
        "subject_id": row.get("subject_id", ""),
        "edf_relpath": row.get("edf_relpath", ""),
        "segment_start_sec": row.get("segment_start_sec", 0.0),
        "split": row.get("split", ""),
        "shard_name": row.get("shard_name", ""),
        "global_idx": row.get("global_idx", -1),
        "dataset_key": row.get("dataset_key", ""),
    }


def align_to_standard_channels(
    signal: np.ndarray,
    channel_names: List[str],
    standard_channels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:

    C_in, T = signal.shape
    C_std = len(standard_channels)

    aligned_sum = np.zeros((C_std, T), dtype=np.float32)
    aligned_count = np.zeros(C_std, dtype=np.float32)

    std_name_to_idx = {
        normalize_channel_name(name): i
        for i, name in enumerate(standard_channels)
    }

    for src_idx, src_name in enumerate(channel_names):
        src_name = normalize_channel_name(src_name)

        if src_name in ["PAD", "UNK_EEG", ""]:
            continue

        if src_name not in std_name_to_idx:
            continue

        dst_idx = std_name_to_idx[src_name]

        aligned_sum[dst_idx] += signal[src_idx]
        aligned_count[dst_idx] += 1.0

    aligned_signal = np.zeros((C_std, T), dtype=np.float32)
    channel_valid_mask = np.zeros(C_std, dtype=np.float32)

    valid = aligned_count > 0

    aligned_signal[valid] = aligned_sum[valid] / aligned_count[valid, None]
    channel_valid_mask[valid] = 1.0

    return aligned_signal.astype(np.float32), channel_valid_mask.astype(np.float32)


# =============================================================================
# 8. 重采样、padding/crop、标准化
# =============================================================================

def resample_signal(
    signal: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float,
) -> np.ndarray:

    if abs(orig_sfreq - target_sfreq) < 1e-6:
        return signal.astype(np.float32)

    orig = int(round(orig_sfreq))
    target = int(round(target_sfreq))

    gcd = math.gcd(orig, target)
    up = target // gcd
    down = orig // gcd

    resampled = resample_poly(signal, up=up, down=down, axis=1)
    return resampled.astype(np.float32)


def pad_or_crop_signal(
    signal: np.ndarray,
    target_num_points: int,
) -> np.ndarray:

    C, T = signal.shape

    if T == target_num_points:
        return signal.astype(np.float32)

    if T > target_num_points:
        return signal[:, :target_num_points].astype(np.float32)

    out = np.zeros((C, target_num_points), dtype=np.float32)
    out[:, :T] = signal
    return out


def standardize_signal_by_channel(
    signal: np.ndarray,
    channel_valid_mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:

    out = np.zeros_like(signal, dtype=np.float32)

    valid = channel_valid_mask.astype(bool)

    if valid.sum() == 0:
        return out

    x = signal[valid]

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)

    out[valid] = (x - mean) / (std + eps)
    out[~valid] = 0.0

    return out.astype(np.float32)


# =============================================================================
# 9. 时频表示
# =============================================================================

def _next_power_of_two(x: int) -> int:
    n = 1
    while n < x:
        n *= 2
    return n


def compute_continuous_bandpower_trajectory(
    signal: np.ndarray,
    sfreq: float,
    band_defs,
    stft_window_seconds: float,
    stft_hop_seconds: float,
    eps: float = 1e-6,
):
    sfreq = float(sfreq)

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
    sfreq: float,
    band_defs,
    stft_window_seconds: float,
    stft_hop_seconds: float,
    target_sfreq: float,
    eps: float = 1e-6,
):
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

    original_times = np.arange(T, dtype=np.float32) / float(target_sfreq)

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


def standardize_timefreq(
    timefreq: np.ndarray,
    channel_valid_mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:

    out = np.zeros_like(timefreq, dtype=np.float32)

    valid = channel_valid_mask.astype(bool)

    if valid.sum() == 0:
        return out

    x = timefreq[valid]

    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True)

    out[valid] = (x - mean) / (std + eps)
    out[~valid] = 0.0

    return out.astype(np.float32)


# =============================================================================
# 10. patch 切割和 target 生成
# =============================================================================

def compute_num_patches(
    total_points: int,
    patch_len: int,
    patch_stride: int,
) -> int:
    if total_points < patch_len:
        return 0
    return int(np.floor((total_points - patch_len) / patch_stride) + 1)


def cut_signal_patches(
    signal: np.ndarray,
    sfreq: float,
    patch_seconds: float,
    patch_stride_seconds: float,
) -> np.ndarray:

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
    sfreq: float,
    patch_seconds: float,
    patch_stride_seconds: float,
    target_frame_seconds: float,
) -> np.ndarray:

    C, F, T = timefreq_repr.shape

    patch_len = int(round(patch_seconds * sfreq))
    patch_stride = int(round(patch_stride_seconds * sfreq))

    N = compute_num_patches(
        total_points=T,
        patch_len=patch_len,
        patch_stride=patch_stride,
    )

    # 关键：K 来自 target_frame_seconds。
    K = int(round(patch_seconds / target_frame_seconds))

    if N == 0:
        return np.zeros((C, 0, F, K), dtype=np.float32)

    targets = np.zeros((C, N, F, K), dtype=np.float32)

    # 在 patch 内取 K 个时间点。
    # patch_seconds=1.0, target_frame_seconds=0.2 -> K=5
    # patch_len=100 -> indices approximately [0, 24, 49, 74, 99]
    base_indices = np.linspace(0, patch_len - 1, K).astype(np.int64)

    for n in range(N):
        start = n * patch_stride
        target_indices = start + base_indices
        targets[:, n, :, :] = timefreq_repr[:, :, target_indices]

    return targets.astype(np.float32)


# =============================================================================
# 11. 单样本完整预处理
# =============================================================================

def preprocess_one_raw_sample(
    raw_sample: Dict[str, Any],
) -> Dict[str, Any]:

    signal = raw_sample["signal"]
    channel_names = raw_sample["channel_names"]
    orig_sfreq = float(raw_sample["sfreq"])

    target_num_points = TOTAL_POINTS

    signal_64, channel_valid_mask = align_to_standard_channels(
        signal=signal,
        channel_names=channel_names,
        standard_channels=STANDARD_64_CHANNELS,
    )

    signal_64 = resample_signal(
        signal=signal_64,
        orig_sfreq=orig_sfreq,
        target_sfreq=TARGET_SFREQ,
    )

    signal_64 = pad_or_crop_signal(
        signal=signal_64,
        target_num_points=target_num_points,
    )

    signal_64 = standardize_signal_by_channel(
        signal=signal_64,
        channel_valid_mask=channel_valid_mask,
        eps=1e-8,
    )

    timefreq_repr = compute_continuous_timefreq_representation(
        signal=signal_64,
        sfreq=TARGET_SFREQ,
        band_defs=BAND_DEFS,
        stft_window_seconds=STFT_WINDOW_SECONDS,
        stft_hop_seconds=STFT_HOP_SECONDS,
        target_sfreq=TARGET_SFREQ,
        eps=EPS,
    )

    timefreq_repr = standardize_timefreq(
        timefreq=timefreq_repr,
        channel_valid_mask=channel_valid_mask,
        eps=1e-8,
    )

    signal_patches = cut_signal_patches(
        signal=signal_64,
        sfreq=TARGET_SFREQ,
        patch_seconds=PATCH_SECONDS,
        patch_stride_seconds=PATCH_STRIDE_SECONDS,
    )

    targets = generate_targets_from_timefreq(
        timefreq_repr=timefreq_repr,
        sfreq=TARGET_SFREQ,
        patch_seconds=PATCH_SECONDS,
        patch_stride_seconds=PATCH_STRIDE_SECONDS,
        target_frame_seconds=TARGET_FRAME_SECONDS,
    )

    C, N, L = signal_patches.shape
    _, _, F, K = targets.shape

    token_inputs = signal_patches.reshape(C * N, L).astype(np.float32)
    token_targets = targets.reshape(C * N, F, K).astype(np.float32)

    token_channel_indices = np.repeat(np.arange(C), N).astype(np.int64)
    token_time_indices = np.tile(np.arange(N), C).astype(np.int64)

    token_valid_mask = channel_valid_mask[token_channel_indices].astype(np.float32)

    processed = {
        "token_inputs": token_inputs,
        "targets": token_targets,
        "token_channel_indices": token_channel_indices,
        "token_time_indices": token_time_indices,
        "token_valid_mask": token_valid_mask,
        "channel_valid_mask": channel_valid_mask.astype(np.float32),

        "num_tokens": int(C * N),
        "patch_len": int(L),
        "n_bands": int(F),
        "frames_per_patch": int(K),
        "n_channels": int(C),
        "num_patches": int(N),
    }

    # 强制检查，防止又生成旧 shape。
    assert processed["patch_len"] == PATCH_LEN, (
        f"patch_len mismatch: got {processed['patch_len']}, expected {PATCH_LEN}"
    )
    assert processed["frames_per_patch"] == FRAMES_PER_PATCH, (
        f"frames_per_patch mismatch: got {processed['frames_per_patch']}, "
        f"expected {FRAMES_PER_PATCH}"
    )
    assert processed["n_bands"] == N_BANDS, (
        f"n_bands mismatch: got {processed['n_bands']}, expected {N_BANDS}"
    )
    assert processed["num_tokens"] == NUM_TOKENS, (
        f"num_tokens mismatch: got {processed['num_tokens']}, expected {NUM_TOKENS}"
    )

    return processed


# =============================================================================
# 12. 写入 Lance 的 row 构造
# =============================================================================

def processed_to_lance_row(
    processed: Dict[str, Any],
    raw_sample: Dict[str, Any],
) -> Dict[str, Any]:

    token_inputs = processed["token_inputs"].astype(np.float32)
    targets = processed["targets"].astype(np.float32)

    token_channel_indices = processed["token_channel_indices"].astype(np.int64)
    token_time_indices = processed["token_time_indices"].astype(np.int64)
    token_valid_mask = processed["token_valid_mask"].astype(np.float32)
    channel_valid_mask = processed["channel_valid_mask"].astype(np.float32)

    row = {
        "num_tokens": int(processed["num_tokens"]),
        "patch_len": int(processed["patch_len"]),
        "n_bands": int(processed["n_bands"]),
        "frames_per_patch": int(processed["frames_per_patch"]),
        "n_channels": int(processed["n_channels"]),
        "num_patches": int(processed["num_patches"]),

        "token_inputs": token_inputs.reshape(-1).tolist(),
        "targets": targets.reshape(-1).tolist(),

        "token_channel_indices": token_channel_indices.reshape(-1).tolist(),
        "token_time_indices": token_time_indices.reshape(-1).tolist(),
        "token_valid_mask": token_valid_mask.reshape(-1).tolist(),
        "channel_valid_mask": channel_valid_mask.reshape(-1).tolist(),

        "sample_id": int(raw_sample.get("sample_id", -1)),
        "global_idx": int(raw_sample.get("global_idx", -1)),
        "subject_id": str(raw_sample.get("subject_id", "")),
        "edf_relpath": str(raw_sample.get("edf_relpath", "")),
        "segment_start_sec": float(raw_sample.get("segment_start_sec", 0.0)),
        "split": str(raw_sample.get("split", "")),
        "shard_name": str(raw_sample.get("shard_name", "")),
        "dataset_key": str(raw_sample.get("dataset_key", "")),

        "valid_channel_count": int(channel_valid_mask.sum()),
    }

    return row


# =============================================================================
# 13. 主转换函数
# =============================================================================

def build_processed_lance():
    raw_lance_path = RAW_LANCE_PATH
    output_lance_path = OUTPUT_LANCE_PATH
    electrode_vocab_path = ELECTRODE_VOCAB_PATH

    print("\n===== Offline EEG Lance Conversion =====")
    print("raw_lance_path:", raw_lance_path)
    print("output_lance_path:", output_lance_path)
    print("electrode_vocab_path:", electrode_vocab_path)

    if not os.path.exists(raw_lance_path):
        raise FileNotFoundError(f"Raw Lance path not found: {raw_lance_path}")

    if not os.path.exists(electrode_vocab_path):
        raise FileNotFoundError(f"Electrode vocab json not found: {electrode_vocab_path}")

    output_parent = os.path.dirname(output_lance_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    if OVERWRITE and os.path.exists(output_lance_path):
        print(f"Removing existing output Lance: {output_lance_path}")
        shutil.rmtree(output_lance_path)

    id2name = load_electrode_vocab(electrode_vocab_path)

    raw_ds = lance.dataset(raw_lance_path)
    total_rows = raw_ds.count_rows()

    print("\n===== Raw Lance Info =====")
    print("rows:", total_rows)
    print("schema:")
    print(raw_ds.schema)

    raw_columns = [
        "signals",
        "electrode_ids",
        "bad_channel_mask",
        "channel_counts",
        "sample_id",
        "subject_id",
        "edf_relpath",
        "segment_start_sec",
        "split",
        "shard_name",
        "global_idx",
        "dataset_key",
    ]

    available_columns = set(raw_ds.schema.names)
    raw_columns = [c for c in raw_columns if c in available_columns]

    first_write = True
    written = 0
    skipped = 0

    for start in range(0, total_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_rows)

        print(f"\n===== Processing rows {start}:{end} =====")

        batch = raw_ds.take(
            indices=list(range(start, end)),
            columns=raw_columns,
        )

        rows = batch.to_pylist()

        out_rows = []

        for local_i, row in enumerate(rows):
            global_row_idx = start + local_i

            try:
                raw_sample = parse_raw_lance_row(
                    row=row,
                    id2name=id2name,
                    orig_sfreq=ORIG_SFREQ,
                    denorm_clip_divide_200=DENORM_CLIP_DIVIDE_200,
                    drop_bad_channels=DROP_BAD_CHANNELS,
                )

                processed = preprocess_one_raw_sample(raw_sample)

                lance_row = processed_to_lance_row(
                    processed=processed,
                    raw_sample=raw_sample,
                )

                out_rows.append(lance_row)

                if global_row_idx == 0:
                    print("\n===== First Sample Debug =====")
                    print("raw signal shape:", raw_sample["signal"].shape)
                    print("first 20 channel names:", raw_sample["channel_names"][:20])
                    print("token_inputs shape:", processed["token_inputs"].shape)
                    print("targets shape:", processed["targets"].shape)
                    print("channel_valid_mask shape:", processed["channel_valid_mask"].shape)
                    print("valid_channel_count:", processed["channel_valid_mask"].sum())
                    print("token_valid_mask shape:", processed["token_valid_mask"].shape)
                    print("num_tokens:", processed["num_tokens"])
                    print("patch_len:", processed["patch_len"])
                    print("n_bands:", processed["n_bands"])
                    print("frames_per_patch:", processed["frames_per_patch"])
                    print("n_channels:", processed["n_channels"])
                    print("num_patches:", processed["num_patches"])

            except Exception as e:
                skipped += 1
                print(f"[WARN] Skip row {global_row_idx}, error: {repr(e)}")
                continue

        if len(out_rows) == 0:
            print("No valid rows in this batch.")
            continue

        out_table = pa.Table.from_pylist(out_rows)

        if first_write:
            lance.write_dataset(
                out_table,
                output_lance_path,
                mode="create",
            )
            first_write = False
        else:
            lance.write_dataset(
                out_table,
                output_lance_path,
                mode="append",
            )

        written += len(out_rows)

        print(f"written: {written}/{total_rows}, skipped: {skipped}")

    print("\n===== Done =====")
    print("output_lance_path:", output_lance_path)
    print("written:", written)
    print("skipped:", skipped)

    if written > 0:
        out_ds = lance.dataset(output_lance_path)

        print("\n===== Processed Lance Info =====")
        print("rows:", out_ds.count_rows())
        print("schema:")
        print(out_ds.schema)

        first = out_ds.take(indices=[0]).to_pylist()[0]

        print("\n===== First Processed Row Check =====")
        print("num_tokens:", first["num_tokens"])
        print("patch_len:", first["patch_len"])
        print("n_bands:", first["n_bands"])
        print("frames_per_patch:", first["frames_per_patch"])
        print("n_channels:", first["n_channels"])
        print("num_patches:", first["num_patches"])
        print("len(token_inputs):", len(first["token_inputs"]))
        print("len(targets):", len(first["targets"]))
        print("len(token_valid_mask):", len(first["token_valid_mask"]))
        print("len(channel_valid_mask):", len(first["channel_valid_mask"]))
        print("valid_channel_count:", first["valid_channel_count"])

        expected_token_inputs_len = NUM_TOKENS * PATCH_LEN
        expected_targets_len = NUM_TOKENS * N_BANDS * FRAMES_PER_PATCH

        print("\n===== Expected Length Check =====")
        print("expected len(token_inputs):", expected_token_inputs_len)
        print("expected len(targets):", expected_targets_len)

        assert int(first["num_tokens"]) == NUM_TOKENS
        assert int(first["patch_len"]) == PATCH_LEN
        assert int(first["n_bands"]) == N_BANDS
        assert int(first["frames_per_patch"]) == FRAMES_PER_PATCH
        assert int(first["n_channels"]) == N_CHANNELS
        assert int(first["num_patches"]) == NUM_TIME_PATCHES

        assert len(first["token_inputs"]) == expected_token_inputs_len, (
            f"token_inputs length mismatch: got {len(first['token_inputs'])}, "
            f"expected {expected_token_inputs_len}"
        )

        assert len(first["targets"]) == expected_targets_len, (
            f"targets length mismatch: got {len(first['targets'])}, "
            f"expected {expected_targets_len}"
        )

        print("\nShape check passed.")
        print(
            "Expected tensor shapes after Dataset reshape:\n"
            f"  token_inputs: ({NUM_TOKENS}, {PATCH_LEN})\n"
            f"  targets:      ({NUM_TOKENS}, {N_BANDS}, {FRAMES_PER_PATCH})"
        )


if __name__ == "__main__":
    build_processed_lance()
