import re
import math
import numpy as np
from scipy.signal import resample_poly

from channel_config import STANDARD_64_CHANNELS, CHANNEL_ALIASES


def _canonical_name(name: str) -> str:
    """
    把原始 EEG 通道名规范化。

    处理例子：
        "EEG Fp1-Ref" -> "Fp1"
        "FP1" -> "Fp1"
        "Cz." -> "Cz"
        "T3" -> "T7"
    """
    if name is None:
        return ""

    x = str(name).strip()

    # 去掉常见前缀
    x = re.sub(r"^EEG\s+", "", x, flags=re.IGNORECASE)

    # 去掉常见后缀 / reference 标记
    x = re.sub(r"[-_ ]?(REF|REFERENCE|AVG|A1|A2|M1|M2)$", "", x, flags=re.IGNORECASE)

    # 去掉非字母数字字符
    x = re.sub(r"[^A-Za-z0-9]", "", x)

    if x == "":
        return ""

    # 先处理全大写别名
    xu = x.upper()
    if xu in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[xu]

    # 处理 T3/T4/T5/T6
    if xu in ["T3", "T4", "T5", "T6"]:
        return CHANNEL_ALIASES[xu]

    # 标准化大小写：
    # FP1 -> Fp1, AFZ -> AFz, CZ -> Cz
    letters = "".join([c for c in x if c.isalpha()])
    digits = "".join([c for c in x if c.isdigit()])

    lu = letters.upper()

    if lu == "FP":
        out = "Fp" + digits
    elif lu == "AF":
        out = "AF" + digits
    elif lu == "F":
        out = "F" + digits
    elif lu == "FT":
        out = "FT" + digits
    elif lu == "FC":
        out = "FC" + digits
    elif lu == "T":
        out = "T" + digits
    elif lu == "C":
        out = "C" + digits
    elif lu == "TP":
        out = "TP" + digits
    elif lu == "CP":
        out = "CP" + digits
    elif lu == "P":
        out = "P" + digits
    elif lu == "PO":
        out = "PO" + digits
    elif lu == "O":
        out = "O" + digits
    elif lu == "I":
        out = "I" + digits
    else:
        out = x

    # z 通道特殊处理
    z_map = {
        "FPZ": "Fpz",
        "AFZ": "AFz",
        "FZ": "Fz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "POZ": "POz",
        "OZ": "Oz",
        "IZ": "Iz",
    }

    if xu in z_map:
        out = z_map[xu]

    # 再过一次 alias
    if out in CHANNEL_ALIASES:
        out = CHANNEL_ALIASES[out]

    return out


def _resample_signal(signal: np.ndarray, orig_sfreq: float, target_sfreq: float):
    """
    使用 polyphase resampling 重采样。
    signal: [C, T]
    """
    if abs(orig_sfreq - target_sfreq) < 1e-6:
        return signal.astype(np.float32)

    # 近似成整数比例
    gcd = math.gcd(int(round(orig_sfreq)), int(round(target_sfreq)))
    up = int(round(target_sfreq)) // gcd
    down = int(round(orig_sfreq)) // gcd

    out = resample_poly(signal, up=up, down=down, axis=1)
    return out.astype(np.float32)


def _crop_or_pad(signal: np.ndarray, target_num_points: int):
    """
    把信号裁剪或补零到固定长度。
    signal: [C, T]
    """
    C, T = signal.shape

    if T == target_num_points:
        return signal

    if T > target_num_points:
        return signal[:, :target_num_points]

    pad_len = target_num_points - T
    pad = np.zeros((C, pad_len), dtype=signal.dtype)
    return np.concatenate([signal, pad], axis=1)


def preprocess_eeg(
    signal,
    channel_names,
    orig_sfreq,
    target_sfreq,
    target_num_points=None,
):
    """
    把原始 EEG 预处理成固定 64 通道模板。

    Args:
        signal:
            np.ndarray, [C_in, T]
        channel_names:
            list[str], 长度 C_in
        orig_sfreq:
            原始采样率
        target_sfreq:
            目标采样率
        target_num_points:
            如果不为 None，则最终输出固定为 [64, target_num_points]

    Returns:
        signal_64:
            np.ndarray, [64, T_out]
        channel_valid_mask:
            np.ndarray, [64]
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError(f"signal must be [C, T], got shape {signal.shape}")

    if len(channel_names) != signal.shape[0]:
        raise ValueError(
            f"len(channel_names)={len(channel_names)} does not match signal.shape[0]={signal.shape[0]}"
        )

    # 1) 规范化原始通道名
    canonical_names = [_canonical_name(x) for x in channel_names]

    # 2) 构造 name -> index
    name_to_index = {}
    for i, name in enumerate(canonical_names):
        if name == "":
            continue
        # 如果重复通道名，只保留第一个
        if name not in name_to_index:
            name_to_index[name] = i

    # 3) 映射到标准 64 通道
    C_std = len(STANDARD_64_CHANNELS)
    T = signal.shape[1]

    signal_64 = np.zeros((C_std, T), dtype=np.float32)
    channel_valid_mask = np.zeros((C_std,), dtype=np.float32)

    for out_idx, std_name in enumerate(STANDARD_64_CHANNELS):
        if std_name in name_to_index:
            in_idx = name_to_index[std_name]
            signal_64[out_idx] = signal[in_idx]
            channel_valid_mask[out_idx] = 1.0

    # 4) 平均参考，只对真实存在的通道做
    valid_indices = np.where(channel_valid_mask > 0.5)[0]

    if len(valid_indices) > 0:
        ref = signal_64[valid_indices].mean(axis=0, keepdims=True)
        signal_64[valid_indices] = signal_64[valid_indices] - ref

    # 5) 重采样
    signal_64 = _resample_signal(
        signal=signal_64,
        orig_sfreq=orig_sfreq,
        target_sfreq=target_sfreq,
    )

    # 6) 固定长度
    if target_num_points is not None:
        signal_64 = _crop_or_pad(signal_64, target_num_points)

    return signal_64.astype(np.float32), channel_valid_mask.astype(np.float32)


def print_channel_mapping_summary(channel_names):
    """
    调试用：打印原始通道名清洗后的结果。
    """
    print("Original -> Canonical")
    for x in channel_names:
        print(f"{x} -> {_canonical_name(x)}")
