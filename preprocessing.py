import numpy as np
from scipy.signal import resample_poly
from channel_config import STANDARD_64_CHANNELS, CHANNEL_ALIASES


def canonicalize_name(name: str) -> str:
    name = name.strip()
    name = name.replace("EEG ", "").replace("-REF", "").replace("-LE", "").replace("-A1", "").replace("-A2", "")
    if name in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name]
    if name.upper() in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name.upper()]
    return name


def map_to_fixed_64_channels(signal: np.ndarray, channel_names: list):
    """
    signal: [C_in, T]
    channel_names: len = C_in
    return:
        fixed_signal: [64, T]
        valid_mask: [64]
    """
    T = signal.shape[1]
    fixed = np.zeros((len(STANDARD_64_CHANNELS), T), dtype=np.float32)
    valid = np.zeros((len(STANDARD_64_CHANNELS),), dtype=np.float32)

    name_to_idx = {}
    for i, ch in enumerate(channel_names):
        cname = canonicalize_name(ch)
        name_to_idx[cname] = i

    for j, std_name in enumerate(STANDARD_64_CHANNELS):
        if std_name in name_to_idx:
            fixed[j] = signal[name_to_idx[std_name]]
            valid[j] = 1.0

    return fixed, valid


def average_reference(signal_64: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-6):
    """
    signal_64: [64, T]
    valid_mask: [64]
    only valid channels participate in average reference
    """
    valid_idx = valid_mask > 0.5
    if valid_idx.sum() <= 1:
        return signal_64.copy()

    referenced = signal_64.copy()
    avg = referenced[valid_idx].mean(axis=0, keepdims=True)
    referenced[valid_idx] = referenced[valid_idx] - avg
    referenced[~valid_idx] = 0.0
    return referenced


def resample_signal(signal: np.ndarray, orig_sfreq: int, target_sfreq: int):
    """
    signal: [C, T]
    """
    if orig_sfreq == target_sfreq:
        return signal.astype(np.float32)
    resampled = resample_poly(signal, up=target_sfreq, down=orig_sfreq, axis=1)
    return resampled.astype(np.float32)


def preprocess_eeg(signal: np.ndarray, channel_names: list, orig_sfreq: int, target_sfreq: int):
    fixed, valid = map_to_fixed_64_channels(signal, channel_names)
    fixed = average_reference(fixed, valid)
    fixed = resample_signal(fixed, orig_sfreq, target_sfreq)
    return fixed.astype(np.float32), valid.astype(np.float32)
