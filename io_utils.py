import os
import numpy as np
import mne


def load_eeglab_set_as_raw(set_path: str):
    """
    读取 EEGLAB .set 文件，返回 MNE Raw 对象。
    """
    if not os.path.exists(set_path):
        raise FileNotFoundError(f"Cannot find .set file: {set_path}")

    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    return raw


def split_raw_to_fixed_length_samples(
    raw,
    clip_seconds: float,
    clip_stride_seconds: float,
    convert_v_to_uv: bool = True,
):
    """
    把连续 EEG Raw 切成多个固定长度 sample。

    Args:
        raw:
            mne.io.Raw
        clip_seconds:
            每个样本长度，单位秒
        clip_stride_seconds:
            样本滑动步长，单位秒
        convert_v_to_uv:
            MNE 读 EEG 通常是 Volt。
            如果 True，则乘以 1e6 转成 microvolt。

    Returns:
        samples:
            list[dict]
            每个 dict:
                {
                    "signal": np.ndarray [C, T],
                    "channel_names": list[str],
                    "sfreq": float,
                }
    """
    signal = raw.get_data().astype(np.float32)  # [C, T]

    if convert_v_to_uv:
        signal = signal * 1e6

    channel_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])

    clip_len = int(round(clip_seconds * sfreq))
    stride_len = int(round(clip_stride_seconds * sfreq))

    if clip_len <= 0:
        raise ValueError("clip_len must be positive.")

    if stride_len <= 0:
        raise ValueError("stride_len must be positive.")

    total_points = signal.shape[1]

    samples = []

    start = 0
    while start + clip_len <= total_points:
        end = start + clip_len

        clip_signal = signal[:, start:end].astype(np.float32)

        samples.append({
            "signal": clip_signal,
            "channel_names": channel_names,
            "sfreq": sfreq,
        })

        start += stride_len

    return samples


def load_eeglab_set_as_fixed_length_samples(
    set_path: str,
    clip_seconds: float,
    clip_stride_seconds: float,
    convert_v_to_uv: bool = True,
):
    """
    读取一个 .set 文件，并切成多个固定长度样本。
    """
    raw = load_eeglab_set_as_raw(set_path)

    samples = split_raw_to_fixed_length_samples(
        raw=raw,
        clip_seconds=clip_seconds,
        clip_stride_seconds=clip_stride_seconds,
        convert_v_to_uv=convert_v_to_uv,
    )

    return samples


def load_multiple_eeglab_sets_as_samples(
    set_paths,
    clip_seconds: float,
    clip_stride_seconds: float,
    convert_v_to_uv: bool = True,
):
    """
    读取多个 .set 文件，并合并成一个 samples list。
    """
    all_samples = []

    for set_path in set_paths:
        print(f"Loading: {set_path}")

        samples = load_eeglab_set_as_fixed_length_samples(
            set_path=set_path,
            clip_seconds=clip_seconds,
            clip_stride_seconds=clip_stride_seconds,
            convert_v_to_uv=convert_v_to_uv,
        )

        print(f"  created {len(samples)} clips")

        all_samples.extend(samples)

    return all_samples
