import argparse
import numpy as np
import pyarrow as pa
import lance

from configs import Config
from io_utils import load_multiple_eeglab_sets_as_samples
from dataset import preprocess_all_samples


def flatten_sample(sample):
    return {
        "token_inputs": sample["token_inputs"].reshape(-1).astype(np.float32).tolist(),
        "targets": sample["targets"].reshape(-1).astype(np.float32).tolist(),
        "token_channel_indices": sample["token_channel_indices"].astype(np.int64).tolist(),
        "token_time_indices": sample["token_time_indices"].astype(np.int64).tolist(),
        "token_valid_mask": sample["token_valid_mask"].astype(np.float32).tolist(),
        "channel_valid_mask": sample["channel_valid_mask"].astype(np.float32).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pretrain_processed.lance")
    args = parser.parse_args()

    cfg = Config()

    print("===== Loading Raw EEG Clips =====")
    raw_samples = load_multiple_eeglab_sets_as_samples(
        set_paths=cfg.data.set_paths,
        clip_seconds=cfg.data.clip_seconds,
        clip_stride_seconds=cfg.data.clip_stride_seconds,
        convert_v_to_uv=cfg.data.convert_v_to_uv,
    )

    print(f"Total raw clips: {len(raw_samples)}")

    if len(raw_samples) == 0:
        raise ValueError("No raw clips found.")

    print("===== Preprocessing All Clips =====")
    processed_samples = preprocess_all_samples(raw_samples, cfg)

    if len(processed_samples) == 0:
        raise ValueError("No processed samples generated.")

    first = processed_samples[0]

    token_inputs_shape = first["token_inputs"].shape
    targets_shape = first["targets"].shape
    token_channel_indices_shape = first["token_channel_indices"].shape
    token_time_indices_shape = first["token_time_indices"].shape
    token_valid_mask_shape = first["token_valid_mask"].shape
    channel_valid_mask_shape = first["channel_valid_mask"].shape

    print("===== First Processed Sample Shape =====")
    print("token_inputs:", token_inputs_shape)
    print("targets:", targets_shape)
    print("token_channel_indices:", token_channel_indices_shape)
    print("token_time_indices:", token_time_indices_shape)
    print("token_valid_mask:", token_valid_mask_shape)
    print("channel_valid_mask:", channel_valid_mask_shape)

    num_tokens, patch_len = token_inputs_shape
    _, n_bands, frames_per_patch = targets_shape
    n_channels = channel_valid_mask_shape[0]

    rows = []

    for i, sample in enumerate(processed_samples):
        if sample["token_inputs"].shape != token_inputs_shape:
            raise ValueError(f"sample {i} token_inputs shape mismatch")
        if sample["targets"].shape != targets_shape:
            raise ValueError(f"sample {i} targets shape mismatch")

        row = flatten_sample(sample)
        row.update({
            "sample_idx": i,
            "num_tokens": num_tokens,
            "patch_len": patch_len,
            "n_bands": n_bands,
            "frames_per_patch": frames_per_patch,
            "n_channels": n_channels,
        })
        rows.append(row)

    schema = pa.schema([
        pa.field("sample_idx", pa.int64()),

        pa.field("num_tokens", pa.int64()),
        pa.field("patch_len", pa.int64()),
        pa.field("n_bands", pa.int64()),
        pa.field("frames_per_patch", pa.int64()),
        pa.field("n_channels", pa.int64()),

        pa.field("token_inputs", pa.list_(pa.float32(), num_tokens * patch_len)),
        pa.field("targets", pa.list_(pa.float32(), num_tokens * n_bands * frames_per_patch)),
        pa.field("token_channel_indices", pa.list_(pa.int64(), num_tokens)),
        pa.field("token_time_indices", pa.list_(pa.int64(), num_tokens)),
        pa.field("token_valid_mask", pa.list_(pa.float32(), num_tokens)),
        pa.field("channel_valid_mask", pa.list_(pa.float32(), n_channels)),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)

    print(f"===== Writing Lance Dataset: {args.output} =====")
    lance.write_dataset(table, args.output, mode="overwrite")

    print("Done.")
    print(f"Wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
