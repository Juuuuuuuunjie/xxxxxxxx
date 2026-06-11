import json

import lance
import numpy as np
import torch
from torch.utils.data import Dataset


def load_storage_options(storage_options_path):
    with open(storage_options_path, "r", encoding="utf-8") as f:
        storage_options = json.load(f)

    # Lance / object_store 一般要求 value 是字符串
    return {k: str(v) for k, v in storage_options.items()}


class LanceEEGPretrainDataset(Dataset):
    """
    读取离线处理好的 EEG 预训练 Lance。
    支持 __getitems__ 批量读取，减少每个 batch 的 Lance take 次数。
    """

    COLUMNS = [
        "num_tokens",
        "patch_len",
        "n_bands",
        "frames_per_patch",
        "n_channels",
        "token_inputs",
        "targets",
        "token_channel_indices",
        "token_time_indices",
        "token_valid_mask",
        "channel_valid_mask",
    ]

    def __init__(self, lance_path, storage_options=None):
        self.lance_path = lance_path
        self.storage_options = storage_options
        self.ds = None
        self.length = self._get_length()

    def _open(self):
        if self.ds is None:
            if self.storage_options is None:
                self.ds = lance.dataset(self.lance_path)
            else:
                self.ds = lance.dataset(
                    self.lance_path,
                    storage_options=self.storage_options,
                )
        return self.ds

    def _get_length(self):
        ds = self._open()
        return ds.count_rows()

    def __len__(self):
        return self.length

    def _row_to_sample(self, row):
        num_tokens = int(row["num_tokens"])
        patch_len = int(row["patch_len"])
        n_bands = int(row["n_bands"])
        frames_per_patch = int(row["frames_per_patch"])
        n_channels = int(row["n_channels"])

        token_inputs = np.asarray(row["token_inputs"], dtype=np.float32).reshape(
            num_tokens, patch_len
        )

        targets = np.asarray(row["targets"], dtype=np.float32).reshape(
            num_tokens, n_bands, frames_per_patch
        )

        token_channel_indices = np.asarray(
            row["token_channel_indices"],
            dtype=np.int64,
        )

        token_time_indices = np.asarray(
            row["token_time_indices"],
            dtype=np.int64,
        )

        token_valid_mask = np.asarray(
            row["token_valid_mask"],
            dtype=np.float32,
        )

        channel_valid_mask = np.asarray(
            row["channel_valid_mask"],
            dtype=np.float32,
        )

        if channel_valid_mask.shape[0] != n_channels:
            raise ValueError(
                f"channel_valid_mask shape mismatch: "
                f"{channel_valid_mask.shape[0]} != {n_channels}"
            )

        return {
            "token_inputs": torch.from_numpy(token_inputs),
            "targets": torch.from_numpy(targets),
            "token_channel_indices": torch.from_numpy(token_channel_indices),
            "token_time_indices": torch.from_numpy(token_time_indices),
            "token_valid_mask": torch.from_numpy(token_valid_mask),
            "channel_valid_mask": torch.from_numpy(channel_valid_mask),
        }

    def __getitem__(self, idx):
        ds = self._open()

        batch = ds.take(
            indices=[int(idx)],
            columns=self.COLUMNS,
        )
        row = batch.to_pylist()[0]
        return self._row_to_sample(row)

    def __getitems__(self, indices):
        ds = self._open()

        if len(indices) == 0:
            return []

        indices = [int(i) for i in indices]
        batch = ds.take(
            indices=indices,
            columns=self.COLUMNS,
        )
        rows = batch.to_pylist()
        return [self._row_to_sample(row) for row in rows]
