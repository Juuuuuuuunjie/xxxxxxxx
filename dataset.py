import json
import os

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
    从 Lance 读取已经预处理好的 EEG 预训练样本。

    支持本地路径，也支持远程 S3/TOS URI。
    """

    def __init__(self, lance_path, storage_options_path=None, storage_options=None):
        if storage_options is None and storage_options_path is not None:
            storage_options = load_storage_options(storage_options_path)

        if storage_options is None:
            self.ds = lance.dataset(lance_path)
        else:
            self.ds = lance.dataset(
                lance_path,
                storage_options=storage_options,
            )

        self.length = self.ds.count_rows()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch = self.ds.take(
            [idx],
            columns=[
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
            ],
        )

        row = batch.to_pylist()[0]

        num_tokens = int(row["num_tokens"])
        patch_len = int(row["patch_len"])
        n_bands = int(row["n_bands"])
        frames_per_patch = int(row["frames_per_patch"])
        n_channels = int(row["n_channels"])

        token_inputs = np.asarray(row["token_inputs"], dtype=np.float32)
        token_inputs = token_inputs.reshape(num_tokens, patch_len)

        targets = np.asarray(row["targets"], dtype=np.float32)
        targets = targets.reshape(num_tokens, n_bands, frames_per_patch)

        token_channel_indices = np.asarray(row["token_channel_indices"], dtype=np.int64)
        token_time_indices = np.asarray(row["token_time_indices"], dtype=np.int64)
        token_valid_mask = np.asarray(row["token_valid_mask"], dtype=np.float32)
        channel_valid_mask = np.asarray(row["channel_valid_mask"], dtype=np.float32)

        if channel_valid_mask.shape[0] != n_channels:
            raise ValueError(
                f"channel_valid_mask shape mismatch: "
                f"{channel_valid_mask.shape[0]} != {n_channels}"
            )

        return {
            "token_inputs": torch.tensor(token_inputs, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "token_channel_indices": torch.tensor(token_channel_indices, dtype=torch.long),
            "token_time_indices": torch.tensor(token_time_indices, dtype=torch.long),
            "token_valid_mask": torch.tensor(token_valid_mask, dtype=torch.float32),
            "channel_valid_mask": torch.tensor(channel_valid_mask, dtype=torch.float32),
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
