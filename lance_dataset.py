import numpy as np
import torch
from torch.utils.data import Dataset
import lance


class LanceEEGPretrainDataset(Dataset):
    """
    从 Lance 读取已经预处理好的 EEG 预训练样本。

    返回格式要和 EEGPretrainDataset 完全一致，这样原来的 collate_fn、
    trainer、model、loss 都不用改。
    """

    def __init__(self, lance_path):
        self.ds = lance.dataset(lance_path)
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
