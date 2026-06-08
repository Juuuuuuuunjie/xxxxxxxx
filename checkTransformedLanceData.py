import lance
import numpy as np

path = (
    "./serverData/eeg_openneuro_76ch_val_processed.lance"
)

ds = lance.dataset(path)

print("rows:", ds.count_rows())
print(ds.schema)

row = ds.take(indices=[0]).to_pylist()[0]

num_tokens = row["num_tokens"]
patch_len = row["patch_len"]
n_bands = row["n_bands"]
frames_per_patch = row["frames_per_patch"]

token_inputs = np.asarray(row["token_inputs"], dtype=np.float32).reshape(
    num_tokens,
    patch_len,
)

targets = np.asarray(row["targets"], dtype=np.float32).reshape(
    num_tokens,
    n_bands,
    frames_per_patch,
)

token_valid_mask = np.asarray(row["token_valid_mask"], dtype=np.float32)
channel_valid_mask = np.asarray(row["channel_valid_mask"], dtype=np.float32)

print("token_inputs:", token_inputs.shape)
print("targets:", targets.shape)
print("token_valid_mask:", token_valid_mask.shape)
print("channel_valid_mask:", channel_valid_mask.shape)
print("valid channels:", channel_valid_mask.sum())
print("token_inputs mean/std:", token_inputs.mean(), token_inputs.std())
print("targets mean/std:", targets.mean(), targets.std())
