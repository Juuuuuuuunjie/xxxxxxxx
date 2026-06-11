import queue
import threading

import torch


def _pin_memory_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    if isinstance(batch, dict):
        return {k: _pin_memory_batch(v) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_pin_memory_batch(v) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_pin_memory_batch(v) for v in batch)
    return batch


def _move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    return batch


class PrefetchLoader:
    def __init__(self, loader, device, prefetch=2):
        self.loader = loader
        self.device = torch.device(device)
        self.prefetch = max(1, int(prefetch))

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        q = queue.Queue(maxsize=self.prefetch)
        stop_token = object()
        error_holder = []

        def producer():
            try:
                for batch in self.loader:
                    q.put(_pin_memory_batch(batch))
            except Exception as exc:
                error_holder.append(exc)
            finally:
                q.put(stop_token)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        transfer_stream = torch.cuda.Stream(device=self.device)

        def preload():
            batch = q.get()
            if batch is stop_token:
                return None
            with torch.cuda.stream(transfer_stream):
                batch = _move_batch_to_device(batch, self.device)
            return batch

        next_batch = preload()

        while next_batch is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(transfer_stream)
            current_batch = next_batch

            if error_holder:
                raise error_holder[0]

            next_batch = preload()
            yield current_batch

        if error_holder:
            raise error_holder[0]
