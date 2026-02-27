from __future__ import annotations


def patch_sequence(seq: list[float], patch_len: int = 4) -> list[list[float]]:
    if patch_len <= 0:
        return [seq]
    return [seq[i : i + patch_len] for i in range(0, len(seq), patch_len) if seq[i : i + patch_len]]
