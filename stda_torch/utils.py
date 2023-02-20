from __future__ import annotations
from typing import Union
import torch


def isqrtm(A: torch.Tensor) -> torch.Tensor:
    eva, eve = torch.linalg.eigh(A)
    return eve @ torch.diag(eva ** (-0.5)) @ eve.T


def sqrtm(A: torch.Tensor) -> torch.Tensor:
    eva, eve = torch.linalg.eigh(A)
    return eve @ torch.diag(eva**0.5) @ eve.T


def direct_diagonalization(
    a: torch.Tensor, nstates: int = 3
) -> Union[torch.Tensor, torch.Tensor]:
    if a.ndim == 4:
        nocc, nvir, _, _ = a.shape
        a = a.reshape(nocc * nvir, nocc * nvir)
    elif a.ndim == 2:
        pass
    else:
        raise RuntimeError(f"a.ndim={a.ndim} not supported")
    e, v = torch.linalg.eig(a)
    # trick to get 'idx' of the same length of e
    e[e < 0] = torch.infty
    idx = torch.argsort(e)
    e = e[idx][:nstates]
    v = v[:, idx][:, :nstates]
    return e, v
