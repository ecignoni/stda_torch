from __future__ import annotations
from typing import List, Tuple, Union, Any

from collections import namedtuple
import torch
import numpy as np

sTDA = Any

physconst = namedtuple(
    "PhysicalConstants",
    ["au_to_ev", "ang_to_bohr"],
)(27.211396641308, 1.8897259886)

symbol_to_charge = {
    "  H": 1.0,
    " He": 2.0,
    " Li": 3.0,
    " Be": 4.0,
    "  B": 5.0,
    "  C": 6.0,
    "  N": 7.0,
    "  O": 8.0,
    "  F": 9.0,
    " Ne": 10.0,
}
symbol_to_charge = {k.strip(): v for k, v in symbol_to_charge.items()}


def ensure_torch(t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if torch.is_tensor(t):
        return t
    elif type(t) is np.ndarray:
        return torch.from_numpy(t)
    else:
        raise ValueError("input variable is not torch.Tensor nor np.ndarray")


def isqrtm(A: torch.Tensor) -> torch.Tensor:
    eva, eve = torch.linalg.eigh(A)
    idx = eva > 1e-15
    return eve[:, idx] @ torch.diag(eva[idx] ** (-0.5)) @ eve[:, idx].T


def sqrtm(A: torch.Tensor) -> torch.Tensor:
    eva, eve = torch.linalg.eigh(A)
    idx = eva > 1e-15
    return eve[:, idx] @ torch.diag(eva[idx] ** 0.5) @ eve[:, idx].T


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
    e, v = torch.linalg.eigh(a)
    # trick to get 'idx' of the same length of e
    e[e < 0] = float("inf")
    idx = torch.argsort(e)
    e = e[idx][:nstates]
    v = v[:, idx][:, :nstates]
    return e, v


def mulliken_population(
    mo_coeff: torch.Tensor,
    mo_occ: torch.Tensor,
    ovlp: torch.Tensor,
    natm: int,
    ao_labels: List[Tuple[int, str, str, str]],
) -> torch.Tensor:
    dm = 2 * torch.matmul(mo_coeff[:, mo_occ == 2], mo_coeff[:, mo_occ == 2].T)
    pop = torch.einsum("ij,ji->i", dm, ovlp)
    q = torch.zeros(natm)
    for i, (atidx, *_) in enumerate(ao_labels):
        q[atidx] += pop[i]
    return q, pop


def normalize_ao(
    mo_coeff: torch.Tensor,
    ovlp: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor]:
    """normalizes the AO basis

    Normalizes the atomic orbital basis, modifying
    the overlap matrix S and the MO coefficients C.
    Rows and columns of S are divided by diag(S)^½,
    while rows of C are multiplied by diag(S)^½

    Args:
        mo_coeff: MO coefficients (C matrix)
        ovlp: overlap (S matrix)
    Returns:
        mo_coeff: normalized MO coefficients
        ovlp: normalized overlap
    """
    norm = torch.diag(ovlp) ** 0.5
    ovlp = torch.einsum("i,ij,j->ij", 1.0 / norm, ovlp, 1.0 / norm)
    mo_coeff = torch.einsum("i,ij->ij", norm, mo_coeff)
    return mo_coeff, ovlp


def unnormalize_ao(mo_coeff: torch.Tensor, ovlp: torch.Tensor) -> torch.Tensor:
    norm = torch.diag(ovlp) ** 0.5
    mo_coeff = torch.einsum("i,ij->ij", 1.0 / norm, mo_coeff)
    return mo_coeff
