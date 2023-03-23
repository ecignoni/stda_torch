from __future__ import annotations
from typing import Tuple

import warnings

import torch


def get_nto(
    x: torch.Tensor, mo_occ: torch.Tensor, mo_vir: torch.Tensor, state: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """get the Natural Transition Orbitals associated with a transition

    Args:
        x: transition density, x_ia, shape=(nocc, nvir)
        mo_occ: coefficients of the occupied MOs, shape=(nao, nocc)
        mo_vir: coefficients of the virtual MOs, shape=(nao, nvir)
        state: index of the excited state. 1 corresponds to the first
             excited state
    Returns:
        weights: weight of each NTO (squared singular values
               of x)
        nto_occ: coefficients of the hole NTOs, shape=(nao, nocc)
        nto_vir: coefficients of the particle NTOs, shape=(nao, nvir)
        nto_U: mixing matrix of occupied NTOs (U in X = U Σ V.T)
        nto_V: mixing matrix of unoccupied NTOs (V in X = U Σ V.T)
    """
    # to provide a similar API to PySCF
    if state == 0:
        warnings.warn(
            "Excited state starts from 1. " "Set state=1 for the first excited state"
        )
    elif state < 0:
        pass
    else:
        state = state - 1

    # get transition density
    t = x[state].clone()

    nocc, nvir = t.shape

    # check normalization
    if abs(torch.linalg.norm(t) - 1.0) > 1e-10:
        norm = torch.linalg.norm(t).item()
        warnings.warn(
            f"Transition amplitudes X are not normalized (norm={norm}) "
            "Normalizing now."
        )
        t *= 1.0 / torch.linalg.norm(t)

    u, s, vt = torch.linalg.svd(t)
    v = vt.conj().T
    weights = s**2

    # enforce reproducible sign:
    # look at the biggest absolute value for each column
    # if negative, change the sign of the column
    idx = torch.argmax(abs(u.real), axis=0)
    u[:, u[idx, torch.arange(nocc)].real < 0] *= -1
    idx = torch.argmax(abs(v.real), axis=0)
    v[:, v[idx, torch.arange(nvir)].real < 0] *= -1

    # NTOs
    nto_occ = torch.matmul(mo_occ, u)
    nto_vir = torch.matmul(mo_vir, v)
    # nto_coeff = torch.column_stack((nto_occ, nto_vir))

    return weights, nto_occ, nto_vir, u, v
