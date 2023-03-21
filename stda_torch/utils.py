from __future__ import annotations
from typing import List, Tuple, Union, Any

import warnings

from collections import namedtuple
import torch

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
    #     " Na": ,
    #     " Mg": ,
    #     " Al": ,
    #     " Si": ,
    #     "  P": ,
    #     "  S": ,
    #     " Cl": ,
    #     " Ar": ,
    #     "  K": ,
    #     " Ca": ,
    #     " Sc": ,
    #     " Ti": ,
    #     "  V": ,
    #     " Cr": ,
    #     " Mn": ,
    #     " Fe": ,
    #     " Co": ,
    #     " Ni": ,
    #     " Cu": ,
    #     " Zn": ,
    #     " Ga": ,
    #     " Ge": ,
    #     " As": ,
    #     " Se": ,
    #     " Br": ,
    #     " Kr": ,
    #     " Rb": ,
    #     " Sr": ,
    #     "  Y": ,
    #     " Zr": ,
    #     " Nb": ,
    #     " Mo": ,
    #     " Tc": ,
    #     " Ru": ,
    #     " Rh": ,
    #     " Pd": ,
    #     " Ag": ,
    #     " Cd": ,
    #     " In": ,
    #     " Sn": ,
    #     " Sb": ,
    #     " Te": ,
    #     "  I": ,
    #     " Xe": ,
    #     " Cs": ,
    #     " Ba": ,
    #     " La": ,
    #     " Ce": ,
    #     " Pr": ,
    #     " Nd": ,
    #     " Pm": ,
    #     " Sm": ,
    #     " Eu": ,
    #     " Gd": ,
    #     " Tb": ,
    #     " Dy": ,
    #     " Ho": ,
    #     " Er": ,
    #     " Tm": ,
    #     " Yb": ,
    #     " Lu": ,
    #     " Hf": ,
    #     " Ta": ,
    #     "  W": ,
    #     " Re": ,
    #     " Os": ,
    #     " Ir": ,
    #     " Pt": ,
    #     " Au": ,
    #     " Hg": ,
    #     " Tl": ,
    #     " Pb": ,
    #     " Bi": ,
    #     " Po": ,
    #     " At": ,
    #     " Rn": ,
    #     " Fr": ,
    #     " Ra": ,
    #     " Ac": ,
    #     " Th": ,
    #     " Pa": ,
    #     "  U": ,
    #     " Np": ,
    #     " Pu": ,
}
symbol_to_charge = {k.strip(): v for k, v in symbol_to_charge.items()}


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
    if abs(torch.linalg.norm(t) - 1.0) > 1e-15:
        warnings.warn("Transition amplitudes X are not normalized. " "Normalizing now.")
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
