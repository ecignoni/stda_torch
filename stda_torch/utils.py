from __future__ import annotations
from collections import namedtuple
from typing import List, Tuple, Union
import torch

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
