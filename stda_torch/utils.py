from __future__ import annotations
from typing import List, Tuple, Union, Any

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


# def excitation_composition(
#     stda: sTDA,
#     idx: int,
#     topk: int = 3,
#     original_numbering: bool = True,
#     verbose: bool = True,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if original_numbering:
#         # number the MOs according to their original label (i.e., as if the excitation
#         # space is not truncated)
#         indices = torch.LongTensor(
#             [
#                 [o, v]
#                 for o in stda.mask_occ
#                 for v in stda.mask_vir + stda.mask_occ[-1] + 1
#             ],
#         )
#     else:
#         # number the MOs according to their "sTDA" label (i.e., the same number that
#         # is printed when calling `kernel` on the sTDA object)
#         indices = torch.LongTensor(
#             [
#                 [o, stda.nocc + v]
#                 for o in torch.arange(stda.nocc)
#                 for v in torch.arange(stda.nvir)
#             ]
#         )
#
#     x = stda.x[idx].clone() ** 2
#
#     if abs(torch.sum(x) - 1.0) > 1e-6:
#         raise ValueError("Something is wrong with transition amplitudes matrix")
#
#     topk_values, topk_indices = torch.topk(x.reshape(-1), k=topk)
#     topk_pairs = indices[topk_indices]
#
#     for value, pair in zip(topk_values, topk_pairs):
#         print(f"{value.item()*100:6.2f} % for MO({pair[0]:4d}) -> MO({pair[1]:4d})")
#
#     return topk_values, topk_pairs
