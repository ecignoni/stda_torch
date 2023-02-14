from typing import List
import torch
from .parameters import chemical_hardness
from .utils import sqrtm


def charge_density_monopole(ovlp: torch.Tensor, natm: int, ao_labels: List, mo_coeff_a: torch.Tensor, mo_coeff_b: torch.Tensor) -> torch.Tensor:
    """computes the q_pq^A using Löwdin population analysis.
        q_pq^A = Σ_(μ ϵ A) C'μp^(a) C'μq^(b)
        C' = S^(½) C
    Args:
        ovlp (n_ao, n_ao): AO overlap matrix (S).
        natm: number of atoms.
        ao_labels: list of tuples describing AOs.
                          same as calling mol.ao_labels(fmt=None) from pyscf.
                          tuple fmt: (atom_index: int, atom: str, ao_name: str, m_def: str)
        mo_coeff_a (n_ao_a, n_mo_a): MO coefficients matrix (C).
        mo_coeff_b (n_ao_b, n_mo_b): MO coefficients matrix (C).
    Returns:
        q (natm, n_mo_a, n_mo_b): charges from Löwdin population analysis.
    """
    ovlp_i12 = sqrtm(ovlp)
    coeff_orth_a = torch.matmul(ovlp_i12, mo_coeff_a)
    coeff_orth_b = torch.matmul(ovlp_i12, mo_coeff_b)
    nmo_a = coeff_orth_a.shape[1]
    nmo_b = coeff_orth_b.shape[1]
    q = torch.zeros((natm, nmo_a, nmo_b))
    for i, (atidx, *_) in enumerate(ao_labels):
        q[atidx] += torch.einsum("p,q->pq", coeff_orth_a[i], coeff_orth_b[i]).real
    return q


def distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """computes the matrix of pairwise distances.

        R_ij = ‖c_i - c_j‖₂

    Args:
        coords (n_atoms, 3): coordinates of the molecule in Bohr.
    Returns:
        R : matrix of pairwise distances
    """
    R = torch.cdist(coords, coords, p=2.0)
    return R


def hardness_matrix(atom_pure_symbols: List[str]) -> torch.Tensor:
    """computes the matrix of average chemical hardnesses

        η_ij = (η_i + η_j) / 2

    Args:
        atom_pure_symbols: list of atom symbols (e.g., ['O', 'H', 'H'] for water).
    Returns:
        η: matrix of average chemical hardness
    """
    hrd = chemical_hardness
    eta = torch.DoubleTensor([hrd[sym] for sym in atom_pure_symbols])
    eta = (eta[:, None] + eta[None, :]) / 2
    return eta
