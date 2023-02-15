from typing import Tuple
import torch
from .integrals import eri_mo_monopole


def get_full_ab(
    mo_energy: torch.Tensor,
    mo_coeff: torch.Tensor,
    mo_occ: torch.Tensor,
    ovlp: torch.Tensor,
    natm: torch.Tensor,
    ao_labels: list,
    coords: torch.Tensor,
    atom_pure_symbols: list,
    ax: int,
    alpha: int = None,
    beta: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """computes the full matrices A and B in the sTDA approximation.

        A_ia,jb = δ_ia δ_jb (ε_a - ε_i) + 2 (ia|jb)' - (ij | ab)'

        B_ia,jb = 0

    Here by "full" we mean that no truncation of the active space is performed,
    as in the original sTDA. Only the two electron integrals are approximated
    in the same way.
    Args:
        mo_energy (n_mo): MO energies (ε).
        mo_coeff (n_ao_a, n_mo): MO coefficients matrix (C).
        mo_occ (n_mo): MO occupancy.
        ovlp (n_ao, n_ao): AO overlap matrix (S).
        natm: number of atoms.
        ao_labels: list of tuples describing AOs.
                   same as calling mol.ao_labels(fmt=None) from pyscf.
                   tuple fmt: (atom_index: int, atom: str, ao_name: str, m_def: str)
        coords (n_atoms, 3): coordinates of the molecule in Bohr.
        atom_pure_symbols: list of atom symbols (e.g., ['O', 'H', 'H'] for water).
        ax: fraction of exact Hartree Fock exchange.
        alpha: α parameter of sTDA approximate integrals.
        beta: β parameter of sTDA approximate integrals.
    Returns:
        A (n_mo_occ, n_mo_vir, n_mo_occ, n_mo_vir): A matrix of the Casida equations.
        B (n_mo_occ, n_mo_vir, n_mo_occ, n_mo_vir): B matrix of the Casida equations.
    """
    occidx = torch.where(mo_occ == 2)[0]
    viridx = torch.where(mo_occ == 0)[0]
    orbv = mo_coeff[:, viridx]
    orbo = mo_coeff[:, occidx]
    nvir = orbv.shape[1]
    nocc = orbo.shape[1]

    e_ia = (mo_energy[viridx, None] - mo_energy[occidx]).T

    a = torch.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
    b = torch.zeros_like(a)

    eri_J, eri_K = eri_mo_monopole(
        ovlp=ovlp,
        natm=natm,
        ao_labels=ao_labels,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        coords=coords,
        atom_pure_symbols=atom_pure_symbols,
        ax=ax,
        alpha=alpha,
        beta=beta,
        mode="stda",
    )

    a += eri_K * 2 - eri_J

    return a, b
