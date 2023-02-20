from __future__ import annotations
from typing import Tuple
import torch
from .integrals import eri_mo_monopole
from .excitation_space import (
    select_csf_by_energy,
    select_csf_by_perturbation,
    restrict_to_stda_excitation_space,
)


def get_ab(
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
    mask_occ: torch.Tensor = None,
    mask_vir: torch.Tensor = None,
    excitation_space: str = "stda",
    e_max: float = None,
    tp: float = None,
    verbose: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
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
        mask_occ: indices of occupied MOs to consider. The first occupied MO
                  has index 0.
        mask_vir: indices of virtual MOs to consider. The first virtual MO
                  has index 0.
        excitation_space: whether to use the excitation space of sTDA ('stda')
                          or the full excitation space ('full')
        e_max: energy threshold of sTDA
        tp: perturbative threshold of sTDA
        verbose: whether to be verbose
    Returns:
        A (n_mo_occ, n_mo_vir, n_mo_occ, n_mo_vir): A matrix of the Casida equations.
        B (n_mo_occ, n_mo_vir, n_mo_occ, n_mo_vir): B matrix of the Casida equations.
    """
    occidx = torch.where(mo_occ == 2)[0]
    viridx = torch.where(mo_occ == 0)[0]
    if mask_occ is not None:
        occidx = occidx[mask_occ]
    if mask_vir is not None:
        viridx = viridx[mask_vir]
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
        mask_occ=mask_occ,
        mask_vir=mask_vir,
    )

    a += eri_K * 2 - eri_J

    if excitation_space == "full":
        idx_pcsf = None
        idx_scsf = None
        idx_ncsf = None
        e_pt_ncsf = None

    elif excitation_space == "stda":
        if e_max is None:
            raise RuntimeError("you have to provide e_max if excitation_space='stda'")

        if tp is None:
            raise RuntimeError("you have to provide tp if excitation_space='stda'")

        idx_pcsf, idx_ncsf = select_csf_by_energy(a, e_max=e_max, verbose=verbose)
        idx_scsf, idx_ncsf, e_pt_ncsf = select_csf_by_perturbation(
            a, idx_pcsf, idx_ncsf, e_max=e_max, tp=tp, verbose=verbose
        )
        a, b = restrict_to_stda_excitation_space(
            a, b, idx_pcsf=idx_pcsf, idx_scsf=idx_scsf, e_pt_ncsf=e_pt_ncsf
        )

    return a, b, idx_pcsf, idx_scsf, idx_ncsf, e_pt_ncsf
