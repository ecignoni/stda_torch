from __future__ import annotations
from typing import Tuple, Union, Any

import numpy as np
import torch

from .integrals import eri_mo_monopole
from .excitation_space import (
    select_csf_by_energy,
    select_csf_by_perturbation,
    restrict_to_stda_excitation_space,
)
from .utils import ensure_torch

Mol = Any


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
    mo_orth: bool = False,
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
        mo_orth: whether the MO are orthonormal. If so, the Löwdin
                 orthogonalization is skipped
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
        mo_orth=mo_orth,
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


def transition_density(
    orbo: torch.Tensor, orbv: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Computes the transition density in the AO basis

    Computes the transition density in the AO basis, as:

        X^(e)_μν = Σ_i Σ_a X^(e)_ia C_μi C_νa

    for each excitation indexed by e.

    Args:
        orbo: coefficients of the occupied MOs, shape (nao, nocc)
        orbv: coefficients of the virtual MOs, shape (nao, nvir)
        x: transition amplitudes for each excited state, shape (nexc, nocc, nvir)
    Returns:
        x_ao: transition densities in the AO basis, shape (nexc, nao, nao)
    """
    return torch.einsum("eia,mi,na->emn", x, orbo, orbv) * 2


def transition_dipole(
    ints_ao: Union[torch.Tensor, Mol],
    orbo: torch.Tensor,
    orbv: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Computes the transition dipoles

    Computes the transition dipoles in the length gauge.
    Transition dipoles are computed for each excitation contained in x.

    Args:
        ints_ao: position integrals in the AO basis: <μ|r|ν>
                 should be provided with shape (3, nao, nao).
                 (pay attention to the origin when computing AO
                 integrals, e.g., use the center of charge as
                 origin)
                 If a pyscf.gto.Mol object is given, integrals
                 are computed using PySCF
        orbo: coefficients of the occupied MOs, shape (nao, nocc)
        orbv: coefficients of the virtual MOs, shape (nao, nvir)
        x: transition amplitudes for each excited state, shape (nexc, nocc, nvir)
    Returns:
        trn_dip: transition dipoles in atomic units, shape (nexc, 3)
    """
    if torch.is_tensor(ints_ao) or type(ints_ao) is np.ndarray:
        ints_ao = ensure_torch(ints_ao)
    else:
        # we assume it's a pyscf.gto.Mol object, if it
        # fails we let it fail
        def _charge_center(mol):
            # taken from pyscf
            charges = mol.atom_charges()
            coords = mol.atom_coords()
            return np.einsum("z,zr->r", charges, coords) / charges.sum()

        mol = ints_ao
        with mol.with_common_origin(_charge_center(mol)):
            ints_ao = torch.from_numpy(mol.intor_symmetric("int1e_r", comp=3))

    # convert position integrals in ao basis
    # to the mo basis (occupied - virtual block only)
    #
    # <i|r|a> = Σ_μν <μ|r|ν> C_μi C_νa
    #
    ints_mo = torch.einsum("umn,mi,na->uia", ints_ao, orbo, orbv)

    # contract position integrals in the mo basis with
    # the transition amplitudes from stda
    # 2 is for alpha + beta electrons
    #
    # μ^tr_e = Σ_ia <i|r|a> X_ia^(e)
    #
    trdip = torch.einsum("uia,eia->eu", ints_mo, x) * 2
    return trdip


def static_polarizability(
    ints_ao: Union[torch.Tensor, Mol],
    orbo: torch.Tensor,
    orbv: torch.Tensor,
    x: torch.Tensor,
    e: torch.Tensor,
) -> torch.Tensor:
    """Computes the static polarizability

    Computes the static polarizability using the sum over states
    (SOS) framework.

        α_ζη = Σ_n (<Ψ_n|μ_ζ|Ψ_0> <Ψ_0|μ_η|Ψ_n>) / (E_n - E_0)

    Args:
        ints_ao: position integrals in the AO basis: <μ|r|ν>
                 should be provided with shape (3, nao, nao).
                 (pay attention to the origin when computing AO
                 integrals, e.g., use the center of charge as
                 origin)
                 If a pyscf.gto.Mol object is given, integrals
                 are computed using PySCF
        orbo: coefficients of the occupied MOs
        orbv: coefficients of the virtual MOs
        x: transition amplitudes for each excited state
           should be provided with shape (nexc, nocc, nvir)
        e: excitation energies
    Returns:
        pol: polarizability tensor in atomic units, shape (3, 3)
    """
    trdip = transition_dipole(ints_ao=ints_ao, orbo=orbo, orbv=orbv, x=x)
    if len(trdip.shape) != 2:
        raise RuntimeError(f"wrong shape for transition dipoles: shape={trdip.shape}")
    if len(e.shape) != 1:
        raise RuntimeError(f"wrong shape for stda excitation energies: shape={e.shape}")
    pol = torch.einsum("em,en->mn", trdip / e[:, None], trdip) * 2
    return pol
