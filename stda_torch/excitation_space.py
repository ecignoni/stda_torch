from __future__ import annotations
from typing import Tuple, Union
import torch
from .utils import physconst


def screen_mo(
    mo_energy: torch.Tensor,
    mo_occ: torch.Tensor,
    ax: int,
    e_max: int = 7.0,
    verbose: bool = False,
    mesh_idx: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """computes the indices of retained occupied and virtual MOs.
    MOs are retained if their energy falls within a window around the
    HOMO and LUMO. The window is computed as:

        w = 2 * (1 + 0.8 * a_x) * ε_max

    occupied MOs are retained if their energy falls in the interval

        (ε_LUMO - w, ε_LUMO)

    virtual MOs are retained if their energy falls in the interval

        (ε_HOMO, ε_HOMO + w)

    Args:
        mo_energy (n_mo): MO energies (ε).
        mo_occ (n_mo): MO occupancy.
        ax: fraction of exact Hartree Fock exchange.
        e_max: energy threshold (eV) for MO screening.
        verbose: whether to be verbose.
        mesh_idx: whether to create a mask as an open mesh from the
                  occupied - virtual - occupied - virtual retained indices.
                  In this case, a tuple of four torch.Tensor is returned.
    Returns:
        mask_occ: indices of retained occupied MO.
        mask_vir: indices of retained virtual MO.

    Note:
        each returned mask starts from 0, e.g., if only one virtual
        MO is retained, then mask_vir value is tensor([0]).
    """
    occidx = torch.where(mo_occ == 2)[0]
    viridx = torch.where(mo_occ == 0)[0]

    window = 2 * (1.0 + 0.8 * ax) * (e_max / physconst.au_to_ev)
    vthr = torch.max(mo_energy[occidx]) + window
    othr = torch.min(mo_energy[viridx]) - window

    mask_occ = torch.where(mo_energy[occidx] > othr)[0]
    mask_vir = torch.where(mo_energy[viridx] < vthr)[0]

    if verbose:
        print("%-40s = %15.8f" % ("spectral range up to (eV)", e_max))
        print("%-40s = %15.8f" % ("occ MO cut-off (eV)", othr * physconst.au_to_ev))
        print("%-40s = %15.8f" % ("virtMO cut-off (eV)", vthr * physconst.au_to_ev))

    if mesh_idx:
        return (
            mask_occ[:, None, None, None],
            mask_vir[None, :, None, None],
            mask_occ[None, None, :, None],
            mask_vir[None, None, None, :],
        )
    else:
        return mask_occ, mask_vir, othr, vthr


def select_csf_by_energy(
    a: torch.Tensor, e_max: float, verbose: bool = False
) -> Union[torch.Tensor, torch.Tensor]:
    """selects CSFs according to their energy

    Given a threshold ε_max, divides the set of CSF in A in two:
    P-CSF (primary CIS configurations) and N-CSF (neglected CIS
    configurations). A CSF is classified as P-CSF if its energy
    is below the threshold:

        ε_ia <= ε_max => P-CSF
        e_ia >  ε_max => N-CSF

    Args:
        a: A matrix of the Casida equations
        e_max: energy threshold of sTDA
        verbose: whether to be verbose
    Returns:
        idx_pcsf: indices of P-CSF
        idx_ncsf: indices of N-CSF
    """
    _e_max = e_max / physconst.au_to_ev
    nocc, nvir, _, _ = a.shape
    diag_a = torch.diag(a.reshape(nocc * nvir, nocc * nvir))

    idx_pcsf = torch.where(diag_a <= _e_max)[0]
    idx_ncsf = torch.where(diag_a > _e_max)[0]

    if idx_pcsf.size()[0] == 0:
        errmsg = f"No CSF below the energy threshold ({e_max} eV),"
        errmsg += " you may want to increase it"
        raise RuntimeError(errmsg) from None

    if verbose:
        print("%d CSF included by energy" % len(idx_pcsf))
        print("%d considered in PT2" % len(idx_ncsf))

    return idx_pcsf, idx_ncsf


def csf_idx_as_ia(idx_csf: torch.Tensor, nvir: int) -> torch.Tensor:
    """rewrites the indices of a CSF

    Given a set of CSF indices and the number of virtual orbitals,
    rewrites the CSF indices as a pair [i, a], where i is the index
    of an occupied orbital, and a is the index of a virtual orbital.
    Note that a indices start from 0 (the first virtual orbital).

    Args:
        idx_csf: indices of a set of CSFs
        nvir: number of virtual MOs
    Returns:
        new_idx: indices of the set of CSFs, rewritten as [i, a] pairs
    """
    csf_i = torch.div(idx_csf, nvir, rounding_mode="trunc")
    csf_a = torch.remainder(idx_csf, nvir)
    return torch.column_stack((csf_i, csf_a))


def select_csf_by_perturbation(
    a: torch.Tensor,
    idx_pcsf: torch.Tensor,
    idx_ncsf: torch.Tensor,
    e_max: int,
    tp: float,
    diag_a: torch.Tensor = None,
    verbose: bool = False,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    """selects CSFs using perturbation theory

    Given a set of P-CSFs (primary) and a set of N-CSFs (neglected),
    evaluates the perturbative contribution of the N-CSFs and possibly
    retains some of the N-CSFs. The retained set of N-CSFs is called
    S-CSFs (secondary CI configurations).

    The perturbative contribution is evaluated as:

        E_u^(2) = Σ_v^(P-CSF) |A_uv|^2 / (E_u - E_v)

        where u=ia and v=jb

    A N-CSF indexed by u is relabeled S-CSF if E_u^(2) >= τ, where
    τ is a user-defined threshold.

    The perturbative energy of the remaining N-CSF is summed for each
    P-CSF and returned:

        E_v^(2) = Σ_u^(N-CSF) |A_uv|^2 / (E_u - E_v)

    Args:
        a: A matrix of the Casida equations
        idx_pcsf: indices of P-CSFs
        idx_ncsf: indices of N-CSFs
        e_max: energy threshold of sTDA
        tp: perturbative threshold of sTDA (τ)
        diag_a: diagonal elements of A
        verbose: whether to be verbose
    Returns:
        idx_scsf: indices of S-CSFs
        idx_ncsf: indices of N-CSFs
        e_pt_ncsf: perturbative interaction of N-CSFs to P-CSFs
    """
    nocc, nvir, _, _ = a.shape
    if diag_a is None:
        diag_a = torch.diag(a.reshape(nocc * nvir, nocc * nvir))
    a_uv = a.reshape(nocc * nvir, nocc * nvir)[idx_ncsf, :][:, idx_pcsf]
    denom = diag_a[idx_ncsf, None] - diag_a[idx_pcsf]

    e_pt = torch.divide(
        a_uv**2,
        denom,
    )
    e_u = torch.sum(e_pt, axis=1)

    idx_scsf = idx_ncsf[e_u >= tp]
    idx_ncsf = idx_ncsf[e_u < tp]

    if verbose:
        print("%d CSF included by PT" % len(idx_scsf))
        print("%d CSF in total" % (len(idx_scsf) + len(idx_pcsf)))

    # perturbative contribution
    e_pt_ncsf = -torch.sum(e_pt[e_u < tp], axis=0)

    return idx_scsf, idx_ncsf, e_pt_ncsf


def restrict_to_stda_excitation_space(
    a: torch.Tensor,
    b: torch.Tensor,
    idx_pcsf: torch.Tensor,
    idx_scsf: torch.Tensor,
    e_pt_ncsf: torch.Tensor,
) -> Union[torch.Tensor, torch.Tensor]:
    """yields new A, B matrices with truncated CI space

    Given the set of indices for P-CSFs and S-CSFs, the active
    space in sTDA, yields the corresponding A and B matrices
    truncated to that subspace.

    The perturbative interaction of N-CSFs with P-CSFs is summed
    to the corresponding diagonal elements of A.

    Args:
        a: A matrix of the Casida equation
        b: B matrix of the Casida equation
        idx_pcsf: indices of P-CSFs
        idx_scsf: indices of S-CSFs
        e_pt_ncsf: perturbative interaction of N-CSFs with P-CSFs
    Returns:
        a: A matrix truncated to the sTDA CI space
        b: B matrix truncated to the sTDA CI space
    """
    nocc, nvir, _, _ = a.shape

    a = a.reshape(nocc * nvir, nocc * nvir)
    b = b.reshape(nocc * nvir, nocc * nvir)

    a[idx_pcsf, idx_pcsf] += e_pt_ncsf

    idx_active = torch.concatenate((idx_pcsf, idx_scsf))

    a = a[idx_active, :][:, idx_active]
    b = b[idx_active, :][:, idx_active]

    return a, b
