from __future__ import annotations
from typing import List, Tuple
import torch
from .parameters import chemical_hardness, get_alpha_beta
from .utils import sqrtm, physconst


def charge_density_monopole(
    ovlp: torch.Tensor,
    natm: int,
    ao_labels: List,
    mo_coeff_a: torch.Tensor,
    mo_coeff_b: torch.Tensor,
) -> torch.Tensor:
    """computes the q_pq^A using Löwdin population analysis.

        q_pq^A = Σ_(μ ϵ A) C'μp^(a) C'μq^(b)
        C' = S^(½) C

    Args:
        ovlp (n_ao, n_ao): AO overlap matrix (S).
        natm: number of atoms.
        ao_labels: list of tuples describing AOs.
                   same as calling mol.ao_labels(fmt=None) from pyscf.
                   tuple fmt:
                   (atom_index: int, atom: str, ao_name: str, m_def: str)
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
        R : matrix of pairwise distances.
    """
    xyz = coords * physconst.ang_to_bohr
    R = torch.cdist(xyz, xyz, p=2.0)
    return R


def hardness_matrix(atom_pure_symbols: List[str]) -> torch.Tensor:
    """computes the matrix of average chemical hardnesses

        η_ij = (η_i + η_j) / 2

    Args:
        atom_pure_symbols: list of atom symbols (e.g., ['O', 'H', 'H'] for water).
    Returns:
        η: matrix of average chemical hardness.
    """
    hrd = chemical_hardness
    eta = torch.DoubleTensor([hrd[sym] for sym in atom_pure_symbols])
    eta = (eta[:, None] + eta[None, :]) / 2
    return eta


def gamma_J(
    coords: torch.Tensor, atom_pure_symbols: List[str], ax: int, beta: int = None
) -> torch.Tensor:
    """computes the Coulomb gamma matrix (Matanaga-Nishimoto-Ohno-Klopman)

        γ(A, B)^J = (1 / ( R_AB^β + (ax * η)^-β  ))^(1/β)

    Args:
        coords (n_atoms, 3): coordinates of the molecule in Bohr.
        atom_pure_symbols: list of atom symbols (e.g., ['O', 'H', 'H'] for water).
        ax: fraction of exact Hartree Fock exchange.
        beta: β parameter of sTDA approximate integrals.
    Returns:
        γ(A, B)^J: matrix of Coulomb gamma values.
    """
    R = distance_matrix(coords)
    eta = hardness_matrix(atom_pure_symbols)
    if beta is None:
        _, beta = get_alpha_beta(ax)
    denom = ((R * ax * eta) ** beta + 1) ** (1.0 / beta)
    gamma = ax * eta / denom
    return gamma


def gamma_K(
    coords: torch.Tensor, atom_pure_symbols: List[str], ax: int, alpha: int = None
) -> torch.Tensor:
    """computes the Exchange gamma matrix (Matanaga-Nishimoto-Ohno-Klopman)

        γ(A, B)^K = (1 / (R_AB^α + η^-α))^(1/α)

    Args:
        coords (n_atoms, 3): coordinates of the molecule in Bohr.
        atom_pure_symbols: list of atom symbols (e.g., ['O', 'H', 'H'] for water).
        ax: fraction of exact Hartree Fock exchange.
        alpha: α parameter of sTDA approximate integrals.
    Returns:
        γ(A, B)^K: matrix of Coulomb gamma values.
    """
    R = distance_matrix(coords)
    eta = hardness_matrix(atom_pure_symbols)
    if alpha is None:
        alpha, _ = get_alpha_beta(ax)
    denom = ((R * eta) ** alpha + 1) ** (1.0 / alpha)
    gamma = eta / denom
    return gamma


def eri_mo_monopole(
    ovlp: torch.Tensor,
    natm: int,
    ao_labels: List,
    mo_coeff: torch.Tensor,
    mo_occ: torch.Tensor,
    coords: torch.Tensor,
    atom_pure_symbols: List[str],
    ax: int,
    alpha: int = None,
    beta: int = None,
    mode: str = "stda",
    mask_occ: torch.Tensor = None,
    mask_vir: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """computes the electron repulsion integrals in the sTDA approximation.

        (pq|rs) = Σ_(A,B) q_pq^A * q_rs^B * γ(A, B)

    if `mode` = 'stda', then it computes only the coulomb and exchange integrals:

        (ij|ab) = Σ_(A,B) q_ij^A * q_ab^B * γ(A, B)^J

        (ia|jb) = Σ_(A,B) q_ia^A * q_jb^B * γ(A, B)^K

    Args:
        ovlp (n_ao, n_ao): AO overlap matrix (S).
        natm: number of atoms.
        ao_labels: list of tuples describing AOs.
                          same as calling mol.ao_labels(fmt=None) from pyscf.
                          tuple fmt: (atom_index: int, atom: str, ao_name: str, m_def: str)
        mo_coeff (n_ao_a, n_mo): MO coefficients matrix (C).
        mo_occ (n_mo): MO occupancy.
        coords (n_atoms, 3): coordinates of the molecule in Bohr.
        atom_pure_symbols: list of atom symbols (e.g., ['O', 'H', 'H'] for water).
        ax: fraction of exact Hartree Fock exchange.
        alpha: α parameter of sTDA approximate integrals.
        beta: β parameter of sTDA approximate integrals.
        mode: which integral to compute, can be either 'stda' or 'full'.
              Note: 'full' is here only for debugging, and the only mode that
              makes sense is the default 'stda' mode.
        mask_occ: indices of occupied MOs to consider. The first occupied MO
                  has index 0.
        mask_vir: indices of virtual MOs to consider. The first virtual MO
                  has index 0.
    Returns:
        eri_J (n_mo_occ, n_mo_vir, n_mo_occ, n_mo_vir): electron repulsion integrals of Coulomb type.
        eri_K (n_mo_occ, n_mo_vir, n_mo_occ, n_mo_vir): electron repulsion integrals of Exchange type.
    """
    if mode != "stda" and mode != "full":
        raise RuntimeError(f"mode is either 'stda' or 'full', given '{mode}'")
    gam_J = gamma_J(coords, atom_pure_symbols, ax, beta)
    gam_K = gamma_K(coords, atom_pure_symbols, ax, alpha)
    if mode == "full":
        q = charge_density_monopole(ovlp, natm, ao_labels, mo_coeff, mo_coeff)
        eri_J = torch.einsum("Apq,AB,Brs->pqrs", q, gam_J, q)
        eri_K = torch.einsum("Apq,AB,Brs->pqrs", q, gam_K, q)
    elif mode == "stda":
        occidx = torch.where(mo_occ == 2)[0]
        viridx = torch.where(mo_occ == 0)[0]
        if mask_occ is not None:
            occidx = occidx[mask_occ]
        if mask_vir is not None:
            viridx = viridx[mask_vir]

        q_oo = charge_density_monopole(
            ovlp, natm, ao_labels, mo_coeff[:, occidx], mo_coeff[:, occidx]
        )
        q_ov = charge_density_monopole(
            ovlp, natm, ao_labels, mo_coeff[:, occidx], mo_coeff[:, viridx]
        )
        q_vv = charge_density_monopole(
            ovlp, natm, ao_labels, mo_coeff[:, viridx], mo_coeff[:, viridx]
        )
        eri_J = torch.einsum("Aij,AB,Bab->iajb", q_oo, gam_J, q_vv)
        eri_K = torch.einsum("Aia,AB,Bjb->iajb", q_ov, gam_K, q_ov)
    return eri_J, eri_K
