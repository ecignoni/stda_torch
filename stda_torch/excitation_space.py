from typing import Tuple
import torch

AU_TO_EV = 27.211324570273


def screen_mo(
    mo_energy: torch.Tensor,
    mo_occ: torch.Tensor,
    ax: int,
    e_max: int = 7.0,
    verbose: bool = False,
    mesh_idx: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    window = 2 * (1.0 + 0.8 * ax) * (e_max / AU_TO_EV)
    vthr = torch.max(mo_energy[occidx]) + window
    othr = torch.min(mo_energy[viridx]) - window

    mask_occ = torch.where(mo_energy[occidx] > othr)[0]
    mask_vir = torch.where(mo_energy[viridx] < vthr)[0]

    if verbose:
        print("%-40s = %15.8f" % ("spectral range up to (eV)", e_max))
        print("%-40s = %15.8f" % ("occ MO cut-off (eV)", othr * AU_TO_EV))
        print("%-40s = %15.8f" % ("virtMO cut-off (eV)", vthr * AU_TO_EV))

    if mesh_idx:
        return (
            mask_occ[:, None, None, None],
            mask_vir[None, :, None, None],
            mask_occ[None, None, :, None],
            mask_vir[None, None, None, :],
        )
    else:
        return mask_occ, mask_vir
