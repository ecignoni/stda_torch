from __future__ import annotations
from typing import IO, Union

import numpy as np
from collections import defaultdict
import pyscf
from pyscf import lib

import torch


def asnumpy(t):
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()


def atoms_molden_section(mol: pyscf.gto.mole.Mole, fout: IO) -> None:
    """writes the [Atoms] section of a molden file.

    This function tries to be compatible with the output
    of `g2molden` in the original sTDA program (Grimme).
    """
    fout.write("[Atoms] Angs\n")

    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        chg = mol.atom_charge(ia)
        fout.write("%s   %5d   %5d   " % (symb, ia + 1, chg))
        coord = mol.atom_coord(ia, unit="Ang")
        fout.write("%18.10f   %18.10f   %18.10f\n" % tuple(coord))


def gto_molden_section(mol: pyscf.gto.mole.Mole, fout: IO) -> None:
    """writes the [GTO] section of a molden file.

    This function tries to be compatible with the output
    of `g2molden` in the original sTDA program (Grimme).
    """
    fout.write("[GTO]\n")

    for ia, (sh0, sh1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
        fout.write("  %d 0\n" % (ia + 1))

        # to be compatible with the g2molden output, we need
        # to "contract" the AO basis, listing together coefficients
        # with the same exponent
        gto = defaultdict(list)
        ls = defaultdict(list)
        for i, ib in enumerate(range(sh0, sh1)):
            l = mol.bas_angular(ib)  # noqa
            nprim = mol.bas_nprim(ib)
            nctr = mol.bas_nctr(ib)
            es = mol.bas_exp(ib)
            cs = mol.bas_ctr_coeff(ib)
            gto[(nprim, nctr, tuple(es))].append(cs)
            ls[(nprim, nctr, tuple(es))].append(l)

        for key in gto.keys():
            nprim, nctr, es = key
            cntr = np.column_stack(gto[key])
            topr = np.column_stack((es, cntr))
            ll = "".join([lib.param.ANGULAR[l] for l in ls[key]])  # noqa
            fout.write("  {:<3s} {:<3d} 1.00\n".format(ll, nprim))
            for row in topr:
                write = (
                    "  " + "".join(["{:<20.10e}".format(elem) for elem in row]) + "\n"
                )
                write = write.replace("e", "d")
                fout.write(write)

        fout.write("\n")


def reorder_ao_labels(mol: pyscf.gto.mole.Mole) -> np.ndarray:
    "reorders the AO from pyscf to gaussian order"
    ao_labels = mol.ao_labels(fmt=None)

    indices = []
    i = 0
    while True:
        try:
            t = ao_labels[i][2][-1]
        except IndexError:
            break
        if t == "s":
            indices.append([i])
            i += 1
        elif t == "p":
            # x, y, z
            indices.append([i, i + 1, i + 2])
            i += 3
        elif t == "d":
            # xx, yy, zz, xy, xz, yz
            indices.append([i, i + 3, i + 5, i + 1, i + 2, i + 4])
            i += 6
        else:
            raise RuntimeError(f"t={t} not implemented.")

    # flatten the list of lists into a simple list
    indices = np.array([item for sublist in indices for item in sublist])

    # now reorder the shells
    # this is super ugly and may break easily but I really
    # hate to write this
    new_indices = []
    atom_is = np.array([lbl[0] for lbl in ao_labels])
    ns = np.array([int(lbl[2][0]) for lbl in ao_labels])
    for i in np.unique(atom_is):
        # selects the AO corresponding to the i-th atom
        idx = np.where(atom_is == i)[0]
        # sort according to the n values
        idx = idx[np.argsort(ns[idx])]
        # append reordered indices
        new_indices.append(indices[idx])

    indices = np.concatenate(new_indices)
    return indices


def mo_molden_section(
    fout: IO,
    mo_energy: Union[torch.Tensor, np.ndarray],
    mo_occ: Union[torch.Tensor, np.ndarray],
    mo_coeff: Union[torch.Tensor, np.ndarray],
    mol: pyscf.gto.mole.Mole,
) -> None:
    """writes the [MO] section of a molden file.

    This functions tries to be compatible with the
    output of `g2molden` in the original sTDA program.
    """
    fout.write("[MO]\n")

    # work with numpy here
    mo_energy = asnumpy(mo_energy)
    mo_occ = asnumpy(mo_occ)
    mo_coeff = asnumpy(mo_coeff)

    # check for double occupancy
    unique_occ = np.unique(mo_occ)
    for o in unique_occ:
        if o not in [2, 0]:
            raise ValueError("only double occupancies are supported.")

    if mol.cart:
        # pyscf Cartesian GTOs are not normalized. This may not be consistent
        # with the requirements of molden format. Normalize Cartesian GTOs here
        norm = mol.intor("int1e_ovlp").diagonal() ** 0.5
        mo_coeff = np.einsum("i,ij->ij", norm, mo_coeff)
    else:
        raise ValueError("only cartesian basis is supported, but mol.cart={mol.cart}")

    # reorder the AOs to match the order of Gaussian
    idx = reorder_ao_labels(mol)
    mo_coeff = mo_coeff[idx]

    for i, (ene, occ, coeff) in enumerate(zip(mo_energy, mo_occ, mo_coeff.T)):
        fout.write(" Sym= {:^20d}\n".format(i + 1))
        fout.write(" Ene= {:20.13f}\n".format(ene))
        fout.write(" Spin= Alpha\n")
        fout.write(" Occup= {:10.6f}\n".format(occ))
        for j, c in enumerate(coeff):
            fout.write(" {:>5d} {:20.13f}\n".format(j + 1, round(c, ndigits=5)))


def to_stda_molden(
    outfile: str,
    mol: pyscf.gto.mole.Mole,
    mo_coeff: Union[torch.Tensor, np.ndarray],
    mo_occ: Union[torch.Tensor, np.ndarray],
    mo_energy: Union[torch.Tensor, np.ndarray],
) -> None:
    """writes a molden file in a format accepted by sTDA (Grimme)

    Note that sTDA (Grimme) only accepts a cartesian basis.
    Args:
        mol: a PySCF Mole object
        mo_coeff: MO coefficients, shape (n_ao, n_mo)
        mo_occ: occupancy of the MOs (only supports double occupancies)
        mo_energy: MO energies
    """
    with open(outfile, "w") as handle:
        handle.write("[Molden Format]\n")
        atoms_molden_section(mol, fout=handle)
        gto_molden_section(mol, fout=handle)
        mo_molden_section(handle, mo_energy, mo_occ, mo_coeff, mol)
