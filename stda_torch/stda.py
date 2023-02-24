from __future__ import annotations
from typing import List, Tuple

import time
from datetime import datetime

from .parameters import get_alpha_beta
from .utils import symbol_to_charge, direct_diagonalization, physconst
from .excitation_space import screen_mo, csf_idx_as_ia
from .linear_response import get_ab
from .integrals import charge_density_monopole

import torch


class sTDA:
    "simplified Tamm-Dancoff approximation"

    def __init__(
        self,
        mo_energy: torch.Tensor,
        mo_coeff: torch.Tensor,
        mo_occ: torch.Tensor,
        ovlp: torch.Tensor,
        natm: int,
        ao_labels: List[Tuple[int, str, str, str]],
        coords: torch.Tensor,
        atom_pure_symbols: List[str],
        ax: float,
        alpha: float = None,
        beta: float = None,
        e_max: float = 7.5,
        tp: float = 1e-4,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            mo_energy: MO energies ε (AU)
            mo_coeff: MO coefficients C (not orthonormal)
            mo_occ: MO occupancies (only double occupancies are supported)
            ovlp: overlap matrix S
            natm: number of atoms
            ao_labels: labels of the cartesian AO (they have to be cartesian)
                       (atom_index, atom_name, n_symbol, aux_symbol)
                       (can also provide a 2-tuple, only the first element is used)
            coords: coordinates in Angstrom
            atom_pure_symbols: symbols of the atoms ('C' for carbon, 'O' for oxygen, ...)
            ax: percentage of exact Hartree-Fock exchange
            alpha: parameter α of sTDA (exchange integral)
            beta: parameter β of sTDA (coulomb integral)
            e_max: energy threshold of sTDA
            tp: perturbative threshold of sTDA
            verbose: whether to be verbose
        """
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.check_restricted()
        self.ovlp = ovlp
        self.natm = natm
        self.ao_labels = ao_labels
        self.coords = coords
        self.atom_pure_symbols = atom_pure_symbols
        self.ax = ax
        self._ax_alpha, self._ax_beta = get_alpha_beta(ax)
        self.alpha = alpha
        self.beta = beta
        self.e_max = e_max
        self.tp = tp
        self.verbose = verbose

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = self._ax_alpha if value is None else value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = self._ax_beta if value is None else value

    def check_restricted(self):
        occ = self.mo_occ[self.mo_occ != 0]
        if not torch.all(occ == 2):
            errmsg = (
                "This sTDA implementation only works for closed shell calculations, "
            )
            errmsg += "but you provided MO with occupancies different from 2"
            raise ValueError(errmsg)

    def welcome(self):
        print(datetime.now())
        print("\n\n   " + "*" * 50)
        print("   " + "*" + " " * 48 + "*")
        print("   " + "*" + " " * 20 + "s T D A " + " " * 20 + "*")
        print("   " + "*" + " " * 48 + "*")
        print("   " + "*" * 50 + "\n")

    def all_credits_to_grimme(self):
        print("This is a PyTorch implementation of sTDA.")
        print("The original sTDA is the work of Stefan Grimme.")
        print()
        print("S. Grimme, J. Chem. Phys. 138 (2013) 244104")
        print("M. de Wergifosse, S. Grimme, J. Phys. Chem A")
        print("125 (2021) 18 3841-3851\n")

    def mo_ao_input(self):
        print("=" * 65)
        print(" " * 22 + "M O / A O   I N P U T ")
        print("=" * 65)
        title = "{:3s} {:^15s} {:^15s} {:^15s} {:^10s}".format(
            "atom #", "x", "y", "z", "charge"
        )
        print(title)
        for symbol, (x, y, z) in zip(self.atom_pure_symbols, self.coords):
            charge = symbol_to_charge[symbol]
            row = "%3s %15.8f %15.8f %15.8f %10.1f" % (
                symbol,
                x.item(),
                y.item(),
                z.item(),
                charge,
            )
            print(row)

        print()
        print(" {:20s} = {:d}".format("# atoms", self.natm))
        print(" {:20s} = {:d}".format("# mos", self.mo_coeff.shape[0]))
        print(" {:20s} = {:s}".format("# primitive aos", "not provided"))
        print(" {:20s} = {:d}".format("# contracted aos", len(self.ao_labels)))

    def print_ordered_frontier_orbitals(self):
        print("ordered frontier orbitals")
        print("{:8s} {:8s} {:8s}".format("", "eV", "# centers"))

        ene_occ = self.mo_energy[self.mask_occ]
        qij = charge_density_monopole(
            ovlp=self.ovlp,
            natm=self.natm,
            ao_labels=self.ao_labels,
            mo_coeff_a=self.mo_coeff[:, self.mask_occ],
            mo_coeff_b=self.mo_coeff[:, self.mask_occ],
        )
        qij = torch.einsum("Aii->Ai", qij)
        centers = 1.0 / torch.einsum("Ai->i", qij**2)
        for i, (e, c) in enumerate(zip(ene_occ, centers)):
            print("{:^8d} {:^8.3f} {:.1f}".format(i + 1, e * physconst.au_to_ev, c))
            if i == 10:
                break

        print()
        nocc = sum(self.mo_occ == 2)
        ene_vir = self.mo_energy[nocc + self.mask_vir]
        qab = charge_density_monopole(
            ovlp=self.ovlp,
            natm=self.natm,
            ao_labels=self.ao_labels,
            mo_coeff_a=self.mo_coeff[:, nocc + self.mask_vir],
            mo_coeff_b=self.mo_coeff[:, nocc + self.mask_vir],
        )
        qab = torch.einsum("Auu->Au", qab)
        centers = 1.0 / torch.einsum("Au->u", qab**2)
        for i, (e, c) in enumerate(zip(ene_vir, centers)):
            print(
                "{:^8d} {:^8.3f} {:.1f}".format(
                    i + len(self.mask_occ) + 1, e * physconst.au_to_ev, c
                )
            )
            if i == 10:
                break

    def kernel(self, nstates=3):
        self.nstates = nstates
        if self.verbose:
            self.welcome()
            self.all_credits_to_grimme()
            self.mo_ao_input()

            print("=" * 65)
            print(" " * 28 + "s T D A ")
            print("=" * 65)
            print("{:30s} : {:.8f}".format("spectral range up to (eV)", self.e_max))

        self.mask_occ, self.mask_vir, self.occthr, self.virthr = screen_mo(
            mo_energy=self.mo_energy, mo_occ=self.mo_occ, ax=self.ax, e_max=self.e_max
        )

        if self.verbose:
            print(
                "{:30s} : {:.8f}".format(
                    "occ MO cut-off (eV)", self.occthr * physconst.au_to_ev
                )
            )
            print(
                "{:30s} : {:.8f}".format(
                    "virtMO cut-off (eV)", self.virthr * physconst.au_to_ev
                )
            )
            print("{:30s} : {:.8f}".format("perturbation thr", self.tp))
            print("{:30s} : {:.8s}".format("triplet", "F"))
            print(
                "{:15s}: {:d}".format(
                    "MOs in TDA", len(self.mask_occ) + len(self.mask_vir)
                )
            )
            print("{:15s}: {:d}".format("oMOs in TDA", len(self.mask_occ)))
            print("{:15s}: {:d}".format("vMOs in TDA", len(self.mask_vir)))

            print("\nSCF atom population (using active MOs):")
            qij = (
                charge_density_monopole(
                    ovlp=self.ovlp,
                    natm=self.natm,
                    ao_labels=self.ao_labels,
                    mo_coeff_a=self.mo_coeff[:, self.mask_occ],
                    mo_coeff_b=self.mo_coeff[:, self.mask_occ],
                )
                * 2  # noqa: W503
            )
            pop = torch.einsum("Aii->A", qij)
            print(" ", pop)

            print("\n# electrons in TDA: {:.3f}".format(torch.sum(pop)))

            print("\n")
            print("{:20s}: {:.8f}".format("ax(DF)", self.ax))
            print("{:20s}: {:.8f}".format("beta (J)", self.beta))
            print("{:20s}: {:.8f}".format("alpha (K)", self.alpha))

        a, b, *meta = get_ab(
            mo_energy=self.mo_energy,
            mo_coeff=self.mo_coeff,
            mo_occ=self.mo_occ,
            ovlp=self.ovlp,
            natm=self.natm,
            ao_labels=self.ao_labels,
            coords=self.coords,
            atom_pure_symbols=self.atom_pure_symbols,
            ax=self.ax,
            alpha=self.alpha,
            beta=self.beta,
            mask_occ=self.mask_occ,
            mask_vir=self.mask_vir,
            excitation_space="stda",
            e_max=self.e_max,
            tp=self.tp,
            verbose=False,
        )

        self.idx_pcsf, self.idx_scsf, self.idx_ncsf, self.e_pt_ncsf = meta

        nvir = len(self.mask_vir)
        self.pcsf = csf_idx_as_ia(self.idx_pcsf, nvir)
        self.scsf = csf_idx_as_ia(self.idx_scsf, nvir)
        self.ncsf = csf_idx_as_ia(self.idx_ncsf, nvir)

        if self.verbose:
            print("\n{:d} CSF included by energy.".format(len(self.idx_pcsf)))
            print(
                "{:d} considered in PT2.".format(
                    len(self.idx_ncsf) + len(self.idx_scsf)
                )
            )
            self.print_ordered_frontier_orbitals()

            print("\n{:d} CSF included by PT.".format(len(self.idx_scsf)))
            print("{:d} CSF in total.".format(len(self.idx_pcsf) + len(self.idx_scsf)))

            print("diagonalizing...")

        start = time.perf_counter()
        self.e, x = direct_diagonalization(a, nstates=self.nstates)
        end = time.perf_counter()

        if self.verbose:
            print("estimated time (min)      {:.1f}".format((end - start) / 60.0))
            print(
                "\t{:d} roots found, lowest/highest eigenvalue: {:.3f} {:.3f}".format(
                    self.e.shape[0],
                    torch.min(self.e) * physconst.au_to_ev,
                    torch.max(self.e) * physconst.au_to_ev,
                )
            )

        # sTDA transition amplitudes
        nocc, nvir = len(self.mask_occ), len(self.mask_vir)
        xy = [(xi, 0.0) for xi in x.T]
        active = torch.concatenate((self.pcsf, self.scsf))
        self.x = []
        self.y = []
        for x, y in xy:
            new_x = torch.zeros((nocc, nvir))
            new_y = torch.zeros((nocc, nvir))
            new_x[active[:, 0], active[:, 1]] = x
            self.x.append(new_x)
            self.y.append(new_y)
        self.x = torch.stack(self.x)
        self.y = torch.stack(self.y)

        if self.verbose:
            occ_idx = torch.arange(nocc)
            vir_idx = torch.arange(nvir)
            indices = torch.LongTensor(
                [[o, nocc + v] for o in occ_idx for v in vir_idx]
            )
            print("\nexcitation energies, transition moments and TDA amplitudes")
            print("state    eV      nm       fL        Rv(corr)")
            for i, (e, x) in enumerate(zip(self.e, self.x)):
                _, top3indices = torch.topk(abs(x.reshape(-1)), 3)
                top3_ia_pairs = indices[top3indices]
                top3values = x.reshape(-1)[top3indices]
                print(
                    "{:<5d} {:^8.3f} {:^8.1f} {:^9s} {:^9s} {:6.2f}({:4d}->{:4d}) {:6.2f}({:4d}->{:4d}) {:6.2f}({:4d}->{:4d})".format(
                        i + 1,
                        e * physconst.au_to_ev,
                        1e7 / (e * 2.19474625e5),
                        "n.a.",
                        "n.a.",
                        top3values[0],
                        top3_ia_pairs[0][0] + 1,
                        top3_ia_pairs[0][1] + 1,
                        top3values[1],
                        top3_ia_pairs[1][0] + 1,
                        top3_ia_pairs[1][1] + 1,
                        top3values[2],
                        top3_ia_pairs[2][0] + 1,
                        top3_ia_pairs[2][1] + 1,
                    )
                )

        if self.verbose:
            print("\nsTDA done.")

        return self.e
