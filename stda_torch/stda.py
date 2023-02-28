from __future__ import annotations
from typing import List, Tuple

import warnings
import time
from datetime import datetime

from .parameters import get_alpha_beta
from .utils import symbol_to_charge, direct_diagonalization, physconst, normalize_ao
from .excitation_space import screen_mo, csf_idx_as_ia
from .linear_response import get_ab
from .integrals import charge_density_monopole

import torch


class sTDAVerboseMixin:
    def welcome(self):
        if self.verbose:
            print(datetime.now())
            print("\n\n   " + "*" * 50)
            print("   " + "*" + " " * 48 + "*")
            print("   " + "*" + " " * 20 + "s T D A " + " " * 20 + "*")
            print("   " + "*" + " " * 48 + "*")
            print("   " + "*" * 50 + "\n")

    def all_credits_to_grimme(self):
        if self.verbose:
            print("This is a PyTorch implementation of sTDA.")
            print("The original sTDA is the work of Stefan Grimme.")
            print()
            print("S. Grimme, J. Chem. Phys. 138 (2013) 244104")
            print("M. de Wergifosse, S. Grimme, J. Phys. Chem A")
            print("125 (2021) 18 3841-3851\n")

    def stda_section(self):
        if self.verbose:
            print("=" * 65)
            print(" " * 28 + "s T D A ")
            print("=" * 65)

    def mo_ao_input(self):
        if self.verbose:
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

    def thresholds_info(self):
        if self.verbose:
            print("{:30s} : {:.8f}".format("spectral range up to (eV)", self.e_max))
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

    def scf_atom_population(self):
        qij = (
            charge_density_monopole(
                ovlp=self.ovlp,
                natm=self.natm,
                ao_labels=self.ao_labels,
                mo_coeff_a=self.mo_coeff[:, self.mask_occ],
                mo_coeff_b=self.mo_coeff[:, self.mask_occ],
                mo_orth=self.mo_orth,
            )
            * 2  # noqa: W503
        )
        pop = torch.einsum("Aii->A", qij)
        nelec = torch.sum(pop)

        if self.verbose:
            print("\nSCF atom population (using active MOs):")
            print(" ", pop)
            print("\n# electrons in TDA: {:.3f}".format(nelec))

        # raise an exception if the number of electrons is not an integer
        if abs(nelec - len(self.mask_occ) * 2) > 1e-2:
            errmsg = "strange number of electrons in sTDA, "
            errmsg += "perhaps is worth checking the MO coefficients"
            raise RuntimeError(errmsg)

    def parameters_info(self):
        if self.verbose:
            print("\n{:20s}: {:.8f}".format("ax(DF)", self.ax))
            print("{:20s}: {:.8f}".format("beta (J)", self.beta))
            print("{:20s}: {:.8f}".format("alpha (K)", self.alpha))

    def selection_by_energy(self):
        if self.verbose:
            print("\n{:d} CSF included by energy.".format(len(self.idx_pcsf)))
            print(
                "{:d} considered in PT2.".format(
                    len(self.idx_ncsf) + len(self.idx_scsf)
                )
            )

    def selection_by_pt(self):
        if self.verbose:
            print("\n{:d} CSF included by PT.".format(len(self.idx_scsf)))
            print("{:d} CSF in total.".format(len(self.idx_pcsf) + len(self.idx_scsf)))

    def diag_info(self, diag_time):
        if self.verbose:
            print("estimated time (min)      {:.1f}".format((diag_time) / 60.0))
            print(
                "\t{:d} roots found, lowest/highest eigenvalue: {:.3f} {:.3f}".format(
                    self.e.shape[0],
                    torch.min(self.e) * physconst.au_to_ev,
                    torch.max(self.e) * physconst.au_to_ev,
                )
            )

    def ordered_frontier_orbitals(self):
        if self.verbose:
            print("ordered frontier orbitals")
            print("{:8s} {:8s} {:8s}".format("", "eV", "# centers"))

            ene_occ = self.mo_energy[self.mask_occ]
            qij = charge_density_monopole(
                ovlp=self.ovlp,
                natm=self.natm,
                ao_labels=self.ao_labels,
                mo_coeff_a=self.mo_coeff[:, self.mask_occ],
                mo_coeff_b=self.mo_coeff[:, self.mask_occ],
                mo_orth=self.mo_orth,
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
                mo_orth=self.mo_orth,
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

    def excens_and_amplitudes(self):
        if self.verbose:
            occ_idx = torch.arange(self.nocc)
            vir_idx = torch.arange(self.nvir)
            indices = torch.LongTensor(
                [[o, self.nocc + v] for o in occ_idx for v in vir_idx]
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

    def normalize_ao_basis(self):
        if not self.mo_orth:
            # if the MO coefficients need to be orthonormalized
            # we assume the 'ovlp' given in input is a valid one
            self.mo_coeff, self.ovlp = normalize_ao(
                mo_coeff=self.mo_coeff, ovlp=self.ovlp
            )
        else:
            # if the MO are orthonormalized, then 'ovlp' is ignored
            # and everything (e.g., None) can be passed as input
            # in this case print a warning as the user must be free
            # to avoid computing the overlap (e.g., if the mo coeffs
            # are the output of a ML model)
            errmsg = "\nWarning: you are passing Lowdin orthogonalized MOs, "
            errmsg += "make sure they are computed in a normalized AO basis.\n"
            warnings.warn(errmsg)

    def stda_done(self):
        if self.verbose:
            print("\nsTDA done.")


class sTDA(sTDAVerboseMixin):
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
        mo_orth: bool = False,
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
            mo_orth: whether the MO are orthonormal. If so, 'ovlp' is ignored
                     and Löwdin orthogonalization is skipped
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
        self.mo_orth = mo_orth
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

    def kernel(self, nstates=3):
        self.nstates = nstates

        self.welcome()
        self.all_credits_to_grimme()
        self.mo_ao_input()
        self.stda_section()

        # pyscf cartesian basis is not normalized
        # in general, make sure you are using a normalized basis
        self.normalize_ao_basis()

        self.mask_occ, self.mask_vir, self.occthr, self.virthr = screen_mo(
            mo_energy=self.mo_energy, mo_occ=self.mo_occ, ax=self.ax, e_max=self.e_max
        )

        self.thresholds_info()

        # check on number of electrons + print if verbose
        self.scf_atom_population()

        self.parameters_info()

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
            mo_orth=self.mo_orth,
            verbose=False,
        )

        self.idx_pcsf, self.idx_scsf, self.idx_ncsf, self.e_pt_ncsf = meta

        nvir = len(self.mask_vir)
        self.pcsf = csf_idx_as_ia(self.idx_pcsf, nvir)
        self.scsf = csf_idx_as_ia(self.idx_scsf, nvir)
        self.ncsf = csf_idx_as_ia(self.idx_ncsf, nvir)

        self.selection_by_energy()
        self.ordered_frontier_orbitals()
        self.selection_by_pt()

        if self.verbose:
            print("diagonalizing...")

        start = time.perf_counter()
        self.e, x = direct_diagonalization(a, nstates=self.nstates)
        end = time.perf_counter()

        self.diag_info(end - start)

        # sTDA transition amplitudes
        self.nocc, self.nvir = len(self.mask_occ), len(self.mask_vir)
        xy = [(xi, 0.0) for xi in x.T]
        active = torch.concatenate((self.pcsf, self.scsf))
        self.x = []
        self.y = []
        for x, y in xy:
            new_x = torch.zeros((self.nocc, self.nvir))
            new_y = torch.zeros((self.nocc, self.nvir))
            new_x[active[:, 0], active[:, 1]] = x
            self.x.append(new_x)
            self.y.append(new_y)
        self.x = torch.stack(self.x)
        self.y = torch.stack(self.y)

        self.excens_and_amplitudes()

        self.stda_done()

        return self.e
