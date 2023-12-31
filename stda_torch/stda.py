from __future__ import annotations
from typing import List, Tuple, Union, Any

import sys
import warnings
import time
from datetime import datetime

from .parameters import get_alpha_beta
from .utils import (
    symbol_to_charge,
    direct_diagonalization,
    physconst,
    normalize_ao,
    ensure_torch,
)
from .excitation_space import screen_mo, csf_idx_as_ia
from .linear_response import (
    get_ab,
    transition_dipole,
    static_polarizability,
    transition_density,
)
from .integrals import charge_density_monopole
from .nto import get_nto

import numpy as np
import torch

# PySCF Mol object
Mol = Any


class sTDAVerboseMixin:
    def welcome(self):
        if self.verbose:
            print(datetime.now(), file=self.logstream)
            msg = "\n\n   " + "*" * 50 + "\n"
            msg += "   " + "*" + " " * 48 + "*" + "\n"
            msg += "   " + "*" + " " * 20 + "s T D A " + " " * 20 + "*" + "\n"
            msg += "   " + "*" + " " * 48 + "*" + "\n"
            msg += "   " + "*" * 50 + "\n"
            print(msg, file=self.logstream)

    def all_credits_to_grimme(self):
        if self.verbose:
            msg = "This is a PyTorch implementation of sTDA.\n"
            msg += "The original sTDA is the work of Stefan Grimme.\n\n"
            msg += "S. Grimme, J. Chem. Phys. 138 (2013) 244104\n"
            msg += "M. de Wergifosse, S. Grimme, J. Phys. Chem A\n"
            msg += "125 (2021) 18 3841-3851\n"
            print(msg, file=self.logstream)

    def stda_section(self):
        if self.verbose:
            msg = "=" * 65 + "\n"
            msg += " " * 28 + "s T D A \n"
            msg += "=" * 65
            print(msg, file=self.logstream)

    def mo_ao_input(self):
        if self.verbose:
            msg = "=" * 65 + "\n"
            msg += " " * 22 + "M O / A O   I N P U T \n"
            msg += "=" * 65 + "\n"
            print(msg, file=self.logstream)
            title = "{:3s} {:^15s} {:^15s} {:^15s} {:^10s}".format(
                "atom #", "x", "y", "z", "charge"
            )
            print(title, file=self.logstream)

            for symbol, (x, y, z) in zip(self.atom_pure_symbols, self.coords):
                charge = symbol_to_charge[symbol]
                row = "%3s %15.8f %15.8f %15.8f %10.1f" % (
                    symbol,
                    x.item(),
                    y.item(),
                    z.item(),
                    charge,
                )
                print(row, file=self.logstream)

            msg = "\n {:20s} = {:d}\n".format("# atoms", self.natm)
            msg += " {:20s} = {:d}\n".format("# mos", self.mo_coeff.shape[0])
            msg += " {:20s} = {:s}\n".format("# primitive aos", "not provided")
            msg += " {:20s} = {:d}\n".format("# contracted aos", len(self.ao_labels))
            print(msg, file=self.logstream)

    def thresholds_info(self):
        if self.verbose:
            msg = "{:30s} : {:.8f}\n".format("spectral range up to (eV)", self.e_max)
            msg += "{:30s} : {:.8f}\n".format(
                "occ MO cut-off (eV)", self.occthr * physconst.au_to_ev
            )
            msg += "{:30s} : {:.8f}\n".format(
                "virtMO cut-off (eV)", self.virthr * physconst.au_to_ev
            )
            msg += "{:30s} : {:.8f}\n".format("perturbation thr", self.tp)
            msg += "{:30s} : {:.8s}\n".format("triplet", "F")
            msg += "{:15s}: {:d}\n".format(
                "MOs in TDA", len(self.mask_occ) + len(self.mask_vir)
            )
            msg += "{:15s}: {:d}\n".format("oMOs in TDA", len(self.mask_occ))
            msg += "{:15s}: {:d}\n".format("vMOs in TDA", len(self.mask_vir))
            print(msg, file=self.logstream)

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
            print("\nSCF atom population (using active MOs):", file=self.logstream)
            print(" ", pop, file=self.logstream)
            print("\n# electrons in TDA: {:.3f}\n".format(nelec), file=self.logstream)

        # raise an exception if the number of electrons is not an integer
        if abs(nelec - len(self.mask_occ) * 2) > 1e-2:
            errmsg = "strange number of electrons in sTDA, "
            errmsg += "perhaps is worth checking the MO coefficients"
            raise RuntimeError(errmsg)

    def parameters_info(self):
        if self.verbose:
            msg = "\n{:20s}: {:.8f}\n".format("ax(DF)", self.ax)
            msg += "{:20s}: {:.8f}\n".format("beta (J)", self.beta)
            msg += "{:20s}: {:.8f}\n".format("alpha (K)", self.alpha)
            print(msg, file=self.logstream)

    def selection_by_energy(self):
        if self.verbose:
            msg = "\n{:d} CSF included by energy.\n".format(len(self.idx_pcsf))
            msg += "{:d} considered in PT2.\n".format(
                len(self.idx_ncsf) + len(self.idx_scsf)
            )
            print(msg, file=self.logstream)

    def selection_by_pt(self):
        if self.verbose:
            msg = "\n{:d} CSF included by PT.\n".format(len(self.idx_scsf))
            msg += "{:d} CSF in total.\n".format(
                len(self.idx_pcsf) + len(self.idx_scsf)
            )
            print(msg, file=self.logstream)

    def diag_info(self, diag_time):
        if self.verbose:
            msg = "estimated time (min)      {:.1f}\n".format((diag_time) / 60.0)
            msg += (
                "\t{:d} roots found, lowest/highest eigenvalue: {:.3f} {:.3f}\n".format(
                    self.e.shape[0],
                    torch.min(self.e) * physconst.au_to_ev,
                    torch.max(self.e) * physconst.au_to_ev,
                )
            )
            print(msg, file=self.logstream)

    def ordered_frontier_orbitals(self):
        if self.verbose:
            msg = "ordered frontier orbitals\n"
            msg += "{:8s} {:8s} {:8s}\n".format("", "eV", "# centers")
            print(msg, file=self.logstream)

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
                print(
                    "{:^8d} {:^8.3f} {:.1f}".format(i + 1, e * physconst.au_to_ev, c),
                    file=self.logstream,
                )
                if i == 10:
                    break

            print("", file=self.logstream)
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
                    ),
                    file=self.logstream,
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
            print(
                "\nexcitation energies, transition moments and TDA amplitudes",
                file=self.logstream,
            )
            print("state    eV      nm       fL        Rv(corr)", file=self.logstream)
            for i, (e, x) in enumerate(zip(self.e, self.x)):
                x = x / (0.5**0.5)
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
                    ),
                    file=self.logstream,
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

    def diagonalize(self, a):
        if self.verbose:
            print("diagonalizing...", file=self.logstream)
        return direct_diagonalization(a, nstates=self.nstates)

    def stda_done(self):
        if self.verbose:
            print("\nsTDA done.", file=self.logstream)


class sTDA(sTDAVerboseMixin):
    "simplified Tamm-Dancoff approximation"

    def __init__(
        self,
        mo_energy: Union[torch.Tensor, np.ndarray],
        mo_coeff: Union[torch.Tensor, np.ndarray],
        mo_occ: Union[torch.Tensor, np.ndarray],
        ovlp: Union[torch.Tensor, np.ndarray],
        natm: int,
        ao_labels: List[Tuple[int, str, str, str]],
        coords: Union[torch.Tensor, np.ndarray],
        atom_pure_symbols: List[str],
        ax: float,
        alpha: float = None,
        beta: float = None,
        e_max: float = 7.5,
        tp: float = 1e-4,
        mo_orth: bool = False,
        verbose: bool = True,
        logfile: str = None,
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
            logfile: standard output is written to logfile. If None, sys.stdout is
                     used
        """
        self.mo_energy = ensure_torch(mo_energy)
        self.mo_coeff = ensure_torch(mo_coeff)
        self.mo_occ = ensure_torch(mo_occ)
        self.check_restricted()
        self.ovlp = ensure_torch(ovlp) if ovlp is not None else ovlp
        self.natm = natm
        self.ao_labels = ao_labels
        self.coords = ensure_torch(coords)
        self.atom_pure_symbols = atom_pure_symbols
        self.ax = ax
        self._ax_alpha, self._ax_beta = get_alpha_beta(ax)
        self.alpha = alpha
        self.beta = beta
        self.e_max = e_max
        self.tp = tp
        self.mo_orth = mo_orth
        self.verbose = verbose
        self.logfile = logfile

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

    @property
    def mo_coeff_occ(self):
        """coefficients of occupied MOs in sTDA"""
        return self.mo_coeff[:, self.mask_occ]

    @property
    def mo_coeff_vir(self):
        """coefficients of virtual MOs in sTDA"""
        return self.mo_coeff[:, self.mask_vir + self.mask_occ[-1] + 1]

    def kernel(self, nstates=3):
        # the sTDA mixin uses prints its output to self.logstream
        if self.logfile is not None:
            self.logstream = open(self.logfile, "w")
        else:
            self.logstream = sys.stdout

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

        start = time.perf_counter()
        self.e, x = self.diagonalize(a)
        end = time.perf_counter()

        self.diag_info(end - start)

        # sTDA transition amplitudes
        self.nocc, self.nvir = len(self.mask_occ), len(self.mask_vir)
        # also normalize the transition amplitudes
        xy = [(xi * 0.5**0.5, 0.0) for xi in x.T]
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

        if self.logfile is not None:
            self.logstream.close()

        return self.e

    def get_nto(
        self, state: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the Natural Transition Orbitals associated with a transition

        Args:
            state: index of the excited state (first excited state is 1)
        Returns:
            weights: weight of each NTO (squared singular values
                   of x)
            nto_occ: coefficients of the hole NTOs, shape=(nao, nocc)
            nto_vir: coefficients of the particle NTOs, shape=(nao, nvir)
        """
        if self.mo_orth:
            warnings.warn(
                f"You are giving orthonormalized MOs (mo_orth={self.mo_orth})"
                ". Make sure to use the nonorthogonal MOs to plot these NTOs."
            )
        return get_nto(
            self.x, mo_occ=self.mo_coeff_occ, mo_vir=self.mo_coeff_vir, state=state
        )

    def transition_density(
        self, orbo: torch.Tensor = None, orbv: torch.Tensor = None
    ) -> torch.Tensor:
        """Computes the transition density

        Args:
            orbo: coefficients of the occupied MOs, shape (nao, nocc)
            orbv: coefficients of the virtual MOs, shape (nao, nvir)
        Returns:
            x_ao: transition density in the AO basis, shape (nao, nao)
        """
        if self.mo_orth and (orbo is None or orbv is None):
            errmsg = f"You are using orthonormalized MOs (mo_orth={self.mo_orth})."
            errmsg += " Cannot compute the transition density without the nonorthonormalized MOs."
            errmsg += " Please provide nonorthonormalized MOs as the 'orbo' and 'orbv' arguments"
            raise ValueError(errmsg)

        orbo = self.mo_coeff_occ if orbo is None else orbo
        orbv = self.mo_coeff_vir if orbv is None else orbv

        return transition_density(orbo=orbo, orbv=orbv, x=self.x)

    def transition_dipole(
        self,
        ints_ao: Union[torch.Tensor, np.ndarray, Mol],
        orbo: torch.Tensor = None,
        orbv: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computes the transition dipoles

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
        Returns:
            trn_dip: transition dipoles, shape (nexc, 3)
        """
        if self.mo_orth and (orbo is None or orbv is None):
            errmsg = f"You are using orthonormalized MOs (mo_orth={self.mo_orth})."
            errmsg += " Cannot compute the transition dipoles without the nonorthonormalized MOs."
            errmsg += " Please provide nonorthonormalized MOs as the 'orbo' and 'orbv' arguments"
            raise ValueError(errmsg)

        orbo = self.mo_coeff_occ if orbo is None else orbo
        orbv = self.mo_coeff_vir if orbv is None else orbv

        return transition_dipole(ints_ao=ints_ao, orbo=orbo, orbv=orbv, x=self.x)

    def static_polarizability(
        self,
        ints_ao: Union[torch.Tensor, np.ndarray, Mol],
        orbo: torch.Tensor = None,
        orbv: torch.Tensor = None,
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
            orbo: coefficients of the occupied MOs, shape (nao, nocc)
            orbv: coefficients of the virtual MOs, shape (nao, nvir)
        Returns:
            pol: polarizability tensor in atomic units, shape (3, 3)
        """
        if self.mo_orth and (orbo is None or orbv is None):
            errmsg = f"You are using orthonormalized MOs (mo_orth={self.mo_orth})."
            errmsg += (
                " Cannot compute the polarizability without the nonorthonormalized MOs."
            )
            errmsg += " Please provide nonorthonormalized MOs as the 'orbo' and 'orbv' arguments"
            raise ValueError(errmsg)

        orbo = self.mo_coeff_occ if orbo is None else orbo
        orbv = self.mo_coeff_vir if orbv is None else orbv

        return static_polarizability(
            ints_a=ints_ao,
            orbo=orbo,
            orbv=orbv,
            x=self.x,
            e=self.e,
        )
