from __future__ import annotations
from typing import List, Tuple

from datetime import datetime

from .parameters import get_alpha_beta
from .utils import symbol_to_charge
from .excitation_space import screen_mo, AU_TO_EV
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

    def kernel(self):
        if self.verbose:
            self.welcome()
            self.all_credits_to_grimme()
            self.mo_ao_input()

            print("=" * 65)
            print(" " * 28 + "s T D A ")
            print("=" * 65)
            print("{:30s} : {:.8f}".format("spectral range up to (eV)", self.e_max))

        mask_occ, mask_vir, occthr, virthr = screen_mo(
            mo_energy=self.mo_energy, mo_occ=self.mo_occ, ax=self.ax, e_max=self.e_max
        )

        if self.verbose:
            print("{:30s} : {:.8f}".format("occ MO cut-off (eV)", occthr * AU_TO_EV))
            print("{:30s} : {:.8f}".format("virtMO cut-off (eV)", virthr * AU_TO_EV))
            print("{:30s} : {:.8f}".format("perturbation thr", self.tp))
            print("{:30s} : {:.8s}".format("triplet", "F"))
            print("{:15s}: {:d}".format("MOs in TDA", len(mask_occ) + len(mask_vir)))
            print("{:15s}: {:d}".format("oMOs in TDA", len(mask_occ)))
            print("{:15s}: {:d}".format("vMOs in TDA", len(mask_vir)))

            print("\nSCF atom population (using active MOs):")
            pop = (
                charge_density_monopole(
                    ovlp=self.ovlp,
                    natm=self.natm,
                    ao_labels=self.ao_labels,
                    mo_coeff_a=self.mo_coeff[:, mask_occ],
                    mo_coeff_b=self.mo_coeff[:, mask_occ],
                )
                * 2  # noqa: W503
            )
            pop = torch.einsum("Aii->A", pop)
            print(" ", pop)

            print("\n# electrons in TDA: {:.3f}".format(torch.sum(pop)))

            print("\n")
            print("{:20s}: {:.8f}".format("ax(DF)", self.ax))
            print("{:20s}: {:.8f}".format("beta (J)", self.beta))
            print("{:20s}: {:.8f}".format("alpha (K)", self.alpha))

        a, b, *metadata = get_ab(
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
            mask_occ=mask_occ,
            mask_vir=mask_vir,
            excitation_space="stda",
            e_max=self.e_max,
            tp=self.tp,
            verbose=False,
        )
