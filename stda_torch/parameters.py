from typing import Tuple

# This is a database of chemical hardnesses computed in
# D.C. Ghosh, N. Islam, Int. J. Quantum Chem. 110 (2010) 1206–1213.
# Values in Hartree and multiplied by two.
chemical_hardness = {
    "  H": 0.472592880,
    " He": 0.922033910,
    " Li": 0.174528880,
    " Be": 0.257007330,
    "  B": 0.339490860,
    "  C": 0.421954120,
    "  N": 0.504381930,
    "  O": 0.586918630,
    "  F": 0.669313510,
    " Ne": 0.751916070,
    " Na": 0.179641050,
    " Mg": 0.221572760,
    " Al": 0.263485780,
    " Si": 0.305396450,
    "  P": 0.347340140,
    "  S": 0.389247250,
    " Cl": 0.431156700,
    " Ar": 0.473082690,
    "  K": 0.171054690,
    " Ca": 0.202762440,
    " Sc": 0.210073220,
    " Ti": 0.217396470,
    "  V": 0.224710390,
    " Cr": 0.232015010,
    " Mn": 0.239339690,
    " Fe": 0.246656380,
    " Co": 0.253982550,
    " Ni": 0.261288630,
    " Cu": 0.268594760,
    " Zn": 0.275925650,
    " Ga": 0.307629990,
    " Ge": 0.339315800,
    " As": 0.372359850,
    " Se": 0.402735490,
    " Br": 0.434457760,
    " Kr": 0.466117080,
    " Rb": 0.155850790,
    " Sr": 0.186493240,
    "  Y": 0.193562100,
    " Zr": 0.200633110,
    " Nb": 0.207705220,
    " Mo": 0.214772540,
    " Tc": 0.221846140,
    " Ru": 0.228918720,
    " Rh": 0.235986210,
    " Pd": 0.243056120,
    " Ag": 0.250130180,
    " Cd": 0.257199370,
    " In": 0.287847800,
    " Sn": 0.318486730,
    " Sb": 0.349124310,
    " Te": 0.379765930,
    "  I": 0.410408080,
    " Xe": 0.441057770,
    " Cs": 0.050193320,
    " Ba": 0.067625700,
    " La": 0.085044450,
    " Ce": 0.102477360,
    " Pr": 0.119911050,
    " Nd": 0.137327720,
    " Pm": 0.154762970,
    " Sm": 0.172182650,
    " Eu": 0.189612880,
    " Gd": 0.207047600,
    " Tb": 0.224467520,
    " Dy": 0.241896450,
    " Ho": 0.259325030,
    " Er": 0.276760940,
    " Tm": 0.294182310,
    " Yb": 0.311595870,
    " Lu": 0.329022740,
    " Hf": 0.345922980,
    " Ta": 0.363880480,
    "  W": 0.381305860,
    " Re": 0.398774760,
    " Os": 0.416142980,
    " Ir": 0.433645100,
    " Pt": 0.451040140,
    " Au": 0.468489860,
    " Hg": 0.485845500,
    " Tl": 0.125267300,
    " Pb": 0.142686770,
    " Bi": 0.160116150,
    " Po": 0.177558890,
    " At": 0.194975570,
    " Rn": 0.212407780,
    " Fr": 0.072635250,
    " Ra": 0.094221580,
    " Ac": 0.099202950,
    " Th": 0.104186210,
    " Pa": 0.142356330,
    "  U": 0.163942940,
    " Np": 0.185519410,
    " Pu": 0.223701390,
}
# Trim whitespaces in key values
chemical_hardness = {k.strip(): v for k, v in chemical_hardness.items()}


# alpha and beta parameters
def get_alpha_beta(ax: int) -> Tuple[int, int]:
    beta1 = 0.20
    beta2 = 1.83
    alpha1 = 1.42
    alpha2 = 0.48
    return alpha1 + alpha2 * ax, beta1 + beta2 * ax
