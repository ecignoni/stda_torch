import warnings
import torch
from . import tools  # noqa

warnings.simplefilter("always", UserWarning)
torch.set_default_dtype(torch.float64)
