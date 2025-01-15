from .radgrad import grad

# 'grad' should be the only public entry point for the 'radgrad' package,
# in addition to importing the 'numpy_wrapper' module as 'np'.
__all__ = ["grad"]


# Import this module just for its side effects, in case the user doesn't
# import it directly. The redundant "as numpy_wrapper" alias is to avoid
# ruff linter complaints.
from . import numpy_wrapper as numpy_wrapper
