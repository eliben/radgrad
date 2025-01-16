# Sample of computing derivatives through Python control flow. taylor_sin
# uses a `for` loop to compute the Taylor series of sin(x). The derivative
# of this function is computed using the `grad` function, and the result is
# printed for a few values of x.
from radgrad import grad
import math


def taylor_sin(x):
    # The Taylor series for sin(x) is
    #
    # x - x^3/3! + x^5/5! - x^7/7! + ...
    #
    # This code builds up each term based on the previous term, and computes
    # the first 20 terms of the series - which should provide excellent
    # precision around x=0 (but not *too* far from it, because this is
    # technically the MacLaurin series).
    ans = term = x
    for i in range(0, 20):
        term = -term * x * x / ((2 * i + 3) * (2 * i + 2))
        ans = ans + term
    return ans


dsin_dx = grad(taylor_sin)

for x in ["0.0", "math.pi / 4", "math.pi / 2", "math.pi"]:
    xname, xval = x, eval(x)
    print(f"sin({xname}) = {taylor_sin(xval):.3}")
    print(f"dsin_dx({xname}) = {dsin_dx(xval)[0]:.3}")

try:
    import numpy  # This is the actual Numpy, not radgrad's wrapper
    import matplotlib.pyplot as plt

    x = numpy.linspace(-math.pi, math.pi, 1000)

    y = taylor_sin(x)
    dy = dsin_dx(x)[0]

    plt.axhline(y=0, color="lightgray", alpha=0.5)
    plt.axvline(x=0, color="lightgray", alpha=0.5)
    plt.plot(x, y, label="sin")
    plt.plot(x, dy, label="dsin_dx")
    plt.legend()

    plt.show()
except ImportError:
    print("Please install numpy and matplotlib to plot the function.")
