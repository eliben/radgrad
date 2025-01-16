# Hyperbolic tangent function and its derivative.

import radgrad.numpy_wrapper as np
from radgrad import grad1


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


try:
    import numpy  # This is the actual Numpy, not radgrad's wrapper
    import matplotlib.pyplot as plt

    x = numpy.linspace(-7, 7, 1000)

    y = tanh(x)
    dy = grad1(tanh)(x)
    d2y = grad1(grad1(tanh))(x)
    d3y = grad1(grad1(grad1(tanh)))(x)
    d4y = grad1(grad1(grad1(grad1(tanh))))(x)

    plt.grid(True)
    plt.plot(x, y, label="tanh")
    plt.plot(x, dy, label="1st derivative")
    plt.plot(x, d2y, label="2nd derivative")
    plt.plot(x, d3y, label="3rd derivative")
    plt.plot(x, d4y, label="4th derivative")
    plt.legend()

    plt.show()
except ImportError:
    print("Please install numpy and matplotlib to plot the function.")
