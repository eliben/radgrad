# Basic function with higher order gradients.

import radgrad.numpy_wrapper as np
from radgrad import grad1


def f(x):
    return x + np.sin(x)


print(f(0.5))

df_dx = grad1(f)
print(df_dx(0.5))


d2f_dx2 = grad1(df_dx)
print(d2f_dx2(0.5))

d3f_dx3 = grad1(d2f_dx2)
print(d3f_dx3(0.5))
