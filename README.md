# radgrad

<p align="center">
  <img alt="Logo" src="doc/radgrad-logo.png" />
</p>

----

**radgrad** (**rad** stands for reverse-mode automatic differentiation) is
an educational implementation of automatic differentiation implemented on top
of a Numpy wrapper. It is a (very) simplified clone of
[Autograd](https://github.com/hips/autograd).

Here's a basic example:

```python
import radgrad.numpy_wrapper as np
from radgrad import grad

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

print(tanh(1.0))
dtanh_dx = grad(tanh)
print(dtanh_dx(1.0))
```

`grad` is a higher-order function. It takes a function that contains
mathematical calculation that uses Numpy, and transforms it into a function
that computes the derivative of this computation. In the code above, the call
`tanh(1.0)` evaluates the value of the `tanh` function at `1.0`; the call
`dtanh_dx(1.0)` evaluates the derivative of the `tanh` function with respect
to its inputs at `1.0`.

To understand how `radgrad` works, start by reading
[this blog post](https://eli.thegreenplace.net/2025/reverse-mode-automatic-differentiation/).
Then, just read `radgrad`'s code and play with the examples. The code is
heavily commented to explain what's going on.

To make the learning journey easier, this project is split into two parts:

* [Part 1](https://github.com/eliben/radgrad/tree/main/part1-basic): xx
* [Part 2](https://github.com/eliben/radgrad/tree/main/part2-higher-order): yy





