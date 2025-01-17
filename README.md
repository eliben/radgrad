# radgrad

<p align="center">
  <img alt="Logo" src="doc/radgrad-logo.png" />
</p>

----

**radgrad** (**rad** stands for reverse-mode automatic differentiation) is
an educational implementation of automatic differentiation on top
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

* [Part 1](https://github.com/eliben/radgrad/tree/main/part1-basic): implements
  the simplest AD mechanism possible, with support for only first order
  derivatives.
* [Part 2](https://github.com/eliben/radgrad/tree/main/part2-higher-order):
  builds on top of part 1 to implement higher-order derivatives.

The code of parts 1 and 2 is almost identical; I recommend starting with Part 1,
and once you understand how it works, run a recursive diff (e.g.
`meld part1-basic/ part2-higher-order/`) to get a feeling for the deltas. Read
more on higher-order derivatives in `radgrad` below.

## Tracing AD approach

`radgrad` implements _tracing_ AD; when `grad(f)` is invoked, there's no static
analysis of `f`'s code going on. Instead, `grad` wraps all the arguments passed
into `f` with special `Box` types that keep track of the operations performed
on them (using a mix of operator overloading and specially wrapped Numpy
primitives). This is used to construct an implicit computational graph
(_implicit_ in the sense that the user isn't even aware of it) on which the
[reverse mode AD process can be run](https://eli.thegreenplace.net/2025/reverse-mode-automatic-differentiation/).

This lets us calculate derivatives of code that contains Python control flow;
here's an example from `examples/taylor-sin.py`:

```python
from radgrad import grad
import math

def taylor_sin(x):
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
```

`taylor_sin` computes a Taylor series approximation to `sin`. Note how it
uses a Python loop; `grad(taylor_sin)` still works, even though it's not clear
what the derivative of a Python loop even means! In reality, the tracing
approach ensures that the loop is _unrolled_ in the computational graph - it
only sees the actual path taken by a specific invocation.

## Running the code

I find it easiest to run this code using `uv`. For example:

```shell
$ cd part1-basic
$ PYTHONPATH=. uv run examples/tanh.py
```

Some examples plot graphs using `matplotlib`. If you want to see the plots,
ask `uv` to include `matplotlib` in the dependencies, as follows:

```shell
$ cd part2-higher-order
$ PYTHONPATH=. uv run --with matplotlib examples/tanh.py
```

This produces a plot of several levels of derivatives of the `tanh` function:

<p align="center">
  <img alt="tanh derivatives" src="doc/tanh-derivatives.png" />
</p>



