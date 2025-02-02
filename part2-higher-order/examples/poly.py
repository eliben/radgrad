# Derivatives of a polynomial


from radgrad import grad1


def poly(x):
    return x * x * x + 2 * x * x - 3 * x + 1


dpdx = grad1(poly)
d2pdx = grad1(grad1(poly))
d3pdx = grad1(grad1(grad1(poly)))
d4pdx = grad1(grad1(grad1(grad1(poly))))
              

print("poly(x) = x^3 + 2x^2 - 3x + 1")
print(f'poly(1) = {poly(1)}')
print(f'dpdx(1) = {dpdx(1)}')
print(f'd2pdx(1) = {d2pdx(1)}')
print(f'd3pdx(1) = {d3pdx(1)}')
print(f'd4pdx(1) = {d4pdx(1)}')

