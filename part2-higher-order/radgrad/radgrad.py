from dataclasses import dataclass
from collections.abc import Callable
from contextlib import contextmanager
import typing
import numpy as _np


@dataclass
class Node:
    """Node in the computation graph.

    Nodes enable AD by holding the VJP function for each computational step,
    along with references to this node's predecessors in the graph.
    """

    vjp_func: Callable
    predecessors: list["Node"]


def make_root_node():
    """Empty node with no predecessors."""
    return Node(None, [])


@dataclass
class Box:
    """Box for AD tracing.

    Boxes wrap values and associate them with a Node in the computation graph.
    level specifies the tracing level of the box - higher levels are used for
    higher-order gradients.
    """

    value: typing.Any
    node: Node
    level: int = 0


def wrap_primitive(f):
    """Wrap a primitive function to use Boxes for AD tracing.

    "Primitive" in this context means a function that is unaware of radgrad's
    machinery and just operates on numbers or NumPy arrays.
    """

    def wrapped(*args):
        # Find the highest box level among the arguments, and make sure *all*
        # arguments are boxed at that level. This ensures that lower level
        # boxes are treated like constants when computing higher-order
        # gradients.
        level = max((x.level for x in args if isinstance(x, Box)), default=0)

        if level == 0:
            # If no arguments are boxes, there's no tracing to be done. Just
            # call the primitive and return its result.
            return f(*args)

        boxes = [box_at_level(level, x) for x in args]

        # Unbox the values, compute forward output and obtain the
        # VJP function for this computation.
        output, vjp_func = vjp_rules[f](*[b.value for b in boxes])

        # Box the output to the appropriate level and return it, with an
        # associated Node.
        return Box(
            level=level,
            value=output,
            node=Node(vjp_func=vjp_func, predecessors=[b.node for b in boxes]),
        )

    return wrapped


box_level = 0


# Context manager to increment the box level for recursive calls of grad.
@contextmanager
def new_box_level():
    global box_level
    box_level += 1
    try:
        yield box_level
    finally:
        box_level -= 1


def box_at_level(level, v):
    """Box a value at a specific level, unless it's already boxed at that level."""
    if isinstance(v, Box) and v.level == level:
        return v
    return Box(value=v, node=make_root_node(), level=level)


# vjp_rules holds the calculation and VJP rules for each primitive.
# (VJP = Vector-Jacobian Product)
# Structure:
#   vjp_rules[primitive] = maker_func(*args)
#     primitive: The primitive function we've wrapped.
#     maker_func(*args):
#       takes the runtime values of arguments passed into the primitive and
#       returns a tuple (output, vjp_func). The output is the result of the
#       forward computation of the primitive with *args, and vjp_func
#       calculates the vector-jacobian product. It takes the output gradient
#       and returns input gradients of the primitive for each argument,
#       as a list.
vjp_rules = {}


def add_vjp_rule(np_primitive, vjp_maker_func):
    """Helper to register a VJP rule in vjp_rules."""
    vjp_rules[np_primitive] = vjp_maker_func


def backprop(arg_nodes, out_node, out_g):
    """Backpropagation through the computation graph.

    arg_nodes:
        List of nodes corresponding to the arguments of the function. A
        gradient is computed for each of these nodes.
    out_node:
        Starting node for backpropagation. This should be the output node of
        the computational graph.
    out_g:
        Gradient of the output node - the derivative of some abstract metric
        w.r.t. the graph's output. This is typically set to 1.

    Returns a list of gradients, one for each argument node.
    """
    grads = {id(out_node): out_g}
    for node in toposort(out_node):
        g = grads.pop(id(node))

        inputs_g = node.vjp_func(g)
        assert len(inputs_g) == len(node.predecessors)
        for inp_node, g in zip(node.predecessors, inputs_g):
            grads[id(inp_node)] = grads.get(id(inp_node), 0.0) + g
    return [grads.get(id(node), 0.0) for node in arg_nodes]


def toposort(out_node):
    """Topological sort of the computation graph starting at out_node.

    Yields nodes in topologically sorted order.
    """
    visited = set()

    def postorder(node):
        visited.add(id(node))
        for pred in node.predecessors:
            if id(pred) not in visited:
                yield from postorder(pred)
        yield node

    return reversed([node for node in postorder(out_node) if node.predecessors])


def grad(f):
    """Takes a function f and returns a new function that computes its gradient.

    The gradient is computed at a specific point - the values of f's arguments.
    Let's say f takes n arguments. Then grad(f)(x1, x2, ..., xn) returns
    [df/dx1, df/dx2, ..., df/dxn], where df/dxi is the gradient of f with
    respect to xi, computed at the point (x1, x2, ..., xn).

    This only supports functions f that return a scalar.
    """

    def wrapped(*args):
        with new_box_level() as level:
            # Start by boxing all arguments so we can properly trace out the
            # computational graph.
            boxed_args = [
                Box(value=x, level=level, node=make_root_node()) for x in args
            ]

            # Run the function with boxed arguments; this will construct the
            # computation graph out of Box values, and returns the Box for f's
            # output.
            out = f(*boxed_args)
            out = box_at_level(level, out)

            arg_nodes = [b.node for b in boxed_args]

            # Run backpropagation to compute gradients, starting at the output
            # node.
            return backprop(arg_nodes, out.node, _np.float64(1.0))

    return wrapped


def grad1(f):
    """Just like grad, but returns a single gradient instead of a list.

    This is useful when f has only one argument, since it's easier to compose
    for higher-order gradients.

    Warning: if f has more than one argument, grad1 will return the gradient
    w.r.t. the first argument only, the other gradients will be discarded.
    """

    def wrap_grad(*args):
        return grad(f)(*args)[0]

    return wrap_grad
