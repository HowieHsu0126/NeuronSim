"""Check source transfer values, SymPy types and context restoration."""
from types import SimpleNamespace

import numpy as np
import pytest
import sympy

from . import v1_lgn_transfer as transfer


@pytest.mark.parametrize("value", [-1., -0., 0., 1., np.float64(-.1), np.float64(.1),
                                   sympy.Float(-.1), sympy.Float(.1), sympy.S.Zero,
                                   sympy.Symbol("x"), float("nan"), float("inf"),
                                   float("-inf")])
def test_numeric_heaviside_keeps_source_value_and_type(value):
    actual = transfer.numeric_heaviside(value)
    expected = sympy.Heaviside(value, .5)
    assert actual == expected
    assert type(actual) is type(expected)


@pytest.mark.parametrize("offset", [0., 2., 2.75, 4., 5.5])
def test_generated_scalar_expression_is_exact(offset):
    x = sympy.Symbol("s")
    source = sympy.lambdify(x, sympy.sympify(f"Heaviside(s+{offset})*(s+{offset})"),
                           modules=["sympy"])
    candidate = sympy.lambdify(x, sympy.sympify(f"Heaviside(s+{offset})*(s+{offset})"),
                              modules=["sympy"])
    candidate.__globals__["Heaviside"] = transfer.numeric_heaviside
    rng = np.random.default_rng(9231)
    values = np.concatenate((rng.normal(size=512), [-offset, np.nextafter(-offset, -np.inf),
                            np.nextafter(-offset, np.inf), -0., 0., 1e-300, -1e-300]))
    for value in values:
        expected, actual = source(value), candidate(value)
        assert expected == actual
        assert type(expected) is type(actual)
    assert np.array_equal(np.asarray([source(x) for x in values], dtype=float),
                          np.asarray([candidate(x) for x in values], dtype=float))


def test_dual_expression_is_exact():
    x, y = sympy.symbols("x y")
    expression = sympy.sympify("Heaviside(x)*(x)+Heaviside(y)*(y)")
    source = sympy.lambdify((x, y), expression, modules=["sympy"])
    candidate = sympy.lambdify((x, y), expression, modules=["sympy"])
    candidate.__globals__["Heaviside"] = transfer.numeric_heaviside
    values = [sympy.Float(-.123), sympy.Float(.123), sympy.S.Zero, np.float64(-.2), np.float64(.2)]
    for a in values:
        for b in values:
            expected, actual = source(a, b), candidate(a, b)
            assert expected == actual
            assert type(expected) is type(actual)


def test_context_restores_source_factory_after_failure(monkeypatch):
    module = SimpleNamespace(lambdify=sympy.lambdify)
    monkeypatch.setattr(transfer.importlib, "import_module", lambda name: module)
    with pytest.raises(RuntimeError):
        with transfer.numeric_transfer_support():
            closure = module.lambdify(sympy.Symbol("s"), sympy.sympify("Heaviside(s)*s"),
                                     modules=["sympy"])
            assert closure.__globals__["Heaviside"] is transfer.numeric_heaviside
            raise RuntimeError("execution failure")
    assert module.lambdify is sympy.lambdify
