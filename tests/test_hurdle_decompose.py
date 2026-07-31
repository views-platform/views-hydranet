"""Unit tests for scripts/hurdle_decompose.py attribution logic.

The tool splits E[y] = pi * body into a GATE factor (pi) and a conditional-BODY factor, then
attributes (a) the zero-cell leak and (b) ns/os positive under-firing to one or the other. These
tests pin the attribution on synthetic cells where the dominant factor is known by construction.
"""

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_MOD = os.path.join(_SCRIPTS, "hurdle_decompose.py")
_spec = importlib.util.spec_from_file_location("hurdle_decompose", _MOD)
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)


def test_zero_leak_gate_dominated():
    """π high (fires on zeros) while body sits at the reference → leak is GATE-driven."""
    pi = np.full(100, 0.5)
    body = np.full(100, 5.0)  # == m_pos (body not inflated)
    out = hd.attribute_zero_leak(pi, body, rho=0.01, m_pos=5.0)
    assert out["dominant"] == "gate"
    assert out["frac_gate"] > out["frac_body"]


def test_zero_leak_body_dominated():
    """π calibrated (==prevalence) but body inflated 10× → leak is BODY-driven."""
    pi = np.full(100, 0.01)
    body = np.full(100, 50.0)  # 10× m_pos
    out = hd.attribute_zero_leak(pi, body, rho=0.01, m_pos=5.0)
    assert out["dominant"] == "body"


def test_pos_underfire_gate_driven():
    """Positive cells, pi low (gate fails to fire), body~truth -> under-firing is GATE-driven."""
    pi = np.full(50, 0.1)
    body = np.full(50, 10.0)
    truth = np.full(50, 10.0)
    out = hd.attribute_pos_underfire(pi, body, truth)
    assert out["dominant"] == "gate"


def test_pos_underfire_body_driven():
    """π high (gate fires) but body << truth → under-firing is BODY-driven."""
    pi = np.full(50, 0.95)
    body = np.full(50, 2.0)
    truth = np.full(50, 10.0)
    out = hd.attribute_pos_underfire(pi, body, truth)
    assert out["dominant"] == "body"


def test_body_roundtrip():
    """body := E[y]/pi reconstructs E[y] exactly (pi*body == E[y]); the guard handles pi~0."""
    rng = np.random.default_rng(0)
    pi = rng.uniform(0.01, 1.0, size=200)
    ey = rng.uniform(0.0, 100.0, size=200)
    body = hd.conditional_body(ey, pi)
    assert np.allclose(pi * body, ey, atol=1e-9)
