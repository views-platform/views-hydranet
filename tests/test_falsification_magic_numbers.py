"""
Falsification audit: magic numbers and shadow defaults in source code.

P1: flip probability xfail (C-85 open — hardcoded 0.5, no config field).
P7: shadow defaults now fixed — tests must PASS (not xfail).
"""

import ast
import inspect

import pytest


class TestSoftFalsificationFlipProbability:
    """P1: Data augmentation flip probability is hardcoded at 0.5."""

    @pytest.mark.xfail(reason="Flip probability 0.5 is hardcoded — no config field exists")
    def test_flip_probability_is_config_driven(self):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        assert (
            hasattr(HydraNetConfig.model_fields, "flip_probability")
            or "flip_probability" in HydraNetConfig.model_fields
        ), "flip_probability should be a config field, not hardcoded at 0.5"


class TestRedShadowDefaults:
    """P7: config.get() calls must not have fallback defaults that shadow the schema."""

    def test_random_flips_no_shadow_default(self):
        from views_hydranet.train import training_engine

        source = inspect.getsource(training_engine)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "random_flips"
            ):
                assert len(node.args) < 2, (
                    "config.get('random_flips', True) has a shadow default — "
                    "schema already provides default=True"
                )

    def test_clip_grad_norm_no_shadow_default(self):
        from views_hydranet.train import training_engine

        source = inspect.getsource(training_engine)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) >= 1
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "clip_grad_norm"
            ):
                assert len(node.args) < 2, (
                    "config.get('clip_grad_norm', False) has a shadow default — "
                    "field is required (Field(...)), so config.get() without "
                    "fallback is correct"
                )
