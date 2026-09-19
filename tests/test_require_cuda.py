"""#377: a CPU fallback must say WHY, and a config can refuse it.

On 2026-09-16 a fresh install resolved torch 2.14+cu130, which the machine's 535 driver cannot
run; torch fell back to CPU and `violet_visitor` trained for 6 h 46 m under a banner that said
"No CUDA-capable GPU was detected" — false, the GPU was there. Two fixes: the banner diagnoses
the actual cause, and `require_cuda: true` turns the banner into a hard stop.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.utils import utils_logging  # noqa: E402
from views_hydranet.utils.config_initializer import HydraNetConfig  # noqa: E402


class TestTheConfigField:
    def test_defaults_to_false_so_every_existing_config_is_byte_identical(self, valid_config_dict):
        cfg = dict(valid_config_dict)
        cfg.pop("require_cuda", None)
        assert HydraNetConfig(**cfg).require_cuda is False

    def test_the_field_exists_and_takes_true(self, valid_config_dict):
        cfg = dict(valid_config_dict)
        cfg["require_cuda"] = True
        assert HydraNetConfig(**cfg).require_cuda is True


class TestTheHardStop:
    def test_require_cuda_on_a_cpu_device_raises_with_the_diagnosis(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="require_cuda=True") as exc:
                utils_logging.log_device_report(torch.device("cpu"), "training", require_cuda=True)
        assert "Refusing to run on CPU" in str(exc.value)
        assert "torch " in str(exc.value), "the refusal must name the torch build it found"
        assert any("require_cuda=True" in r.getMessage() for r in caplog.records), (
            "ADR-008: logged before raised"
        )

    def test_without_require_cuda_a_cpu_device_warns_and_proceeds(self, caplog, capsys):
        """Anti-vacuity: the default path is unchanged — a warning and a banner, no raise."""
        with caplog.at_level(logging.WARNING):
            utils_logging.log_device_report(torch.device("cpu"), "training")
        assert any("running on CPU" in r.getMessage() for r in caplog.records)
        assert "RUNNING ON CPU" in capsys.readouterr().out

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to be present")
    def test_require_cuda_on_a_cuda_device_is_silent(self, capsys):
        utils_logging.log_device_report(torch.device("cuda"), "training", require_cuda=True)
        assert "DEVICE REPORT" in capsys.readouterr().out


class TestTheDiagnosisNamesTheActualCause:
    """The old banner said "No CUDA-capable GPU was detected" for every CPU fallback. The three
    causes a reader can act on are distinguished by faking what the diagnosis reads."""

    def test_cpu_only_torch_build(self, monkeypatch):
        monkeypatch.setattr(torch.version, "cuda", None)
        assert "CPU-only build" in utils_logging.cpu_fallback_diagnosis()

    def test_no_driver_visible(self, monkeypatch):
        monkeypatch.setattr(torch.version, "cuda", "13.0")
        monkeypatch.setattr("shutil.which", lambda _name: None)
        msg = utils_logging.cpu_fallback_diagnosis()
        assert "No NVIDIA driver" in msg and "13.0" in msg

    def test_driver_present_but_too_old_for_this_torch_build(self, monkeypatch):
        import subprocess

        monkeypatch.setattr(torch.version, "cuda", "13.0")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/nvidia-smi")

        class _Out:
            returncode = 0
            stdout = "535.309.01, NVIDIA GeForce RTX 4070 Laptop GPU\n"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Out())
        msg = utils_logging.cpu_fallback_diagnosis()
        assert "A GPU is present" in msg and "535.309.01" in msg and "CUDA 13.0" in msg
        assert "cu124" in msg, "the message must say what to install instead"
