"""The BatchNorm buffers a recalibration pass produces must describe the activations inference
actually feeds BatchNorm.

The invariant, stated once: **the recal pass runs in the same dropout regime as inference.**
Inference is `model.eval()` + `set_locked_dropout(True)` — MC-dropout ON with a locked mask
(ADR-059, the production posterior; CIC `HydraBNUNet06LSTM4.md`). The C-184 recal fix was
validated in that regime.

Why this test exists: Epic #353 / S3 switched dropout OFF during recal, on the premise that
inference runs it off. ADR-059 said otherwise, the model's CIC said otherwise, and the
`set_locked_dropout` docstring said otherwise — and none of them was read, because the change was
made on the training side and the invariant was documented only on the consumer side. Prose in
three places did not stop it; `/code-review max` on #372 did, by MEASURING recal `running_var`
against the inference forward (14–21% low at 14 of 15 layers). This test is that measurement,
made permanent. It fails on any future regime drift, not only dropout.

Direction matters. Recal statistics LARGER than inference's spread make eval-mode BN
under-amplify — conservative. SMALLER makes it over-amplify, which is the C-184 seed-bimodal
collapse. So the floor is one-sided.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

import views_hydranet.train.training_engine as te  # noqa: E402

from .conftest import loop_config, loop_handler  # noqa: E402

N_WINDOWS = 40
FLOOR = 0.95  # correct regime measures ~1.12 (min 1.01); dropout-off-during-recal ~0.85 (min 0.79)


def _recalibrated_model_and_windows():
    device = torch.device("cpu")
    cfg = loop_config(
        random_flips=False, bn_recalibrate=True, bn_recal_windows=N_WINDOWS, total_lessons=1
    )
    torch.manual_seed(cfg["torch_seed"])
    model, criterion, optimizer, scheduler = te.make(cfg, device)
    handler = loop_handler(cfg)
    te.training_loop(cfg, model, criterion, optimizer, scheduler, handler, device)
    return cfg, model, criterion, optimizer, scheduler, handler, device


def _inference_regime_bn_input_variance(cfg, model, criterion, optimizer, scheduler, handler, dev):
    """Per-BN-layer, per-channel variance of the inputs BatchNorm receives when the model is run
    exactly as inference runs it, over the same windows the recal pass consumed."""
    model.eval()
    model.set_locked_dropout(True)
    acc: dict[str, list] = {}

    def _hook(name):
        def _h(_m, inp):
            x = inp[0].detach()
            dims = [d for d in range(x.dim()) if d != 1]
            s = acc.setdefault(name, [0.0, 0.0, 0])
            s[0] = s[0] + x.sum(dim=dims)
            s[1] = s[1] + (x * x).sum(dim=dims)
            s[2] += x.numel() // x.shape[1]

        return _h

    handles = [
        m.register_forward_pre_hook(_hook(name))
        for name, m in model.named_modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    sampler = te.VolumeSampler(handler, cfg)
    planner = te.CurriculumLearner(cfg, handler)
    ctx = te.TrainingContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_reg=criterion[0],
        criterion_class=criterion[1],
        multitaskloss_instance=criterion[2],
        config=cfg,
        device=dev,
        viz=te.VisualDiagnostics(cfg),
        forensics=te.TrainingForensics(cfg),
    )
    # The forward goes through the same sequence code the recal pass used, in the OTHER regime.
    # `train()` calls model.train() on entry and may (as the reverted S3 draft did) touch the
    # dropout modules, so the inference regime is RE-IMPOSED right before every model forward,
    # by the model's own API and independent of whatever `train()` did. Without this, a `train()`
    # that switches dropout off would switch it off in this leg too, the two regimes would agree
    # with each other, and the test would pass on exactly the defect it exists to catch.
    model.train = lambda mode=True: model  # type: ignore[method-assign]

    def _impose_inference_regime(_m, _inp):
        for sub in model.modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                sub.eval()
            elif isinstance(sub, nn.Dropout):
                sub.train()
        for sub in model.modules():
            if type(sub).__name__ == "LockedDropout":
                sub.locked = True
                sub.train()  # NOT set_locked_dropout(True): that resets the mask every call

    regime = model.register_forward_pre_hook(_impose_inference_regime)
    handles.append(regime)
    try:
        with torch.no_grad():
            for w in range(N_WINDOWS):
                model.reset_locked_dropout()  # per posterior sample, as predict() does
                target, threshold = planner.get_lesson(w)
                batch, _ = sampler.get_batch(target, threshold, batch_size=1)
                te.train(ctx, batch[0], None, stage_label="", training_augmentation=False)
    finally:
        del model.train
        for h in handles:
            h.remove()
    out = {}
    for name, (s0, s1, n) in acc.items():
        mean = s0 / n
        out[name] = (s1 / n - mean * mean).clamp(min=1e-12)
    return out


def test_recal_running_var_is_not_below_what_inference_feeds_batchnorm():
    cfg, model, criterion, optimizer, scheduler, handler, dev = _recalibrated_model_and_windows()
    recal_var = {
        name: m.running_var.detach().clone()
        for name, m in model.named_modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    }
    assert len(recal_var) == 15, f"expected 15 BatchNorm layers, found {len(recal_var)}"

    inf_var = _inference_regime_bn_input_variance(
        cfg, model, criterion, optimizer, scheduler, handler, dev
    )
    ratios = {name: float((recal_var[name] / inf_var[name]).mean()) for name in recal_var}
    mean_ratio = sum(ratios.values()) / len(ratios)
    worst = min(ratios.items(), key=lambda kv: kv[1])
    assert mean_ratio >= FLOOR and worst[1] >= FLOOR - 0.05, (
        f"BatchNorm recalibration under-estimates the activations inference feeds BN: mean "
        f"running_var / inference_var = {mean_ratio:.3f} (worst {worst[0]} = {worst[1]:.3f}, "
        f"floor {FLOOR}). Eval-mode BN would over-amplify — the C-184 bad-basin direction. The "
        "recal pass is not running in the inference regime (ADR-059: MC-dropout ON, locked mask)."
    )
