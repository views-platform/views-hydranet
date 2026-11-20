"""The one setting that names the regression body, and its five values (ADR-081, issue #633).

A config names its body with ``body``. Each value is spelled ``<loss>_<activation>_<emit>`` in
full and matched whole, never split on the underscores (ADR-081 §2). Until 2026-11-20 three
settings did this together — ``loss_reg``, ``output_distribution`` and ``reg_activation`` — and
27 of 37 configs named a loss that did not train. A config that still writes any of the three is
refused, and the error names the value to write instead (owner decision, 2026-11-20).

``distribution`` and ``loss_name`` below are what a value fixes, read by the code that builds the
head and the loss. They are not settings: nothing can write them except this table.

Torch-free: the config schema imports this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BodyName:
    """What one ``body`` value fixes.

    ``distribution`` is what the head emits: ``"standard"`` is one number per cell and no draws;
    otherwise the name of a distribution family. ``loss_name`` is the regression loss in
    ``LOSS_REG_REGISTRY`` that trains a point body; ``None`` for a distribution, which trains on
    its own likelihood.
    """

    value: str
    distribution: str
    loss_name: str | None

    @property
    def has_distribution(self) -> bool:
        return self.loss_name is None


#: The declared values (ADR-081 §2). Adding a row is a decision with an owner and a stated reason,
#: not a config edit.
BODY_NAMES: dict[str, BodyName] = {
    row.value: row
    for row in (
        BodyName("mse_softplus_point", "standard", "mse"),
        BodyName("listwise_softplus_point", "standard", "listwise_level"),
        BodyName("nll_softplus_nb", "nb", None),
        BodyName("nll_softplus_truncated_nb", "truncated_nb", None),
        BodyName("nll_softplus_mixture_nb", "mixture_nb", None),
    )
}

#: The three settings `body` replaced. A config that writes any of them is refused.
OLD_BODY_SETTINGS: tuple[str, ...] = ("loss_reg", "output_distribution", "reg_activation")


def body_value_for_old_spelling(loss_reg: object, output_distribution: object) -> str | None:
    """The ``body`` value that the old pair of settings trained, or None if no value matches.

    A distribution trained its likelihood whatever ``loss_reg`` said, so ``loss_reg`` decides only
    on ``"standard"``. Used for the refusal message and for a saved record written before #633.
    """
    if output_distribution in (None, "standard"):
        for row in BODY_NAMES.values():
            if row.distribution == "standard" and row.loss_name == loss_reg:
                return row.value
        return None
    for row in BODY_NAMES.values():
        if row.distribution == output_distribution:
            return row.value
    return None


def check_body(value: object) -> None:
    """Raise ``ValueError`` unless ``value`` is a declared body; the message lists all (#602)."""
    if isinstance(value, str) and value in BODY_NAMES:
        return
    declared = list(BODY_NAMES)
    if value is None or (isinstance(value, str) and value.startswith("__MISSING_")):
        err_msg = f"'body' is required. Declared bodies: {declared} (ADR-081)."
    else:
        err_msg = f"body={value!r} is not a declared body. Declared bodies: {declared} (ADR-081)."
    logger.error(err_msg)
    raise ValueError(err_msg)


def refuse_old_body_settings(config: dict) -> None:
    """Raise ``ValueError`` if ``config`` writes any of the three settings that ``body`` replaced.

    The message names the ``body`` value to write when the old settings determine one.
    """
    old = [key for key in OLD_BODY_SETTINGS if key in config]
    if not old:
        return
    err_msg = (
        f"{old} {'is' if len(old) == 1 else 'are'} no longer accepted: one setting, 'body', names "
        f"the regression body (ADR-081, views-hydranet issue #633). Declared bodies: "
        f"{list(BODY_NAMES)}."
    )
    loss_reg = config.get("loss_reg")
    replacement = body_value_for_old_spelling(
        loss_reg, config.get("output_distribution", "standard")
    )
    # A distribution beside a loss other than 'mse' trained the likelihood and threw the named loss
    # away (the ranking loss beside 'nb' was refused for that until #633). Suggesting the
    # distribution's body would repeat the swap, so no value is suggested (code review of #633).
    if replacement is not None and BODY_NAMES[replacement].has_distribution:
        if loss_reg not in (None, "mse"):
            replacement = None
    if replacement is not None and "body" not in config:
        err_msg += f" These settings trained body={replacement!r}; write that and delete them."
    logger.error(err_msg)
    raise ValueError(err_msg)


def body_name(config: dict) -> BodyName:
    """What this config's ``body`` fixes. The one reader: every package site that needs the
    body's distribution or loss calls this. Raises on a missing or undeclared value and on any of
    the three old settings, so no reader can fall back to a default."""
    refuse_old_body_settings(config)
    value = config.get("body")
    check_body(value)
    return BODY_NAMES[value]
