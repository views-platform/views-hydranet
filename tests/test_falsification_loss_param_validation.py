"""
Falsification stubs for PR #34 (strict conditional parameter validation).

F5: CIC §3 field count drift — guard against future miscounts.
"""

import re
from pathlib import Path

from views_hydranet.utils.config_initializer import HydraNetConfig
from views_hydranet.utils.utils import LOSS_REG_REGISTRY


class TestF5_CICFieldCountDrift:
    """CIC §3 field count must stay in sync with actual model_fields."""

    def test_cic_field_count_matches_actual(self):
        """CIC §3 field count must match actual model_fields count."""
        # docs/CICs/HydraNetConfig.md §3 (freeze_h -1; rollout_horizon +1, 2026-06-06;
        # hurdle-NB +3: loss_reg_theta_init, learnable_theta, loss_class_pos_weight; #99;
        # output_distribution +1, #100; min_free_disk_gb +1, #107/C-154;
        # static_channels +1, #108/ADR-060/C-153;
        # reg_activation +1, 2026-06-21 Exp-B head decouple;
        # quantile head +n_quantiles, pinball body +loss_reg_tau/loss_reg_cap;
        # ADR-065 / Epic #158: +body_mask, -hurdle_threshold retired -> net 78 => 81)
        # ADR-067 / Epic #167 A-S5 (#172): +n_head_samples -> 79
        # ADR-067 / Epic #167 A-S8 (#175): +max_posterior_cube_gb -> 80
        # ADR-067 / Epic #167 A-S9 (#176): +pi_penalty_weight, +pi_penalty_prior_logit -> 82
        # ADR-069 / Epic #183 S3 (#186): +forecast_composition, +gate_threshold -> 84
        # (n_head_samples etc. accounted above; 2026-07-27 baseline -> 86)
        # ADR-065 amend. 2026-07-28: +body_supervision, +onset_lead, +cessation_lag, -body_mask
        #   (net +2) -> 88
        # ADR-068 emit_family_core (th_gated_ZINBcore, Epic #167/#183): +emit_family_core -> 89
        # #287 ss_reverse (Teutsch 2022 ITF, increasing teacher forcing): +ss_reverse -> 90
        # #289 pushforward (Brandstetter 2022): +pushforward_weight,
        # +pushforward_detach_state -> 92
        # C-314 (training-loop gradient audit): +clip_grad_max_norm -> 93
        # #308 BPTT-SA: +ss_backprop_through_feedback -> 94
        # #308 GRAD-TRAJ follow-up: +ss_feedback_grad_clip -> 95
        # #311 input noise (Sanchez-Gonzalez 2020, adapted): +input_noise_dropout -> 96
        # ADR-027 §2.1 cell clamp to production: +freeze_recurrent,
        # +freeze_recurrent_weight -> 98
        # C-184 mitigation promoted out of a shadow default: +bn_recalibrate -> 99
        # #377: +require_cuda -> 100
        # ADR-078 / Epic #500 S4: +emit_horizons -> 105
        # #578 ranking body (04_roadmap phase 1): +loss_reg_y_max -> 106
        CIC_CLAIMED_COUNT = 106
        actual = len(HydraNetConfig.model_fields)
        assert actual == CIC_CLAIMED_COUNT, (
            f"CIC §3 claims {CIC_CLAIMED_COUNT} fields but HydraNetConfig has {actual}. "
            f"Update CIC §3 field count after adding/removing fields."
        )

    def test_the_count_is_read_out_of_the_cic_not_restated_here(self):
        """⛔⛔ The check above does NOT read the CIC, and has not since it was written.

        A guard audit on 2026-11-12 measured it: editing `docs/CICs/HydraNetConfig.md` to claim
        *"all 42 fields"* left it **green**, because the "claimed" count is a literal in this
        file. The test is named `test_cic_field_count_matches_actual` and its class docstring says
        *"CIC §3 field count must stay in sync with actual model_fields"* — so it has been
        carried through ~25 field additions asserting a doc-sync property it never had. All the
        `105 -> 106` edit on this branch proved is that one author changed two numbers
        consistently.

        ⇒ This reads the declaration out of the document, so the document is now the thing under
        test.
        """
        cic = Path(__file__).resolve().parents[1] / "docs/CICs/HydraNetConfig.md"
        assert cic.is_file(), f"the CIC is missing at {cic}"
        claimed = re.findall(r"all (\d+) fields", cic.read_text(encoding="utf-8"))
        assert len(claimed) == 1, (
            f"expected exactly one 'all N fields' claim in {cic.name}, found {len(claimed)}: "
            f"{claimed}. The parse is ambiguous, so it cannot be the guard."
        )
        assert int(claimed[0]) == len(HydraNetConfig.model_fields), (
            f"{cic.name} §3 claims 'all {claimed[0]} fields' but HydraNetConfig has "
            f"{len(HydraNetConfig.model_fields)}. Update the DOCUMENT, not just the test."
        )

    def test_every_required_loss_parameter_is_named_in_the_cic(self):
        """⛔ The other half of the same audit finding: **renaming** a field keeps any count
        green. Cardinality is not identity, and a count cannot see a rename.

        ⭐ `test_listwise_level_loss.py`'s registry-snapshot guard already solved exactly
        this for `LOSS_REG_REGISTRY`, and its docstring gives the reason —
        *"a snapshot rather than a count, because a changed `cls` on an existing key would keep
        any count green."* It had not been applied here.

        ⚠️ Scoped to this file's subject — the parameters a `loss_reg` **requires** — rather than
        snapshotting all 106 names in a shared guard. The set is read out of
        `LOSS_REG_REGISTRY`, so it cannot drift: add a loss with a new required parameter and this
        fails until the CIC names it. Renaming one of them fails too, which is the hole.

        ⚠️⚠️ **MENTIONED is not DOCUMENTED, and the second half below is why.** A guard audit on
        2026-11-12 showed that the name test alone can be satisfied by a bare mention: deleting
        `loss_reg_y_max`'s entire documentation paragraph left it green, because the name also
        occurs in the `reject_discarded_or_inert_listwise_level` note. Presence is not
        documentation, exactly as the earlier rewrite's own premise was that cardinality is not
        identity — the same gap one step down.
        """
        cic = Path(__file__).resolve().parents[1] / "docs/CICs/HydraNetConfig.md"
        text = cic.read_text(encoding="utf-8")
        required = {p for entry in LOSS_REG_REGISTRY.values() for p in entry["params"]}
        assert required, "LOSS_REG_REGISTRY declares no required parameters — the parse is broken"
        unnamed = sorted(p for p in required if f"`{p}`" not in text)
        assert not unnamed, (
            f"{cic.name} never names these required loss parameters: {unnamed}. Either the "
            "CIC does not document them, or one was renamed in the schema and the document still "
            "describes the old name — which the field COUNT above cannot see."
        )

        # ⭐ A RATCHET, not a clean bill of health. A parameter counts as *described* when one of
        # its mentions sits near the CIC's own entry idiom ("added …", "default …"). Measured
        # 2026-11-12: 5 of the 7 required parameters are described; `loss_reg_a` and `loss_reg_c`
        # (ShrinkageLoss) are bare mentions and are listed below rather than fixed, because
        # writing entries for two parameters whose semantics I have not studied would be inventing
        # documentation. ⛔ The list may only SHRINK: describe one and remove it from here. A new
        # undescribed parameter fails instead of joining it.
        known_bare_mentions = {"loss_reg_a", "loss_reg_c"}
        described = {
            p
            for p in required
            if any(
                re.search(
                    r"default|added|DOCUMENTED", text[max(0, m.start() - 200) : m.start() + 300]
                )
                for m in re.finditer(re.escape(f"`{p}`"), text)
            )
        }
        regressed = sorted(required - described - known_bare_mentions)
        assert not regressed, (
            f"{cic.name} names these required loss parameters but no longer DESCRIBES them: "
            f"{regressed}. A documentation paragraph was deleted or reworded while the name "
            "survived elsewhere in the file, which the name check above cannot see."
        )
        now_described = sorted(known_bare_mentions & described)
        assert not now_described, (
            f"{now_described} are now described in {cic.name} — remove them from "
            "`known_bare_mentions` so the ratchet tightens."
        )
