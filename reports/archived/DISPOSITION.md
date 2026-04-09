# Disposition: Archived Reports

| Field         | Value                                                        |
|---------------|--------------------------------------------------------------|
| Last Updated  | 2026-04-08                                                   |
| Policy        | Historical record only; do not reference from active docs    |

## Archive Contents

| Directory | Period | Description |
|---|---|---|
| `investigations_pre_adr047/` | Feb 2026 | Exploration and prototyping for the pandas-free output path (ADR-047). Contains sandbox scripts and plans referencing removed classes/methods. See its own DISPOSITION.md. |
| `investigations_jan_feb_2026/` | Jan-Feb 2026 | Major refactoring investigations: VolumeHandler redesign, unified inference, visual diagnostics, test expansion, dead code audits. All plans executed. See its own DISPOSITION.md. |
| `from_pipeline_core_to_hydranet_ticket001/` | Jan 2026 | Migration ticket: initial PredictionFrame and evaluation adapter integration |
| `from_pipeline_core_to_hydranet_ticket002/` | Jan 2026 | Migration ticket: PredictionFrame dispatcher and implementation guide |
| `2026-02-25_hydranet_hardening_dossier/` | Feb 2026 | 6-report hardening dossier: Popperian audit, evaluation handshake, feature lifecycle, technical debt |
| `post_mortems/` | Jan-Mar 2026 | 14 post-mortem reports from the initial collapse through architectural consolidation |
| `2026-03-06_structural_anomalies.md` | Mar 2026 | One-off structural anomaly report |
| `hydranet_memory_investigation.md` | Feb 2026 | GPU memory profiling investigation |
| `offline_chap_draft.md` | Pre-2026 | Draft chapter on offline evaluation methodology |
| `technical_debt_backlog.md` | Pre-2026 | Legacy technical debt list (superseded by `technical_risk_register.md`) |

## Policy

Files in this directory are retained for historical context. They must not be:
- Referenced from active ADRs, CICs, or standards
- Used as implementation guidance
- Treated as current architectural documentation

The authoritative risk tracking artifact is `reports/technical_risk_register.md`.
