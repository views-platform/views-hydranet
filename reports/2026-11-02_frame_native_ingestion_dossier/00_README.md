# Frame-native input ingestion — working dossier

**The decision lives in `docs/ADRs/proposed/072_frame_native_input_ingestion.md`.** This dossier
holds the material that does not belong in an ADR: the primer, the worked example, the code
walkthrough, the directory-reorganisation proposal, and the self-audit.

| file | what it is |
|---|---|
| `01_WORKING_NOTES.md` | the original 487-line document, kept whole as the record of how the design was worked out. ⛔ **Its citations have drifted and four of its implementation claims are false** — see the corrections below before relying on any of it. |

## Corrections to `01_WORKING_NOTES.md`

Established by falsification audit, 2026-11-02. Full sources in
`reports/2026-11-02_adr_claim_verification/00_CLAIMS.md` §6.

| the notes say | actually |
|---|---|
| pandas is an undeclared dependency | **declared since 2026-10-20** (`pyproject.toml:67`), under a comment naming this very ADR as the thing it was *not* doing. One of the two stated motives no longer exists |
| `apply_blueprint`'s one live consumer is the actuals seam | it runs on **every** training, eval and forecast run — `DataFetcher.standardize_raw_df` ends by calling it (`data_fetcher.py:151`), called at `hydranet_manager.py:130` |
| integer identifier dtype gives "completeness" | integer dtype rules out a **null label on a row that exists**. It says nothing about a **row that is absent**. The notes' own Beige rule requires coverage to be tested separately |
| `.duplicated()` is diagnostic-only and out of scope | it is in `_check_spatiotemporal_uniqueness` (`data_sniffer.py:285-286`), called from `sniff_ingestion` (`:54`), which the notes list as **in scope** and which runs on every run |
| §4.2/§6's Inbound role | contradicted by the notes' own Amendment 1, which is the correct version. §6 was never amended, so the scope statement a reviewer reads is the superseded one |
| the §4.1 worked example | built on the Africa crop (`row_offset=87, col_offset=310, 180×180`), which was retired fleet-wide on 2026-09-19 (ADR-077). The live grid is global |

⚠️ **Eight of the 53 `file:line` anchors had drifted or gone** when audited; the list is in
`00_CLAIMS.md` §6. The notes' own Amendment 1 said *"re-pin before review"* and was not acted on.
