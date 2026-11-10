# 01 — Annotated literature

**Serves:** the *"do we think we could"* half of Simon's question — what the field already knows
about training a head to order rather than to estimate, and where it has nothing to say.

Written 2026-11-10. Corpus: `~/brain/9_library` (14 papers fetched 2026-11-09 for this programme,
58 claims, `C-4746`–`C-4803`; three audit rounds closed on 13 of 14). Grouped by the role each
plays in `02_design`, with **what we take** stated per source.

⚠️ **Provenance rule.** A library claim id (`C-xxxx`) means the passage was extracted and
opponent-audited. Where an audit is **pending**, no claim is taken and any quotation was verified
directly against the PDF, which is said in place.

---

## 1 The objective — what the body would be trained against

**`Zhu2020_ListwiseUniqueRatings`** · Zhu, *Listwise Learning to Rank by Exploring Unique Ratings*,
arXiv:2001.01828 · `C-4769`–`C-4772`
⭐⭐ **The load-bearing paper.** Standard Plackett–Luce listwise losses index their steps by
*document*; this one indexes by **distinct rating level**, which collapses `n` steps to `|R|`.
**What we take:** Eq. 2 entire, as `02 §3.1` — including the two properties that make our 99.6%-zero
field tractable (all-equal ratings ⇒ loss exactly 0; the lowest level is the dropped final step).
Four model variants are offered (uRank, uBoost, uMart, uMart+); we take the loss, not the variants.
⚠️ Its benchmarks are OHSUMED/MQ2007/MSLR — graded relevance with a handful of levels, never a
field that is 99.6% one level.

**`Jagerman2017_LabelAmbiguityListwise`** · Jagerman, Kiseleva & de Rijke, *Modeling Label Ambiguity
for Neural List-Wise Learning to Rank*, arXiv:1707.07493 · `C-4773`–`C-4775`
Label ambiguity — many items sharing a relevance label, so their relative order is undefined — makes
ListNet and ListMLE **overfit**. ListPL samples training permutations from the Plackett–Luce
distribution *of the labels* rather than from one ground-truth permutation.
**What we take:** the **failure signature**, not the method. Mishandled target-side ties present as
validation decay while training performance still improves. That is a watchdog for `04` phase 4, not
a bias we must correct up front.
⚠️ Renamed in the library from an initial `Bruch2019` misattribution; identity verified from page 1.

**`Bai2023_RegressionCompatibleListwise`** · Bai et al., CIKM 2023
A standard ranking loss improves NDCG and **degrades** regression accuracy; they propose an
objective that improves both.
**What we take:** the trade-off as a **named, accepted cost** (`02 §6.6`). Simon has scoped magnitude
out — *"Only ranking"* — so the regression-compatible variant is held in reserve for the day
magnitude returns, not adopted now.

**`Wang2018_LambdaLossRankingOptimization`**, **`Xia2008_ListwiseLearningToRank`**,
**`Burges2005_LearningToRankGradientDescent`** · pre-existing in the library
The lineage: RankNet's pairwise logistic, ListMLE as the one listwise loss of three satisfying
consistency, LambdaLoss/NDCG-Loss2 as the metric-driven family.
**What we take:** the map, and the reason we are not at the pairwise end of it — see §3 below.

## 2 The emission — how a ranking becomes magnitudes

**`Clark2004_SchaakeShuffle`** · Clark, Gangopadhyay, Hay, Rajagopalan & Wilby, *J. Hydrometeorology*
5(1), 243–262 · `C-4746`+ · obtained via institutional DOI
⭐⭐ The original Schaake shuffle: reorder ensemble members to match the rank structure of historical
observations.
**What we take:** the construction of `02 §1`, **and** §4c's warning. Where the ensemble has fewer
wet days than observed, zero-forecast days get matched to positive observed amounts and *"their
assignment to a given ensemble member is entirely random. **This latter problem is intractable.**"*
⭐ That is our tie-at-zero block, named in 2004 in a dry-basin regime by the method's own authors —
and `02 §5` measures its size (`tie@0`, 0.635 → 0.000 across τ) and shows it tracks the entire
penalty at Spearman 0.994. ⚠️ The same paragraph concludes the shuffle *"certainly provides a
significant improvement over the unshuffled ensemble output"*, so §4c names which direction of
mismatch to instrument — it is not a veto. Quote verified locally at PDF lines 828–835.

**`Schefzik2013_EnsembleCopulaCoupling`** · Schefzik, Thorarinsdottir & Gneiting, arXiv:1302.7149
ECC as the unifying framework; the Schaake shuffle is one member of it.
**What we take:** the vocabulary, and two inherited assumptions stated in `02 §6.3` — ECC assumes a
**perfect model for the rank dependence structure**, and the output ensemble size equals the raw
one (**16** for us).

**`Walz2024_PrecipitationTropicalAfrica`**, **`Walz2024_EasyUQ`**, **`Biegert2026_CalibrationIDR`**,
**`Lipiecki2024_IDRElectricityPrices`** · pre-existing
A two-stage CNN + EasyUQ (isotonic distributional regression) beat raw and postprocessed ECMWF
ensembles on both occurrence and amount over northern tropical Africa.
**What we take:** the strongest existing evidence that a **nonparametric magnitude** stage works in a
zero-inflated geophysical field at scale. ⚠️ Its stage-1 CNN is trained with **MSE**, not a ranking
loss — so it supports the emission half of our design and **not** the training half. That boundary
is `02 §6.1`.

## 3 ⛔ The theory that constrains the choice of loss

**`Calauzenes2012_NonExistenceConvexRankingSurrogates`** · NIPS 2012, from the HAL archive
**Corollary 4:** no convex surrogate is calibrated with respect to Average Precision, ERR, or
pairwise disagreement. **Remark 1:** convex calibrated surrogates **do** exist for DCG.
**What we take:** the boundary of the admissible design space. Verified against the proofs, not the
abstract.

**`Duchi2010_ConsistencyRankingAlgorithms`** · ICML 2010, Berkeley tech report
**Thms 11–12:** standard pairwise convex surrogates are inconsistent *even in low-noise settings*.
**Lemma 7:** optimal `α` for a pairwise loss is NP-hard. **Thm 13 / Cor. 15:** a value-regularized
linear loss **is** consistent under low noise.
**What we take:** the reason `02 §3.2` rules out a RankNet-style objective on theory rather than on
preference. ⚠️ Duchi *conjectured* the pairwise-disagreement non-existence; Calauzènes then proved
it — the two are a sequence, not independent confirmations.

**`Blondel2020_DifferentiableSortingRanking`** · ICML 2020 (PMLR 119, 950–959),
doi:10.48550/arXiv.2002.08871
`O(n log n)` differentiable sorting and ranking via projection onto the permutahedron.
**What we take:** the **reserve option**. It sidesteps Calauzènes by being non-convex rather than by
choosing DCG, so it is the alternative if Eq. 2 proves untrainable at our sparsity. Held, not
adopted — one mechanism at a time.

**`Grover2019_NeuralSortContinuousRelaxations`** (ICLR 2019) ·
**`Swezey2021_PiRankDifferentiableSorting`** (NeurIPS 2021)
NeuralSort's continuous relaxation of the permutation matrix, and PiRank's divide-and-conquer
scaling of it.
**What we take:** nothing yet. Both are `O(n²)`-flavoured in their base form and our list is 1,024
per patch — affordable, but they add a relaxation temperature as a second free parameter when we
have no margin budget. Noted for the reserve.

## 4 Rare spatial events — the closest published analogues

**`Mohler2020_CrimeRankHotspots`** · Mohler, Porter, Carter & LaFree, *Crime Science* 9(3),
doi:10.1186/s40163-020-00112-x · ⭐ **AUDIT CLOSED 2026-11-10 — claims may now be taken**
⭐⭐ The closest published analogue to this programme: learning-to-rank for rare spatial events on a
grid, optimising PAI@k directly via a LambdaMART-style pseudo-derivative of a non-smooth metric.
Reports +36% (burglary) and +28% (robbery) over the next-best method, and **+42% on Baghdad IED
incidents**.
**What we take, verified directly against the PDF because the audit is open:** the top-k blindness
of `02 §3.3` (lines 690–693) — *"PAI is not changed by a re-ordering of the top grid cells ranking"*
— which is the reason we do **not** truncate at `k`. And the empirical demonstration that
Calauzènes is sidesteppable in practice by a non-smooth metric plus a pseudo-derivative.
⭐ **Audit closed 2026-11-10** — reported by the `library` session and **verified here against the
artifacts rather than taken on report**: 6 claims `C-4799`–`C-4804` in
`~/brain/9_library/papers/_claims/Mohler2020_CrimeRankHotspots.json`, 4 relations, and the opponent
seat's dossier at `~/brain/9_library/opponent/dossiers/single_mohler2020_2026-11-10/AUDIT.md`
(three rounds; `H-1` of 2026-11-09 — the metadata/PDF mismatch, where this record described the
2020 paper while holding the 2018 one — discharged, both papers now held separately). The
paraphrase-as-verbatim pattern that gated this paper (`H-2`, 28 of 42 key passages in the original
13-paper batch) was fixed **before** Mohler 2020 was ingested, so its passages were extracted under
the corrected protocol.

⚠️⚠️ **One caveat that survives the closure, and it lands on `02 §3.3`.** The audit's own finding
**`S-6`** is that **`C-4803` drops the paper's hedge**: the paper says *"This is **likely** due to
the fact that PAI is not changed by a re-ordering…"*, and the claim renders it *"arises because"* —
a conjecture restated as a mechanism. The hedge survives in the key passage and is lost only in the
claim. ⇒ **Cite the key passage, not `C-4803`.** `02 §3.3` quotes the passage verbatim *with*
*"likely"* intact, which is what the design argument needs, and §3.3 now states the split
explicitly.

**`Mohler2018_RotationalGridPAI`** · Mohler & Porter, *Statistical Analysis and Data Mining*
Joint selection of grid size/orientation plus a scoring function that directly maximises PAI; top
submission in the 2017 NIJ challenge (team PASDA, 9 of 20 categories).
**What we take:** confirmation that direct optimisation of a non-smooth placement metric trains in
practice. ⚠️ **Not** the paper the batch's own README described — it was fetched in place of
Mohler 2020 and renamed on audit; the README's table was never corrected.

**`An2023_SpatialRankNDCG`** · An, Zhou, Zhong & Yang, *SpatialRank*, NeurIPS 2023 · `C-4746`+
Urban event ranking with NDCG optimisation on spatiotemporal data.
**What we take:** that the NDCG family is the one actually used for spatial event ranking, which
agrees with Calauzènes Remark 1 from the applied side.

**`Guo2019_NeuralRankingModelsSurvey`** · IPM · `C-4746`+ — survey, orientation only.

## 5 The scoring and decision literature already in force

**`Arnold2024_CRPSDecompositions`** · `C-3572`, `C-3574` — `MCB − DSC + UNC`; `DSC_ISO = 0` iff the
index carries no ranking information (Prop 3.1). Behind `02 §5`'s table.
**`Brocker2009_ReliabilitySufficiencyDecomposition`** · `C-52` — proper scores cannot rank
*unreliable* schemes. ⇒ `02 §6.5`: criteria must be conditional.
**`Gneiting2011_MakingEvaluatingPointForecasts`** · `C-3531` — the scoring rule is declared **ex
ante**. ⇒ the reason `05_analysis_plan.md` is a stub rather than a document.
**`Bolin2023_LocalScaleInvarianceScoringRules`** · `C-97` — skill scores are improper even when
built on a proper rule.

---

## 6 ⛔ Gaps — what the literature does not supply

| # | gap | status |
|---|---|---|
| a | **ranking losses under a 99.6% tied-at-zero block** | ⚠️ **partially found.** Zhu 2020 and Jagerman 2017 address target-side ties; the R `PlackettLuce` package (Turner et al. 2020) generalises PL to ties of any order. **None** assumes one level holding 99.6% of items. `02 §2.4` is a construction, not a citation. ⇒ the hardest remaining gap |
| b | **a ranking objective beating a regression objective in forecasting** | ⚠️ indirect only. Bai 2023 is the clearest evidence, and it is from IR. No clean head-to-head in a forecasting context |
| c | **learning-to-rank for rare spatial events** | ✅ found — Mohler 2020 (⭐ audit CLOSED 2026-11-10), Mohler 2018, An 2023 |
| d | **ordinal / rank-based heads for zero-inflated counts** | ⚠️ partial. CORAL/CORN (arXiv:2111.08851) give rank-consistent ordinal heads; Deep Hurdle Networks (arXiv:2010.16040) handle zero-inflated multi-target regression. **No published combination.** Not fetched — neither is needed for the cheapest B |
| e | ⭐⭐ **"ranking index, then nonparametric distribution" as a two-stage design** | ⛔ **NOT FOUND.** Every Schaake/ECC design trains stage 1 with a likelihood or regression loss. This join is genuinely novel territory, which is what `02 §6.1` says |
| f | **the ceiling of a rank-coupled emission** | ⛔ **NOT FOUND.** Arnold 2024's `DSC_ISO` remains the closest. Wang & Li (COLT 2013) show NDCG → 1 as `n` grows, which is a ceiling pathology rather than a useful bound |

### To fetch — and deliberately not fetched yet

⛔ **Nothing.** Gaps (a), (e) and (f) have been searched and are empty; (d)'s two papers are not
needed for the cheapest B. The up-front literature budget is **closed**, per the handover's
*"⛔ Nothing else"*. Reopening it is a request to Simon, not a next action.

⚠️ **Noted, not fixed:** the batch's `incoming/09112026_ranking_losses/README.md` still lists
`Mohler2020_CrimeRankHotspots.pdf` under "Additional papers found" with the 2020 citation, although
the file that landed there at the time was Mohler & Porter 2018. The database is correctly
attributed; the README table is stale. Belongs to the `library` session, not here.
