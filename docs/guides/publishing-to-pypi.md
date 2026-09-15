# Publishing `views-hydranet` to PyPI

## Why this package is published at all

Eight models in `views-models` — `blazing_meteor`, `blue_stranger`, `bold_comet`,
`bright_starship`, `heavy_freighter`, `pink_pirate`, `purple_alien`, `violet_visitor`, the entire
constituent set of the `rusty_bucket` ensemble — declare `views-hydranet` in their
`requirements.txt`. Until this package is on PyPI, **none of them can be installed from a fresh
clone**. It works on a developer machine only because pip finds the library already present from a
local source install and never consults the index.

## TL;DR — cutting a release

```bash
# 1. bump the version on a branch. A published version can NEVER be reused or overwritten.
#    pyproject.toml -> [project] version = "X.Y.Z"

# 2. merge to development, then to main

# 3. cut the GitHub Release from main with tag vX.Y.Z
#    Releases -> Draft a new release -> tag vX.Y.Z -> Publish

# 4. watch Actions -> "Publish Package". It goes TestPyPI -> install-back -> PyPI.
#    Then confirm: https://pypi.org/project/views-hydranet/
```

**A tag is not a release.** The workflow fires on `release: published`. Pushing a `vX.Y.Z` tag and
stopping there leaves a tag in git that looks authoritative while `pip install` 404s. That is not
hypothetical: a sibling package tagged `0.1.0`, never cut the Release, and twelve `views-models`
models ended up pinned to a version that does not exist.

## ⛔ Before the FIRST release: this workflow has never executed

Everything below is reasoned from the file and verified statically. **No run of `publish_package.yml`
has ever happened**, in any form. `workflow_dispatch` only offers workflows that are present on the
**default branch**, so until `main` carries this file the rehearsal cannot even be triggered —
verified: `HTTP 404: workflow publish_package.yml not found on the default branch`.

The order is therefore fixed, and step 3 is not optional:

1. the release PR merges to `main`
2. Actions → **Publish Package** → *Run workflow* — a manual run is always a rehearsal
3. **it goes green**
4. **only then** cut a GitHub Release

**Do not cut a Release before step 3.** The whole argument for putting the rehearsal *inside* the
workflow is that a rehearsal you have to remember is one you skip on the release that matters. A
first Release that fails does so publicly, and a PyPI version can never be deleted or reused.

Delete this section once the first rehearsal has run green. (Epic #353 / S6 #359, S10 #363.)

## One-time setup — Trusted Publishing, and it is TWO publishers

This repository holds **no API token**. Authentication is PyPI Trusted Publishing: PyPI mints a
short-lived credential for a named repo + workflow, so there is no secret to store, rotate or leak.

Because the workflow rehearses on TestPyPI first, you must register the publisher on **both** sites:

| site | where |
|---|---|
| PyPI | <https://pypi.org/manage/account/publishing/> |
| TestPyPI | <https://test.pypi.org/manage/account/publishing/> |

Both with:

```
Owner:        views-platform
Repository:   views-hydranet
Workflow:     publish_package.yml
Environment:  (leave blank)
```

For a package that does not exist on the index yet, register a **pending publisher** — that is
what authorises the first-ever upload. TestPyPI and PyPI are separate services with separate
accounts; registering one does not register the other.

**If the TestPyPI publisher is missing, the workflow fails at the first publish step and nothing
reaches real PyPI.** That is the safe direction to fail in.

## What the workflow does, and why it differs from the sibling repos

`views-baseline`, `views-datafactory`, `views-frames`, `views-postprocessing` and
`views-reporting` all publish straight to PyPI and treat TestPyPI as an optional manual rehearsal.
This one puts the rehearsal **inside** the workflow, so it cannot be skipped on the release that
actually matters.

| step | what it protects against |
|---|---|
| tag == pyproject version | releasing `v0.2.0` from a tree that still says `0.1.0` |
| pyproject version > PyPI | re-releasing a version, or going backwards |
| `uv build` + `twine check` | a malformed wheel or unrenderable README |
| **publish to TestPyPI** | discovering an upload problem on the real index |
| **install back from TestPyPI** | a wheel that uploads but is not actually installable |
| **`.github/scripts/wheel_contract.py`** | a wheel with correct metadata and **no Python modules** |
| publish to PyPI | — |

Not a step, but the invariant that holds the table together: **the tag guard and the real publish
carry the identical condition** (`github.event_name == 'release'`), so there is no path that
publishes while skipping the guard. That was not true before **C-341**.

Two details worth knowing:

- The version guard falls back to `0.0.0` when the package 404s, **not** to the version being
  published. A fallback equal to your own version makes the strict `>` compare a value with itself
  and fail on exactly the first release it exists to permit. (That is a live defect in the
  `views-r2darts2` copy of this guard — do not copy that file.)
- The TestPyPI publish passes `--check-url`, so a file already uploaded is skipped rather than
  failing. TestPyPI refuses to overwrite a version, so without it any re-run of the same version
  would fail. The skip is **hash-checked**: uv skips only a byte-identical file and hard-fails on a
  same-name file with different bytes (verified on uv 0.8.13; the action pins that version). So the
  skip can never validate the wrong file — what it cannot do is un-burn a version.
- **A manual run never uploads at the release version.** Before it builds, a dispatch rewrites the
  `pyproject` version to `X.Y.Z.devN`, where `N` is the run number. PEP 440 orders `X.Y.Z.devN`
  strictly *before* `X.Y.Z`, so a rehearsal can never occupy the version a Release will need on
  TestPyPI (**S6/#359**).
- **The contract check imports the package.** `views_hydranet/__init__.py` is a lazy PEP-562
  `__getattr__`, so `import views_hydranet` is torch-free and runs inside the `--no-deps` venv at no
  cost. Both CI and the release job run the same file, `.github/scripts/wheel_contract.py`; they
  asserted the contract separately until 2026-09-14 and had already drifted.

## Rehearsing without publishing

Actions → **Publish Package** → *Run workflow*. It reads the version, stamps a throwaway `.devN`
onto it, builds, uploads to TestPyPI, installs back and runs the contract check — and stops.
**Nothing reaches real PyPI, and there is no option to make it.**

GitHub offers no branch restriction on `workflow_dispatch`: whoever dispatches picks the ref, and
the job builds that ref's tree. The `.devN` stamp is what makes that harmless. Without it, a
rehearsal from a feature branch would upload that branch's tree at the release version and burn
that version on TestPyPI permanently. The real Release at that version would then reach the
TestPyPI step, find a file with the same name and different bytes, and **die there** — uv refuses a
hash mismatch — with a version it can never reuse on TestPyPI (**S6/#359**). *(An earlier draft of
this section said the mismatched file was silently skipped and an unverified wheel went on to real
PyPI. That is not what uv does; the failure is a blocked release, not a wrong one.)*

⚠️ **A rehearsal exercises neither guard.** Both the tag-versus-`pyproject` check and the
newer-than-PyPI check are `if: github.event_name == 'release'`. The second is release-only because
a rehearsal is usually dispatched from `main` at the version already on PyPI — to prove the Trusted
Publishing setup without spending a version — and a strict `>` would fail every such run. A green
rehearsal is evidence that the build, the upload, the install-back and the contract check work; it
is **not** evidence that either guard works, because neither is on the path a rehearsal takes.

That is deliberate. An earlier version had a "Stop after TestPyPI" checkbox, ticked by default;
unticking it published whatever version sat in `pyproject.toml` on the default branch, with no tag,
no Release, and skipping the tag guard — which only runs on a release event. A PyPI version cannot
be deleted or reused, so a mis-click was permanent and public. The checkbox is gone (**C-341**):
a manual run is always a rehearsal, and real publication happens only through a published Release.

Run it before a real release, and to prove the Trusted Publishing setup works without spending a
version number.

## §B — the full clean-room import check (manual)

The workflow's install-back uses `--no-deps`, and its contract check imports only the **top level**
— which is free, because that import is lazy. What it cannot reach is anything that actually needs
`torch`, whose default CUDA wheel is several GB and would dominate every release.

So the automated check proves the package tree shipped and is importable; it does not prove the
model code loads against real dependencies.

Do the full check by hand once per meaningful release, using the CPU-only torch index so it takes
minutes rather than an hour:

```bash
uv venv /tmp/hydranet-check --python 3.11
uv pip install --python /tmp/hydranet-check/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "views-hydranet==X.Y.Z"

/tmp/hydranet-check/bin/python -c "
import views_hydranet
from views_hydranet.utils.config_initializer import HydraNetConfig
print('OK', views_hydranet.__file__, len(HydraNetConfig.model_fields), 'config fields')
"
```

Note this installs from **real PyPI**, so do it after a release. To do it against TestPyPI, add
`--index-url https://test.pypi.org/simple/` — TestPyPI does not carry our dependencies, so the
`--extra-index-url` is what makes them resolvable.

## Versioning

`0.1.0` was the first published version, deliberately: all eight downstream models pin
`views-hydranet~=0.1.0`, and shipping `1.0.0` would have required editing all eight in lockstep.

Under semver a `0.x` release promises nothing across minors — `0.2.0` may break everything `0.1.0`
did. That is why the downstream pins are `~=0.1.0` (i.e. `>=0.1.0,<0.2.0`) rather than `<1.0.0`,
which would have accepted a breaking `0.2.0` silently.

**Before publishing a `0.2.0` with breaking changes, bump those eight pins in the same window.**

Whether a later release should be `1.0.0` is deferred and unresolved — see issue #347. It is
blocked on #181, because semver's guarantee applies to a *defined* public API and
`views_hydranet/__init__.py` currently declares none.

## Troubleshooting

**"No solution found … there is no version of views-hydranet==X" at the install-back, right after
a successful TestPyPI upload.** The index lags the upload by up to a minute. The step retries five
times, 20 s apart, since the first-ever run hit exactly this (2026-09-15). If it still fails after
five, check `https://test.pypi.org/simple/views-hydranet/` by hand — the file is either there
(then it is something else) or the upload step lied.


**`Trusted publishing exchange failure`** — the publisher is not registered on that site, or one of
owner / repo / workflow filename does not match exactly. Check the *other* site too; TestPyPI and
PyPI are configured independently.

**`File already exists`** — that version is already on the index. Versions are immutable; bump and
release again. On TestPyPI the workflow's `--check-url` should have skipped it, so seeing this on
TestPyPI means the flag was removed.

**Version guard fails on a first release** — check the fallback is `0.0.0`. See the note above.

**Workflow did not run at all** — you pushed a tag instead of publishing a Release.

## Provenance

Written 2026-09-10 alongside the workflow, for issues #340 (publish) and #346 (how this repo
authenticates). Structure follows `views-baseline`'s `docs/guides/publishing-to-pypi.md`; the
in-workflow TestPyPI stage is specific to this repo. Risk register: C-333, C-334, C-335, C-337,
C-340.
