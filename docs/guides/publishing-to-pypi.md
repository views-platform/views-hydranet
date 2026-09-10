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
| publish to PyPI | — |

Two details worth knowing:

- The version guard falls back to `0.0.0` when the package 404s, **not** to the version being
  published. A fallback equal to your own version makes the strict `>` compare a value with itself
  and fail on exactly the first release it exists to permit. (That is a live defect in the
  `views-r2darts2` copy of this guard — do not copy that file.)
- The TestPyPI publish passes `--check-url`, so a file already uploaded is skipped rather than
  failing. TestPyPI refuses to overwrite a version, so without it any re-run of the same version
  would fail.

## Rehearsing without publishing

Actions → **Publish Package** → *Run workflow*, leaving **"Stop after TestPyPI"** ticked. It
builds, runs both guards, uploads to TestPyPI and installs back — and stops. Nothing reaches real
PyPI.

Useful before a real release, and the way to prove the Trusted Publishing setup works without
spending a version number.

## §B — the full clean-room import check (manual)

The workflow's install-back deliberately uses `--no-deps`. It proves the artifact is on the index
and installable; it does **not** import the package, because importing needs `torch` and the
default CUDA wheel is several GB — that would dominate every release.

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
