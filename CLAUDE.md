# views-hydranet — two rules that have cost us real work

⚠️ **Deliberately two rules, not twenty.** A long file gets skimmed, and being skimmed is the
failure this file exists to fix: both rules below were *already written down* elsewhere and were
tripped over anyway. Add a third only when something has actually gone wrong twice.

Both have the same root cause: **`reports/` is in `.gitignore`, but its files ARE tracked** (they
were force-added). The index and the ignore file disagree, and every tool that trusts one of them
is wrong about the other.

---

## 1 ⛔ Use `command grep` for anything under `reports/`

`grep` here is a shell function wrapping **ugrep with `--ignore-files`**, which honours
`.gitignore` rather than the git index. So a recursive `grep` over `reports/` silently under-reports.

Measured 2026-11-03, same pattern, same directory:

```
grep -rn '28 min/origin' .          ->  1 hit
command grep -rn '28 min/origin' .  -> 15 hits, across 10 files
```

⇒ **An absence proved by a recursive `grep` under `reports/` proves nothing.** Never write "not
found" on that basis.

⚠️ It is **intermittent by invocation shape** — passing an explicit `reports/` path can still return
hits — which is exactly why it survives being noticed.

ℹ️ `reports/2026-09-30_evaluation_learning_report.md` Appendix F.4 recorded this hazard and F.5
made it a rule. It was then tripped over again by a session with that file open. That is why it is
here instead.

## 2 ⛔ `git add -f` with NAMED FILES, never a directory (C-395)

To commit anything under `reports/` you must force-add it. Two ways that goes wrong:

- **`git add -u` stages only files git already tracks.** A *new* file under `reports/` is silently
  left out. On 2026-11-04 this nearly shipped four tools importing `exp03_arms.py` — a module that
  would not have been in the repo. Green locally, `ModuleNotFoundError` on any fresh clone.
- **`git add -f <directory>`** pulls in caches, `__pycache__`, and multi-GB result artifacts.

⇒ Name every file. After committing, check what actually went in:

```
git show --stat HEAD
```

---

## Running anything

```
conda run --no-capture-output -n views-hydranet-env python <script>
```

⚠️ Without `--no-capture-output`, `conda run` buffers and a long job looks hung.
