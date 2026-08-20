# Git History Purge Plan — PerforatedAI/PerforatedAI

**Scheduled:** midnight EDT, 2026-08-15 (i.e. 2026-08-14 → 2026-08-15 boundary), run live with Mark present.

## Why

`.git` is 598 MB, almost entirely large binaries committed over the years and never
removed from history — mostly `.pt` model checkpoints under
`examples/submitted_projects/**` (largest single blob ~64 MB) and `.whl` release
artifacts under `releases/**`. A prep commit (`e00defe`, pushed 2026-08-14) already
stopped new `.pt`/`.whl` files from being tracked going forward, but that doesn't
remove the old blobs already baked into every commit that touched them — only a
history rewrite does that.

## Pre-flight state (captured 2026-08-14)

- `.git` size: 598 MB (598.23 MiB packed, 6438 objects)
- Blobs >1MB across all history: 197, totaling ~1.64 GB uncommpressed
- Breakdown of >1MB blobs by extension: 135 `.whl`, 49 `.pt`, 6 `.ipynb`, 4 `wandb`, 2 `.pth`, 1 extensionless checkpoint
- Of the 6 `.ipynb` blobs, only 3 are being stripped (dead duplicates of
  `cifar10_dendritic_comparison.ipynb` under its old pre-rename path); the other 3,
  including the 2 still live in the current tree, are kept — see "Notebook
  exception" below
- Remote branches (23): `main`, `develop`, `Cascor-v1`, `dean_branch`, `docs-generation`,
  `docs-styling`, `examples-docs`, `impulse-testing`, `linear_dendrites`,
  `multi-dim-tensors`, `mw-telemetry-start`, `new-averaging`,
  `new-averaging-isolation`, `new-method`, `nn_customize`, `readme-link-fixes`,
  `repo-cleanup`, `setup-fix`, `skills-discoverability`, `tracker_cleanup`,
  `trainium`, `upstreamChanges`
- Tags: 43 (release tags `b2.x`, `e0.x`, `e3.x`, ...)
- Branch protection: none on `main` or `develop` (confirmed via `gh api`)
- Open PRs: 33 (list below) — all from external forks, will need to be recreated
  after the rewrite since their base history will no longer match origin's.

## wandb run directories untracked (2026-08-14)

Discovered while investigating the 4 large `wandb` blobs in the pre-flight scan:
entire local Weights & Biases run directories (debug logs, run configs,
`wandb-summary.json`, sweep configs, and the binary `.wandb` files themselves — 132
files total) were accidentally committed under
`examples/submitted_projects/{dendritic_mobilenet,emotion-recognition}/wandb/`.
Untracked in commit `48a1179` (pushed to `develop`) and `**/wandb/` added to
`.gitignore`. The 4 large binary `.wandb` blobs were already included in the
194-blob strip list (non-`.ipynb`, >1MB) — no change needed to the strip logic for
tonight, this section just documents why they're in there.

## Notebook exception (per repo owner, 2026-08-14)

The blanket `--strip-blobs-bigger-than 1M` was **replaced** with a targeted blob-ID
list after review: `.ipynb` files legitimately run >1MB because they embed output
images, and none of the 6 large notebook blobs in history exceed 10MB (largest is
2.7MB), so all live notebook content is kept. The only notebook blobs stripped are 3
dead duplicate versions of `cifar10_dendritic_comparison.ipynb` under its old
pre-rename path (`Examples/hackathonProjects/cifar10/...`), which no longer exist in
any branch tip and were called out as safe-to-remove duplicates. Rule going forward:
flag anything `.ipynb` over 10MB before stripping it; under that, keep it.

## Rehearsal already done

A dry run was performed in an isolated mirror clone (not pushed):

```
git clone --mirror <local repo> pai-mirror.git
cd pai-mirror.git
git filter-repo --strip-blobs-with-ids strip_ids.txt --force
```

Where `strip_ids.txt` is 194 blob SHAs = all >1MB blobs **except** `.ipynb` files,
plus the 3 dead duplicate notebook blobs above.

Result: 598 MB → **81 MB**, `git fsck --full --strict` clean, all `.pt`/`.whl` gone
from every branch tip, all live `.ipynb` content intact. That mirror and blob-ID list
are stale now (predate any commits after 2026-08-14) and will be **regenerated
fresh** at execution time, not reused — the size-scan and strip-list build steps
need to be rerun against `origin` state as of tonight.

## Scope update (2026-08-20): strip ALL `.pt`/`.whl`/`.pth`, not just >1MB

The blob-size threshold (>1MB) originally used to build `strip_ids.txt` left small
`.pt`/`.pth` checkpoints (under 1MB each — mostly small hackathon test-run
checkpoints) and `.wandb` run files in history. Per Mark's direction, the strip list
is now: all blobs >1MB excluding `.ipynb` (the original rule) **UNION** all
`.pt`/`.pth`/`.whl` blobs of *any* size, plus the 3 known dead-duplicate notebook
blobs. `.wandb` files are deliberately *not* included in this expanded rule (they're
handled separately below via untracking + `.gitignore`, not via history strip).
Re-rehearsed 2026-08-20 against a fresh mirror of `origin`: 624MB → 50MB, 0
`.pt`/`.pth`/`.whl` blobs of any size remaining, `fsck` clean, live `.ipynb` and tip
content unaffected. Use this expanded rule when rebuilding `strip_ids.txt` at
execution time (step 5 below).

## Pre-rewrite prep: examples-folder cleanup (added 2026-08-20)

**This is the root cause of the Saturday (2026-08-15) failure**, restated precisely so
it isn't repeated: the second push that day (14:36 EDT, "Develop (#132)") was
unrelated cleanup work of exactly this kind — deleting stale example-project output
files — pushed from this working directory *after* the rewritten mirror had already
been force-pushed (04:10 EDT) but *before* this directory had been resynced to the
new history (step 8). That push spliced the old, un-rewritten object graph back onto
`main`/`develop`, undoing the purge. The cleanup work itself wasn't the problem — the
problem was doing it out of sequence, from a stale post-rewrite-pending clone.

**Rule going forward: any prep/cleanup commits (like this one) must be committed and
pushed as an ordinary commit *before* the rewrite begins (step 2's fresh fetch will
pick it up) — never pushed in the gap between the mirror push (step 7) and the local
resync (step 8).**

Cleanup done 2026-08-20, ready to be committed and pushed as prep before tonight's
rewrite:

- **Deleted from disk** (already untracked / gitignored via `PAI*` and not yet covered
  by a wandb pattern — these never made it into a commit on this branch, so no `git rm`
  was needed, just filesystem deletion):
  - `examples/hackathonProjects/emotion-recognition/wandb`
  - `examples/hackathonProjects/dendritic_mobilenet/wandb`
  - `examples/hackathonProjects/perforated-uniplexity-credit-scoring/models/checkpoints`
  - `examples/hackathonProjects/perforated-impulse-nn-block/trained`
- **Reduced to a single `<foldername>.png`** (all other `.png`/`.csv` output files
  deleted; also untracked/gitignored, so filesystem-only, no commit needed) — 10
  folders, listed under "PNG-reduce candidates" in project memory/chat history.
  Two folders (`resnet18_&_resnet34/results`, `adult_credit_dendrites/results`) were
  **skipped**: each contains several PNGs representing genuinely different metrics
  (accuracy, compression, comparison) with no unambiguous single "main" chart —
  picking one to keep is a content judgment call, not a naming cleanup, and was left
  for Mark to decide by hand rather than guessed at.
- **`.gitignore` gap-fill**: added `*.pth`, `*.wandb`, and `**/wandb/` (this branch was
  missing the `**/wandb/` pattern that commit `48a1179` added on `develop`; `*.pt` and
  `PAI*` were already present here). This prevents the exact class of file that caused
  both the original bloat and the Saturday incident from being re-tracked by anyone's
  future commit.

**Repeatable process for next time:** this generalizes — before any future history
rewrite, (1) check whether stale output directories under `examples/**` are tracked
(`git ls-files <path>`) — if untracked, plain `rm -rf` is safe and needs no commit; if
tracked, use `git rm -r`; (2) for any "keep one representative file" cleanups, only
auto-pick when there's an unambiguous naming match (e.g. `<dir>/<dir>.png` or an
un-suffixed file matching the folder's model name) — flag anything requiring a content
judgment call instead of guessing; (3) confirm `.gitignore` covers the file types
involved so the mess can't reaccumulate; (4) commit and push this prep work as a
normal commit *before* starting the rewrite, never mid-rewrite.

## Execution steps (tonight)

1. **Freeze**: confirm no one else is pushing during the window (no branch protection
   exists to enforce this technically — it's a courtesy/timing thing only).
2. **Fresh fetch**: `git fetch --all --prune` in the working repo to pick up anything
   pushed since this doc was written.
3. **Fresh mirror**: `git clone --mirror` the up-to-date local repo into a new scratch
   directory (never reuse a stale mirror — it would silently drop any commits pushed
   after it was made).
4. **Rebuild strip list**: rerun the size scan (`git rev-list --objects --all | git
   cat-file --batch-check ...`) against the fresh mirror, take all blobs >1MB,
   exclude `.ipynb` paths, and re-add the 3 known dead duplicate notebook blob IDs
   (or any new equivalents if history has changed) → fresh `strip_ids.txt`. Flag any
   `.ipynb` >10MB to Mark before including it.
5. **Rewrite**: `git filter-repo --strip-blobs-with-ids strip_ids.txt --force` on the
   fresh mirror. This touches every ref (all branches + all 43 tags).
6. **Verify**:
   - `git fsck --full --strict` → must be clean
   - confirm no non-`.ipynb` blob >1MB remains
   - confirm live `.ipynb` files are still present and byte-identical to before
   - spot-check `main` and `develop` tips still contain the expected current files
   - compare `.git` size before/after
7. **Push**: from the mirror, `git push --force --all origin` then
   `git push --force --tags origin`.
8. **Local resync**: force-refresh the working copy at
   `/Users/mwesterlund/Projects/Perforatedai-cleanup/PerforatedAI` to the new history
   (fetch + hard reset each local branch to its new `origin/<branch>`, or just
   re-clone).

## Post-purge cleanup (not tonight, but needed soon after)

- **33 open PRs will break** (list below) — GitHub will likely show them as
  unmergeable or with garbled diffs since their merge-base no longer exists in
  origin's history. Each will need to be closed with a comment asking the author to
  re-fork/rebase and reopen, or Mark rebases them manually onto the new history.
- Old commit SHAs referenced in closed issues/PR comments will stop resolving once
  GitHub garbage-collects the orphaned objects (may remain fetchable for a while via
  GitHub's cache first).
- GitHub's reported repo size may lag the real number for a while — they repack
  server-side asynchronously after a force-push.
- Anyone else with an existing local clone must re-clone or hard-reset; a normal
  `git pull` will re-merge the old bloated history back in.

## Open PRs at time of writing (will need author follow-up post-purge)

- #100 Oyaabuun:hackathon/perforated-monai-3d-unet
- #97 PioGodwin-M:main
- #94 wildhash:claude/ai-dendritic-optimization-Yg5vu
- #91 lakshmi22-2007:main
- #90 VG-Fish:main
- #86 yeabgenet:main
- #84 PioGodwin-M:fix/resnet-cifar100-import-path
- #83 HomeroRR:main
- #82 Ruchit-ICB:hackathon/neurovision-do
- #81 AmRitJain0442:hackathon-giant-killer-nlp
- #79 AvichalDwivedi2205:add-dendrites-qwen-submission
- #77 Tasfia-17:master
- #75 HectorTa1989:feature/LocalLlamaCoder
- #73 Dreamcatcher23:main
- #68 HectorTa1989:feature/DermCheck
- #67 HectorTa1989:feature/GuardianEdge
- #66 HectorTa1989:feature/DendriticDrive
- #53 wuyaning3288:GNN_model_Replicate
- #51 Ziyan0219:main
- #50 VishalWarke29:hackathon-code-review
- #48 lucylow:main
- #47 lucylow:main
- #45 VishalWarke29:hackathon-cifar10-dendrites
- #44 VishalWarke29:perforated-cifar10-shufflenet-dendrites
- #42 aakanksha-singh-hub:submission/project-nexus
- #41 YuvrajSHAD:main
- #40 siddi7:main
- #37 hackersclub111:hackathon-dendritic-routing
- #34 ngstephen1:main
- #32 lucylow:main
- #31 kamaleshcit2024:patch-2
- #25 riush03:main
- #24 walymostafa646-create:main

## Rollback

If the push goes wrong or something looks broken immediately after: origin's
pre-rewrite state is recoverable from any existing local clone's reflog / packed
refs (including the one at `/Users/mwesterlund/Projects/Perforatedai-cleanup/PerforatedAI`
as it stands right now, before the rewrite) by force-pushing that back up. Worth
keeping this local clone untouched/unrewritten as the rollback source until the new
history is confirmed good on GitHub.

## Post-mortem: attempt #1 (2026-08-15) did not take effect

**What we know happened:** `MwesterlundDev` force-pushed to essentially every
branch (~17 refs: `main`, `develop`, and all the feature branches) within a 5-second
window at 2026-08-15T04:10:55Z–04:11:00Z (00:10 EDT — exactly the scheduled
midnight-EDT window), per GitHub's public repo events API. That's the unmistakable
signature of step 7 (`git push --force --all origin` from the rewritten mirror), and
no one else touched the repo in that window.

**What we verified afterward (2026-08-19, fresh clone + fresh bare clone of
`origin`):** the repo never shrank. `.git` is 625 MB (vs. the pre-purge 598 MB —
if anything slightly larger), still 6,390 objects in one pack, and 183 blobs >1MB
are still present in history with full-size real content, including the exact
`.pt` checkpoints the plan targeted (`unet_dendritic_new.pt` at 67 MB, etc.).

**Root cause: confirmed.** The rewrite at 04:10–04:11 EDT most likely did
succeed and was pushed correctly. The reinflation came from the second push the
same day (`main`/`develop` updated at 2026-08-15T14:36:52Z, commit
"Develop (#132)") — Mark confirmed this second push was his own, done for
unrelated cleanup work (see history further up this doc), and pushed **from this
working directory**
(`/Users/mwesterlund/Projects/Perforatedai-cleanup/PerforatedAI`). That
directory was never resynced per step 8 — it was still sitting on pre-rewrite
history at the time — so the push (or a merge underlying it) spliced the old,
un-rewritten object graph back onto `main`/`develop`. This is exactly the
failure mode called out above: *"a normal `git pull` will re-merge the old
bloated history back in"* if it comes from a stale, un-rewritten local clone.

In short: **step 8 (local resync) was skipped or happened too late** — the
window between the mirror push (04:10) and the next push from this working
copy (14:36) was over 10 hours, plenty of time for other work to happen against
the stale local clone before anyone thought to hard-reset it to the new
`origin` history.

**Process changes for the retry, to make this diagnosable if it fails again:**

- After step 5 (`filter-repo` rewrite), **verify the mirror locally before
  pushing** — `du -sh` the mirror's `.git`, re-run the >1MB blob scan against
  it, and confirm the size actually dropped (target ~81 MB per the rehearsal)
  and no non-`.ipynb` blob >1MB remains. Do not proceed to step 7 until this
  check passes.
- **Keep the rewritten mirror directory** after pushing (don't let it get
  cleaned up) until the new history is independently confirmed good via a
  *fresh* clone of `origin` from a machine/directory that had no prior local
  state — so if something looks wrong afterward, the actual pushed mirror is
  still inspectable rather than lost like this time.
- After the push, immediately re-verify against `origin` itself (not a local
  clone that might already have stale refs) — fresh `git clone --bare` and
  re-run the blob-size scan — before considering the purge done. Attempt #1
  never had this confirmation step, which is why it went unnoticed for four
  days.
- Treat any subsequent push to `main`/`develop` right after the rewrite
  (merges, PR completions, etc.) as suspect until confirmed to be based on
  post-rewrite history — a PR branch created before the rewrite, merged after
  it, will silently reintroduce the old object graph.
- **Do step 8 (local resync) immediately after step 7, before doing anything
  else** — including unrelated cleanup work. This is what actually broke
  attempt #1: this working directory
  (`/Users/mwesterlund/Projects/Perforatedai-cleanup/PerforatedAI`) sat on
  stale pre-rewrite history for ~10 hours after the mirror push before Mark
  pushed unrelated changes from it, silently reintroducing the old object
  graph. No other local clone (or fork/PR branch) should be pushed to
  `origin` until it's confirmed to be based on the new history.
