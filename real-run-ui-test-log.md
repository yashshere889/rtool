# Real-run test of the UI/UX changes (PR #2) on Barkla

Started 2026-11-16. Branch `claude/advisor-fable-87d3bf` at db96c85.

## Deploy (2026-11-16 ~01:30)

- Barkla code before: main 86e90a4 + the replay patch. The files to replace were checked first: all equal to main.
- Backup: `~/scratch/rtool/code-backups/web-before-ui-20261116.tgz` (web/ folder + the two test files).
- Copied: `web/static/` (bundle replaced inside that folder only), `web/api/{hypotheses,literature,rounds}.py`,
  `tests/web/test_{hypotheses,literature}_api.py`. 22 files re-hashed: all match the branch.
- Local check first: a throwaway merge of origin/main into the branch passed tests/web and the full suite (branch deleted).
- Barkla is now: main + replay patch + these web files.

## Feature checks

Barkla tests: job 11333918, full suite passed (1 skip: bundle freshness, no frontend sources there).
UI job 11333919 on node021, tunnel 127.0.0.1:8780.

| Feature | Where | Result |
|---|---|---|
| Tab title count | Projects page | OK: "(7) Research projects" |
| Round banner (real keep/drop round) | ai-safety-benchmarks/20261029-161829-ded6, Literature | OK: 24 dropped + 7 kept titles, "7 new papers" = the 7 kept. **Problem: 24 full titles = a wall of text; long lists need "and N more".** |
| New research form → run | ui-real-test/20261116-013530-1fc7 (started 01:35; ml_dl, every step, 8 GPU-h cap, HF-only constraint) | OK: redirected to the Overview; tab title "Working: Literature · …" |
| Cluster-queue line | same run, Overview | **OK live (could not be shown in the demo):** "Waiting in the cluster queue since just now (job 11333926); the cluster expects it to start 2026-11-16 08:01." **Problem: while queued, step 1 already shows a spinner ("Planning the searches") as if running.** |
| Finished summary | verify-audit-fixes/verify-20261112 | OK: hypothesis, ECE comparison (No clear difference, p adj 0.421), reviewer verdict, links. **Follow-up: main has Report.hypothesis_outcome ("inconclusive" here); the card could show it once the branch has main.** (The "~8 s" first written here was wrong: it was my fixed 8 s wait, not a measured time; see below.) |

## Fixes made while the episode job waits (2026-11-16 ~05:50, uncommitted)

- **Long title lists:** `TitleList` in `RoundBanner.tsx` shows 3 titles + an "and N more" button (expands in place); used for
  kept, dropped and "no longer included". Lines now start with the count ("You dropped 24 papers: …").
  Checked live on ai-safety-benchmarks Literature: three lines of text instead of a wall; expanding works.
- **Spinner while queued:** `stageProgress` shows no step as current while the worker job is queued.
  Checked live on ui-real-test (job 11333926 PENDING): all 10 steps numbered, no spinner.
- Frontend 52 tests, tests/web passed, eslint clean. Bundle rebuilt (stamp 7621651002e7) and copied to Barkla's
  `web/static` (per-file hashes match). The first copy carried macOS `._*` files; deleted, recopied with `COPYFILE_DISABLE=1`.

## Hypothesis outcome on the finished summary (2026-11-16 ~06:15, uncommitted)

- origin/main (da6980a) merged into the branch, uncommitted, no conflicts; main changed no web or frontend files.
  Full Python suite passed on the merged tree.
- `ReportView` gains `hypothesis_outcome` and `hypothesis_checks` (from the Writer's report); FinishedSummary shows a
  Supported/Refuted/Inconclusive badge "judged against its success criterion" with the checks under it.
  Tests: test_report_api asserts the demo's "supported"; FinishedSummary test checks the badge and checks.
- Barkla: `web/api/report.py` (backup `~/scratch/rtool/code-backups/report.py.before-outcome-20261116`) + new bundle
  (stamp f9d377f6e8e2); hashes match. UI job restarted to load the Python change: 11333919 cancelled,
  now **11338803 on node041** (new token). The run's episode job 11333926 was not touched.
- Checked live on verify-audit-fixes/verify-20261112: "Inconclusive" badge + both checks (ECE fell 0.2 pp, p = 0.421;
  accuracy fell 0.357 pp, p = 0.188).

## Overview speed, measured (2026-11-16 ~06:30, UI job 11338803)

- Browser, verify-20261112 Overview after reload: all API calls done ~0.65 s after start (slowest /hypotheses 0.27 s).
- curl over the tunnel, cold then warm: run detail 0.02-0.13 s; /hypotheses 0.33-0.64 s (bdf6); /report 0.15 s; others < 0.06 s.
  Live run (queued job, asks Slurm): 0.06 s; squeue itself 0.02 s. Runs list: 1.26 s first call, 0.27 s after.
- Bundle over the tunnel: index JS 639 kB in 0.08 s (not gzipped; doesn't matter at this speed).
- Not reproduced, so no code change. Not measured: the very first requests right after a UI job starts.

## Real run: literature checkpoint (2026-11-17, UI job 11359839 on node041)

Background: episode 11333926 started on gpu21 (A40) 2026-11-16 08:01, was PREEMPTED twice on gpu-a-lowsmall
(08:25 after download, 13:02 at the gate) and requeued by Slurm; rtool resumed from the last finished phase each time.
Gate literature_review opened 2026-11-16 13:02. The Mac lost the VPN ~10:00, so nothing was driven until 2026-11-17 19:35.

| Feature | Result |
|---|---|
| Tab title | OK: "● Your decision needed: Review the literature · …" |
| Activity feed after preemption | OK: "Literature was interrupted and will continue where it stopped", "Literature started (continuing)". It does not say why (preempted) or show the 4.5 h requeue wait. **Minor: "Literature: Llm server ready" is a raw event name (no DESCRIBE entry for llm_server_ready).** |
| Decision card warnings | None shown: 7 quality checks passed; snowball 404s are not in the allowlist (correct). |
| Keep/Drop marks | Marked 2 Keep (SelfMix, "Is BERT Robust to Label Noise?") + 2 Drop (Annot-Mix, Twin Contrastive). Approve disabled with "You marked 4 papers … send them from the Literature page, or discard them". Bottom bar "You marked 2 to keep and 2 to drop". |
| Marks across the reader | OK: opened the BERT paper in the reader, went back: marks and the note still there. **Reader "Web page" view was still blank after 5 s (not investigated).** |
| Request changes dialog suggestions | No chips, correctly: no warnings, thinnest sub-topic has 5 papers. |
| Send choices only | Confirmation lists Keep/Drop titles; sent with no comment → toast "done", Literature "Will run again". New episode 11359859 got an H100 at once (gpu-h100). |
| Live progress while running | OK: "episode job 11359859 (RUNNING)", step 1 spinning, others numbered. |

Content notes (agent quality, not UI): paper summaries are bent toward the topic ("Annot-Mix … small pretrained text
classifiers" although it is not about them); one included paper is from Physics in Medicine & Biology (doi:10.1088/1361-6560/ad4083).

## Literature redo hang (2026-11-17, episode 11359859 on gpu33)

- **Feed wording fixed:** `llm_server_ready` now reads "AI model ready on H100 80GB HBM3 (gpu33) after 69 s". `llm_server_failed` got a line too. Deployed to Barkla and checked in the live feed.
- **Hang:** after "AI model ready" at 19:38, nothing happened for over an hour. rtool sat at 0.1 % CPU and the GPU at 0 %.
  - The main thread was in `fcntl(F_SETLK)` on `~/.cache/Microsoft/DeveloperTools/.onnxruntime/onnxruntime.db`, the onnxruntime 1.30 telemetry db, reached through fastembed.
  - **Cause:** NFS locking on `/users` is broken on gpu33. `flock -n` on a home file worked on the login node but hung on gpu33 until killed.
  - **Proof on gpu33:** with `ORT_DISABLE_TELEMETRY=1` fastembed loads and never opens the db; without it, it hangs.
  - **Fix:** `os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")` in `FastEmbedEmbedder.__init__` (vectors.py). Local only: the copy to Barkla was blocked.
- **UI gap:** the page showed RUNNING with step 1 spinning for an hour. Nothing marked the stall.
- **After the fix:** vectors.py was copied to Barkla (hashes match), and 11359859 was requeued with gpu33 excluded. It waited in the queue until about 21:48, and the AI model was ready on gpu32 (H100) at 21:49. Literature round 2 finished and the checkpoint reopened at 21:51 (no comment, so no new search). The fix works on a healthy node; it is still untested on gpu33 itself.
