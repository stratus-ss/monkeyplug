# Devlog: Merge origin/main into feature/file-chunking — 2026-07-29

## Objective / Problem
Bring `feature/file-chunking` up to date with `origin/main` (14 commits ahead)
while preserving the production-pinned behavior of `scrubword()`,
`Plugger.should_scrub_word()`, and `WhisperPlugger._poll_for_transcription_result()`
from commit `aa3a62c`. Plan: `zed_plans/monkeyplug_main_merge_2026-07-29.md`.

## Discoveries
- Backup tag `backup/before-merge-main` was created in a previous session before
  aborting an initial merge attempt. Points to `aa3a62c` as expected.
- Working tree has only untracked files: `agent_planning/`, `openspec/`, `zed_plans/`
  (the artifacts produced during plan authoring). No tracked-file modifications.
- `git fetch origin` returned exit 0 with no new commits (origin was already
  current as of plan authoring).

## Execution Log

### ✅ Task 1: Verify safety net and clean working tree — COMPLETE

- **What was done:** Ran 5 verification commands listed in Task 1 STRUCTURE.
- **Files modified/created:** None modified. Devlog created at
  `agent_planning/execution/monkeyplug_main_merge/devlogs/main_merge_2026-07-29.md`.
- **Verification:**

  | Command | Expected | Actual | Status |
  |---------|----------|--------|--------|
  | `git status` | clean tree | 3 untracked dirs (`agent_planning/`, `openspec/`, `zed_plans/`), no tracked changes | ✅ PASS (untracked files are plan artifacts, expected) |
  | `git tag --list backup/before-merge-main` | one match | `backup/before-merge-main` | ✅ PASS |
  | `git rev-parse backup/before-merge-main` | `aa3a62c...` | `aa3a62ce97716a65ede76672d634fdabfb843647` | ✅ PASS |
  | `git log --oneline -1 backup/before-merge-main` | fix commit message | `aa3a62c fix: preserve contractions in swear detection and add remote polling timeout` | ✅ PASS |
  | `git fetch origin` | exit 0 | exit 0 | ✅ PASS |

- **Deviations from plan:** None.

### ✅ Task 2: Initiate merge with origin/main — COMPLETE

- **What was done:** Ran `git merge origin/main --no-ff`. Merge entered MERGING
  state as expected. HEAD still at `aa3a62c`; MERGE_HEAD = `ec822b2f`.
- **Files modified/created:** None modified (state change only; merge still in
  progress, no commit yet).
- **Verification:**

  | Command | Expected | Actual | Status |
  |---------|----------|--------|--------|
  | `git merge origin/main --no-ff` | 2 unmerged files | 2 unmerged files: README.md, src/monkeyplug/monkeyplug.py | ✅ PASS |
  | `cat .git/MERGE_HEAD` | commit hash ≠ HEAD | `ec822b2f2cfdcc5c3f3fc121c28aedd3a584ec82` | ✅ PASS |
  | `git rev-parse HEAD` | still `aa3a62c...` | `aa3a62ce97716a65ede76672d634fdabfb843647` | ✅ PASS |
  | `git ls-files --unmerged` | 2 paths | 2 paths (each with 3 stages) | ✅ PASS |
  | `git log --oneline aa3a62c..ec822b2f` | 14 commits | 14 commits | ✅ PASS |
  | Conflict markers in `monkeyplug.py` | ~30 | 27 | ✅ PASS (within range) |
  | Conflict markers in `README.md` | ~4 | 4 | ✅ PASS |

- **Auto-merged files (staged, no conflicts):**
  - `.github/workflows/monkeyplug-build-push-vosk-ghcr.yml`
  - `.github/workflows/monkeyplug-build-push-whisper-ghcr.yml`
  - `input/Witch_mother1.m4b` (new file, added by upstream)
  - `pyproject.toml` (modified)
  - `requirements.txt` (deleted by upstream)
  - `setup.cfg` (deleted by upstream)
  - `src/monkeyplug/__init__.py` (modified)
  - `tests/test_swears_loading.py` (new file, added by upstream)
  - `tests/test_transcript_save_reuse.py` (new file, added by upstream)

- **14 incoming commits (`aa3a62c..ec822b2f`):**
  ```
  ec822b2 pin actions
  d2ed509 bump version
  a8b6a1e Merge pull request #17 from stratus-ss/feature/transcript-save-reuse
  614294f Merge branch 'main' into feature/transcript-save-reuse
  9d29ec8 Add transcript save/reuse with automatic detection
  d761fc4 fix words like he'll being mistaken for words like hell
  e730812 reformat with black
  cd46537 Merge pull request #16 from stratus-ss/feature/json-swears-list
  87ca35f feat: add JSON swears list support with auto-detection and tests
  85cca30 output format detection
  909dcc0 update readme
  5306293 add options for specifying other audio parameters (bitrate and vorbis qscale) and fix annoyances with output format determination
  4f46e1a modernize build/installation framework
  6613228 modernize build/installation framework
  ```

- **Deviations from plan:** None. Conflict marker count was 27 in
  `monkeyplug.py` (plan estimated ~30) — actual count is within tolerance, no
  impact on resolution strategy.

### ✅ Task 3: Resolve `scrubword()` and `should_scrub_word()` conflicts — COMPLETE

- **What was done:** Resolved 2 conflict blocks in `src/monkeyplug/monkeyplug.py`:
  1. **Lines 130-135** (`scrubword()` function) — took ours. Our `aa3a62c`
     version handles both `\u2019` (right single quotation mark) and uses
     `SCRUB_PUNCTUATION` (apostrophe preserved). Their `d761fc4` version only
     handles `'` and uses `string.punctuation` which would strip apostrophes
     from contractions like `he'll`, breaking swear detection on those words.
  2. **Lines 634-672** (`_should_scrub_word` + `LoadTranscriptFromFile`) —
     took ours. The theirs side was empty (a no-op conflict where their branch
     didn't add anything at this position). Ours preserves both:
     - `_should_scrub_word()` with confidence bypass logic (production-pinned)
     - `LoadTranscriptFromFile()` using `TranscriptManager.load_transcript()`

- **Files modified/created:** `src/monkeyplug/monkeyplug.py` (resolved 2 of
  27 conflict blocks; 25 remain).
- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `grep -c "<<<<<<< HEAD" monkeyplug.py` | 25 (down from 27) | 25 | ✅ PASS |
  | `scrubword("hell")` | `hell` | `hell` | ✅ PASS |
  | `scrubword("he'll")` | `he'll` (apostrophe preserved) | `he'll` | ✅ PASS |
  | `scrubword("he\u2019ll")` | `he'll` (smart-quote normalized) | `he'll` | ✅ PASS |
  | `scrubword("HELL!")` | `hell` (case + punct stripped) | `hell` | ✅ PASS |
  | `scrubword("don't")` | `don't` (contraction preserved) | `don't` | ✅ PASS |
  | `TranscriptManager` imported in monkeyplug.py | present | line 35 | ✅ PASS |

- **Verified `TranscriptManager` exists** in `src/monkeyplug/utilities.py:454`
  and is already imported at `src/monkeyplug/monkeyplug.py:35`, so our
  `LoadTranscriptFromFile` (using `TranscriptManager.load_transcript`) is
  consistent with the rest of the codebase.

- **Deviations from plan:** None.

- **For Task 4 attention:** Theirs also has a `LoadTranscriptFromFile` inline
  implementation (json.load + manual scrub) in conflict block at lines 521-558.
  Task 4 will take ours there too, dropping the inline duplicate and keeping
  ours' `TranscriptManager`-based version (the only one in the codebase).

### ✅ Task 4: Resolve `_poll_for_transcription_result()` and remaining `monkeyplug.py` conflicts — COMPLETE

- **What was done:** Resolved 25 conflict blocks in `src/monkeyplug/monkeyplug.py`
  across the file. Resolution strategy by block class:
  - **Log style changes (mmguero.eprint → self.logger.info):** Took ours.
    Maintained consistency with the rest of the codebase which uses
    `MonkeyplugLogger` (created at line 397-405).
  - **`reportFormat` vs `forceRetranscribe` parameter conflicts:** Combined both
    sides. Both features are used 12 and 9 times respectively; neither can be
    dropped.
  - **Output format detection (`mmguero.remove_suffix`, `AUDIO_MATCH_FORMAT`):**
    Took theirs. Feature originated in upstream `85cca30` + `5306293`; safe to
    adopt because chunking logic doesn't depend on this code path.
  - **`dbug` → `verbose` parameter bug in upstream's Vosk/Whisper init:** Took
    ours. Upstream's code referenced undefined variable `dbug` (no such
    parameter exists); uses `verbose` (the correct parameter name).
  - **`_poll_for_transcription_result` (production-pinned):** Already intact at
    lines 1232-1270, no conflict markers. Wall-clock timeout + orphan-task
    detection preserved from `aa3a62c`.
  - **`_should_scrub_word` confidence bypass:** Preserved from Task 3.
  - **Whisper/WebUI remote URL handling:** Took ours. Auto-adds `http://` if
    user supplies bare hostname.
  - **Auto-detect transcript reuse:** Took theirs. Uses `forceRetranscribe`
    flag to skip auto-detection.
  - **`LoadTranscriptFromFile` from upstream (inline json.load):** Took ours
    (our `TranscriptManager.load_transcript`-based version from Task 3 is the
    only one in the codebase; dropped duplicate).

- **Files modified/created:** `src/monkeyplug/monkeyplug.py` (0 conflicts
  remaining, file is syntactically valid).
- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `grep -c "<<<<<<< HEAD" monkeyplug.py` | 0 | 0 | ✅ PASS |
  | `python -c "import ast; ast.parse(...)"` | valid | valid | ✅ PASS |
  | `git diff --check` | clean | clean | ✅ PASS (no whitespace issues) |
  | `import monkeyplug` | works | works | ✅ PASS |
  | `monkeyplug.scrubword("he'll")` | `"he'll"` (apostrophe preserved) | `"he'll"` | ✅ PASS |
  | `monkeyplug.scrubword("he\u2019ll")` | `"he'll"` (smart-quote normalized) | `"he'll"` | ✅ PASS |
  | `monkeyplug.scrubword("HELL!")` | `"hell"` | `"hell"` | ✅ PASS |
  | `hasattr(Plugger, '_should_scrub_word')` | True | True | ✅ PASS |
  | `hasattr(WhisperPlugger, '_poll_for_transcription_result')` | True | True | ✅ PASS |
  | `hasattr(Plugger, 'LoadTranscriptFromFile')` | True | True | ✅ PASS |

- **Deviations from plan:** None. The `_poll_for_transcription_result` method
  had no conflict markers (only its parameter list did), so DR-2's "take ours"
  was preserved automatically.

### ✅ Task 5: Resolve `README.md` conflicts — COMPLETE

- **What was done:** Resolved 4 conflict blocks in `README.md`:
  1. **Lines 8-21 (engine list and steps):** Combined ours (3 engines incl.
     Whisper-WebUI + confidence threshold) + theirs (transcript save mention).
  2. **Lines 70-80 (CLI args `-m/--mode`):** Took ours (includes
     `remote-whisper` mode).
  3. **Lines 89-98 (CLI args continued):** Combined both — ours
     (`swears`, `confidence-threshold`) + theirs (`force-retranscribe`).
  4. **Lines 153-162 (Remote Whisper Options section):** Took ours (full
     section with `--remote-whisper-url`, `--remote-whisper-timeout`,
     `--remote-whisper-poll-interval`).

- **Files modified/created:** `README.md` (0 conflicts remaining).
- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `grep -c "<<<<<<< HEAD" README.md` | 0 | 0 | ✅ PASS |
  | Remote Whisper Options section | present | present | ✅ PASS |
  | `remote-whisper` mode in CLI args | present | present | ✅ PASS |
  | `confidence-threshold` in CLI args | present | present | ✅ PASS |
  | `force-retranscribe` in CLI args | present | present | ✅ PASS |

- **Known issue to address in Task 11:** README has duplicate "Transcript
  Workflow" sections (lines 193-225 and 263-292). Both remained because both
  branches added similar sections. Task 11 will deduplicate.

- **Deviations from plan:** DR-4 said "theirs as base with chunking docs
  layered". The actual resolution was closer to "combine both sides" because
  the theirs version omitted the 3rd engine (Whisper-WebUI) and the confidence
  threshold mention. Both are our branch's additions and our chunking docs
  are already in the file (lines 227-238). Functionally equivalent to layering
  chunking docs on theirs' base.

### ✅ Task 6: Run pytest and validate in-tree test suite — COMPLETE

- **What was done:** Ran `python -m pytest tests/`. Initial result: 10 failed,
  38 passed. After fixes: **48 passed, 0 failed**.

- **Fixes applied (merge-caused failures):**

  1. **`dbug=True` → `verbose=True`:** 7 occurrences in
     `tests/test_transcript_save_reuse.py`. Upstream's tests use `dbug` (a typo
     of `debug`); our merged code uses `verbose` (the correct parameter name).
     Sed-replaced all 7 occurrences.

  2. **Missing `confidenceThreshold` attr in MockPlugger:** The MockPlugger in
     `tests/test_transcript_save_reuse.py` lacked `self.confidenceThreshold`.
     Our merged `LoadTranscriptFromFile` uses
     `TranscriptManager.load_transcript(... confidence_threshold=self.confidenceThreshold ...)`.
     Added `self.confidenceThreshold = CONFIDENCE_THRESHOLD_DEFAULT` to
     MockPlugger.__init__ + imported `CONFIDENCE_THRESHOLD_DEFAULT` from
     `monkeyplug.monkeyplug`.

- **Fixes applied (pre-existing failures, user-authorized):**

  3. **`MAX_CHUNK_SIZE_MB = 50` → `150` in `src/monkeyplug/audio_chunker.py`:**
     The README documents "audio chunking for large files (>150MB)" and the
     test `test_exactly_at_threshold` expects 150MB threshold. `aa3a62c`
     changed `MAX_CHUNK_SIZE_MB` from 150 to 50 in production code but never
     updated the README or tests. Reverted to 150MB to match docs and tests.

  4. **Added `_log()` and `_log_section()` methods to `AudioChunker`:**
     Delegate to `self.logger.info()` and `self.logger.log_section()`. Tests
     `test_logs_when_debug_enabled` and `test_no_logs_when_debug_disabled`
     call `chunker._log(...)` directly; the code was using `self.logger.info()`
     internally. Added backward-compatible shim methods.

- **Fixes applied (environmental failures, user-provided remote URL):**

  5. **Integration tests now use remote Whisper at `whisper.x86experts.com:8001`:**
     The 4 `TestTranscriptSaveReuseIntegration` tests instantiate `WhisperPlugger`
     with default local-mode (no `remoteUrl`). User authorized use of
     `whisper.x86experts.com` as remote Whisper service. Added
     `remoteUrl=REMOTE_WHISPER_URL` kwarg to all 7 test integration block
     instantiations (some tests have 2 pluggers). Verified
     `http://whisper.x86experts.com:8001/` returns 307 → FastAPI Swagger UI
     ("Whisper-WebUI-Swear-Removal-Backend").

- **Final test result:**

  | Suite | Tests | Passed | Failed | Time |
  |-------|-------|--------|--------|------|
  | `test_audio_chunker.py` | 34 | 34 | 0 | 2.60s |
  | `test_swears_loading.py` | 4 | 4 | 0 | (incl. above) |
  | `test_transcript_save_reuse.py` | 10 | 10 | 0 | ~22s (integration tests hit remote service) |
  | **Total** | **48** | **48** | **0** | **24.57s** |

- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `pytest exit code` | 0 | 0 | ✅ PASS |
  | `test_exactly_at_threshold` (150MB threshold) | pass | pass | ✅ PASS |
  | `test_logs_when_debug_enabled` (`_log` method) | pass | pass | ✅ PASS |
  | `test_no_logs_when_debug_disabled` (`_log` method) | pass | pass | ✅ PASS |
  | `test_save_transcript_creates_file` (remote whisper) | pass | pass | ✅ PASS |
  | `test_automatic_transcript_reuse` (remote whisper) | pass | pass | ✅ PASS |
  | `test_force_retranscribe_flag` (remote whisper) | pass | pass | ✅ PASS |
  | `test_explicit_transcript_reuse` (remote whisper) | pass | pass | ✅ PASS |
  | `test_different_swear_lists_with_same_transcript` (remote whisper) | pass | pass | ✅ PASS |

- **Deviations from plan:** The plan foresaw pytest (a) failing on merge-caused
  issues and (b) potentially failing on pre-existing issues. The pre-existing
  failures (3 in chunker tests) were not in the plan's expected failures list.
  User authorized fixing them. Rolling forward.

## Files Modified / Created
| File | Purpose / Change |
|------|------------------|
| `src/monkeyplug/monkeyplug.py` | All 25 conflict blocks resolved |
| `README.md` | All 4 conflict blocks resolved |
| `src/monkeyplug/audio_chunker.py` | `MAX_CHUNK_SIZE_MB = 150` (was 50) + added `_log`/`_log_section` methods |
| `tests/test_transcript_save_reuse.py` | 7× `dbug`→`verbose` + MockPlugger gained `confidenceThreshold` + 7× `remoteUrl=REMOTE_WHISPER_URL` added to integration tests |
| `agent_planning/execution/monkeyplug_main_merge/devlogs/main_merge_2026-07-29.md` | This devlog |

### ✅ Task 7: Cross-check `WhisperPlugger` API against `audio_cleaner.py` consumer — COMPLETE

- **What was done:** Read `~/git_projects/OpenAudible-To-AudioBookShelf/src/openaudible_to_audiobookshelf/audio_cleaner.py`
  (consumer). Verified `WhisperPlugger.__init__` signature matches all 22
  consumer parameters; verified public API surface (attributes + methods)
  is backward-compatible.

- **Consumer parameter mapping** (audio_cleaner.py → `WhisperPlugger.__init__`):

  | Consumer (kwarg) | WhisperPlugger param | Default | Status |
  |------------------|----------------------|---------|--------|
  | `iFileSpec` | ✓ | (required) | ✅ OK |
  | `oFileSpec` | ✓ | (required) | ✅ OK |
  | `oAudioFileFormat` | ✓ | (required) | ✅ OK |
  | `iSwearsFileSpec` | ✓ | (required) | ✅ OK |
  | `mDir` | ✓ | (required) | ✅ OK |
  | `mName` | ✓ | (required) | ✅ OK |
  | `torchThreads` | ✓ | 0 | ✅ OK |
  | `outputJson` | ✓ | None | ✅ OK |
  | `reportFormat` | ✓ | "txt" | ✅ OK |
  | `inputTranscript` | ✓ | None | ✅ OK |
  | `saveTranscript` | ✓ | False | ✅ OK |
  | `remoteUrl` | ✓ | None | ✅ OK |
  | `apiTimeout` | ✓ | 600 | ✅ OK |
  | `pollInterval` | ✓ | 5 | ✅ OK |
  | `confidenceThreshold` | ✓ | 0.65 | ✅ OK |
  | `beep` | ✓ | False | ✅ OK |
  | `force` | ✓ | False | ✅ OK |
  | `useChunking` | ✓ | False | ✅ OK |
  | `chunkingWorkDir` | ✓ | None | ✅ OK |
  | `parallelEncoding` | ✓ | False | ✅ OK |
  | `maxWorkers` | ✓ | None | ✅ OK |
  | `verbose` | ✓ | False | ✅ OK |

- **Public API surface used by consumer:**

  | Member | Purpose | Status |
  |--------|---------|--------|
  | `WhisperPlugger.__init__(...)` | Constructor | ✅ OK |
  | `plugger.EncodeCleanAudio()` | Main entry point | ✅ OK |
  | `plugger.wordList` | List of {word, start, end, scrub} dicts | ✅ OK |
  | `plugger.naughtyWordList` | Filtered list of scrubbed words | ✅ OK |
  | `plugger.remote_url` | Resolved remote URL (auto-prefixed http://) | ✅ OK |
  | `plugger.debug` | Verbose flag (set via `verbose=`) | ✅ OK |
  | `plugger.confidenceThreshold` | Set by base `Plugger.__init__` | ✅ OK |

- **Module-level imports consumer uses:**

  | Import | Status |
  |--------|--------|
  | `from monkeyplug.monkeyplug import WhisperPlugger` | ✅ OK |

- **Behavior cross-check:**

  | Behavior | Consumer expectation | Code behavior | Status |
  |----------|---------------------|---------------|--------|
  | URL auto-prefix `http://` when missing | Lines 394-414: probes raw URL | Lines 925-931: auto-prefixes if no scheme | ✅ OK |
  | Wall-clock timeout on transcription poll | (Production-pinned behavior in `aa3a62c`) | Lines 1077-1126: enforces `api_timeout` | ✅ OK |
  | `wordList` populated after `EncodeCleanAudio()` | Used in `_log_completion` | Set in `RecognizeSpeech()` + `_aggregate_transcripts` | ✅ OK |
  | `reportFormat` controls censor report | `reportFormat="json" if debug else "txt"` | Plugger internal uses `self.reportFormat` | ✅ OK |
  | `useChunking` for files >150MB | `CHUNKING_THRESHOLD_MB = 150` | `MAX_CHUNK_SIZE_MB = 150` (matches) | ✅ OK |

- **Files modified/created:** None (read-only cross-check).

- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `inspect.signature(WhisperPlugger.__init__)` | all 22 params | all 22 params | ✅ PASS |
  | `hasattr(WhisperPlugger, 'EncodeCleanAudio')` | True | True | ✅ PASS |
  | Default `apiTimeout` | 600 | 600 | ✅ PASS |
  | Default `pollInterval` | 5 | 5 | ✅ PASS |
  | `from monkeyplug.monkeyplug import WhisperPlugger` | works | works | ✅ PASS |

- **Cross-check verdict:** **BACKWARD COMPATIBLE.** No source changes
  required to `audio_cleaner.py`. The consumer can be deployed against the
  merged `feature/file-chunking` branch without modifications.

- **Deviations from plan:** None.

## Files Modified / Created
| File | Purpose / Change |
|------|------------------|
| `src/monkeyplug/monkeyplug.py` | Merged (no consumer-related changes) |
| `src/monkeyplug/audio_chunker.py` | `MAX_CHUNK_SIZE_MB = 150` (matches consumer `CHUNKING_THRESHOLD_MB`) |
| `agent_planning/execution/monkeyplug_main_merge/devlogs/main_merge_2026-07-29.md` | This devlog |

### ✅ Task 8: Commit the merge — COMPLETE

- **What was done:** Staged all resolved files + pre-existing fixes.
  Did not stage untracked plan artifacts (`agent_planning/`, `openspec/`,
  `zed_plans/`). Created merge commit with detailed resolution policy.

- **Files staged:**
  - `README.md` (resolved 4 conflict blocks)
  - `src/monkeyplug/monkeyplug.py` (resolved 25 conflict blocks)
  - `src/monkeyplug/audio_chunker.py` (MAX_CHUNK_SIZE_MB 50→150 + _log/_log_section methods)
  - `tests/test_transcript_save_reuse.py` (dbug→verbose × 7, MockPlugger confidenceThreshold, remoteUrl × 7)
  - Plus 9 auto-merged files from `git merge`

- **Commit message:** Documents full resolution policy:
  - Took ours (production-pinned): scrubword, _should_scrub_word,
    _poll_for_transcription_result, LoadTranscriptFromFile,
    WhisperPlugger dbug→verbose bug fix, logger.info style, URL auto-prefix,
    reportFormat
  - Took theirs (upstream features): m4b/bitrate, mmguero.remove_suffix,
    AUDIO_MATCH_FORMAT, auto-detect transcript reuse, forceRetranscribe
  - Took both: parameters consumed by both sides
  - Dropped: d761fc4 scrunchword bug (string.punctuation strips apostrophes)
  - Pre-existing fixes: MAX_CHUNK_SIZE_MB revert, _log methods, test param renames

- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `git log --oneline -1` | merge commit | `c134df1 Merge origin/main into feature/file-chunking` | ✅ PASS |
  | `git rev-parse HEAD` | new SHA | `c134df1474fbc64873157f33c29f3264f92dadba` | ✅ PASS |
  | `git rev-parse HEAD^1` | `aa3a62c` | `aa3a62ce97716a65ede76672d634fdabfb843647` | ✅ PASS |
  | `git rev-parse HEAD^2` | `ec822b2f` | `ec822b2f2cfdcc5c3f3fc121c28aedd3a584ec82` | ✅ PASS |
  | `git log --merges -1` | merge commit | merge commit | ✅ PASS |
  | `a59b820` reachable | yes | yes | ✅ PASS |
  | diff stat | 12 files changed | 12 files changed, 829 +/107 - | ✅ PASS |

- **Deviations from plan:** None.

## Files Modified / Created
| File | Purpose / Change |
|------|------------------|
| Merge commit `c134df1` | 12 files changed, 829 insertions(+), 107 deletions(-) |
| `agent_planning/execution/monkeyplug_main_merge/devlogs/main_merge_2026-07-29.md` | This devlog |

### ✅ Task 9: Push to origin — COMPLETE

- **What was done:** Ran `git push origin feature/file-chunking`. Result:
  `aa3a62c..c134df1  feature/file-chunking -> feature/file-chunking`.

- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `git push` exit code | 0 | 0 | ✅ PASS |
  | `git log --decorate -1` | HEAD + origin point at c134df1 | `c134df1 (HEAD -> feature/file-chunking, origin/feature/file-chunking)` | ✅ PASS |
  | Working tree | clean (only untracked plan artifacts) | clean | ✅ PASS |
  | Push advance | `aa3a62c..c134df1` | `aa3a62c..c134df1` (+1 commit, 12 files) | ✅ PASS |

- **Deviations from plan:** None.

### ✅ Task 10: Update `openspec/specs/core/spec.md` with post-merge behaviors — COMPLETE

- **What was done:** Updated `openspec/specs/core/spec.md` (untracked, plan
  artifact, not committed per PLAN_CORE §9). Added 7 new requirements to
  cover post-merge behaviors; updated 1 existing requirement (chunking
  threshold).

- **Requirements added (7 new):**

  1. **VoskPlugger and WhisperPlugger __init__ use verbose parameter** —
     Documents the dbug→verbose bug fix and ensures callers don't use the
     upstream's `dbug=` keyword.
  2. **WhisperPlugger auto-prefixes http:// on remote URL when scheme missing**
     — Documents the URL normalization in `__init__`.
  3. **Plugger loads swears list from JSON or text format with auto-detection**
     — Documents the JSON swears list support (upstream `87ca35f`).
  4. **WhisperPlugger saves transcript JSON when saveTranscript is enabled**
     — Documents the `TranscriptManager.save_transcript` call in
     `RecognizeSpeech`.
  5. **Plugger auto-reuses existing transcript unless forceRetranscribe is set**
     — Documents the auto-reuse feature (upstream `9d29ec8`) and
     `forceRetranscribe` override.
  6. **Output filename uses mmguero.remove_suffix to avoid extension duplication**
     — Documents the upstream output-format-detection feature (`85cca30`,
     `5306293`).
  7. **m4b audio format is supported via AAC codec** — Documents the m4b
     entry in `AUDIO_DEFAULT_PARAMS_BY_FORMAT`.

- **Requirements updated (1):**

  - **AudioChunker splits large input files for backend size limits:**
    Updated `MAX_CHUNK_SIZE_BYTES` from "50 MB" to "150 MB" (matches
    pre-existing fix). Added "file exactly at threshold is not chunked"
    scenario.

- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | Spec line count | > 150 | 305 | ✅ PASS |
  | Requirements count | 12 (5 original + 7 new) | 12 | ✅ PASS |
  | Each requirement has scenarios | yes | yes | ✅ PASS |
  | GIVEN/WHEN/THEN format | yes | yes | ✅ PASS |

- **Not committed** (per PLAN_CORE §9): the spec file is in `openspec/`
  which is an untracked plan artifact. The updated spec is available for
  the user to review and commit separately if desired.

- **Deviations from plan:** None.

### ✅ Task 11: Document adaptation notes in README (dedupe Transcript Workflow sections) — COMPLETE

- **What was done:** Removed duplicate "Transcript Workflow" section in
  README. The post-merge file had two near-identical sections (ours at
  lines 165-204 and upstream's at lines 242-271). Kept ours (more
  comprehensive) and added upstream's unique content:
  - "~22x faster than re-transcribing" note in rationale bullets
  - New "Automatic Transcript Reuse" subsection explaining
    `--save-transcript` auto-detection
  - `--force-retranscribe` flag for forcing fresh transcription

- **Files modified/created:**
  - `README.md`: 17 insertions, 32 deletions (-15 lines, 283 → 268)

- **Verification:**

  | Test | Expected | Actual | Status |
  |------|----------|--------|--------|
  | `grep -c "<<<<<<< HEAD" README.md` | 0 | 0 | ✅ PASS |
  | `grep "Transcript Workflow" README.md` | 1 header | 1 header (line 172) | ✅ PASS |
  | `grep "Automatic Transcript Reuse"` | present | present (line 191) | ✅ PASS |
  | `grep --force-retranscribe` | present | present (line 197) | ✅ PASS |
  | `git diff --stat README.md` | -15 net lines | 17 +/32 - = -15 | ✅ PASS |
  | `pytest tests/` (post-commit) | 48 passed | 48 passed in 23.64s | ✅ PASS |

- **Commit:** `2d40cc0 docs: dedupe Transcript Workflow sections in README`

- **Pushed:** `c134df1..2d40cc0  feature/file-chunking -> feature/file-chunking`

- **Deviations from plan:** None.

## Issues Encountered
None.

## Files Modified / Created
| File | Purpose / Change |
|------|------------------|
| `src/monkeyplug/monkeyplug.py` | Resolved 2 conflict blocks (scrubword + _should_scrub_word/LoadTranscriptFromFile) |
| `agent_planning/execution/monkeyplug_main_merge/devlogs/main_merge_2026-07-29.md` | This devlog |

## Remaining Tasks
- [ ] Task 6: Run pytest and validate in-tree test suite
 (dedupe Transcript Workflow sections)
