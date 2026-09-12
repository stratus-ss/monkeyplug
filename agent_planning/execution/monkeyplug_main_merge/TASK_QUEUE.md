# Task Queue — monkeyplug_main_merge

| ID | Task | Status | Notes |
|----|------|--------|-------|
| 1  | Verify safety net and clean working tree | done | Tag at aa3a62c verified, working tree clean (only untracked plan artifacts) |
| 2  | Initiate merge with origin/main | done | MERGING state: 2 unmerged files (monkeyplug.py=27 markers, README.md=4 markers), 14 commits incoming |
| 3  | Resolve `scrubword()` and `should_scrub_word()` conflicts | done | 25 markers remain. scrubword: takes ours (handles \u2019 + preserves apostrophes). _should_scrub_word + LoadTranscriptFromFile: takes ours (TranscriptManager-based) |
| 4  | Resolve `_poll_for_transcription_result()` and remaining `monkeyplug.py` conflicts | done | 0 markers remain. _poll_for_transcription_result intact (no conflict markers). Combined ours + theirs for parameter additions. Took ours for dbug→verbose bug fix. |
| 5  | Resolve `README.md` conflicts | done | 0 markers remain. README known issue: duplicate "Transcript Workflow" sections (Task 11) |
| 6  | Run pytest and validate in-tree test suite | done | 48 passed, 0 failed. Fixed: dbug→verbose, MockPlugger confidenceThreshold, MAX_CHUNK_SIZE_MB=150, _log/_log_section methods, remoteUrl on integration tests |
| 7  | Cross-check `WhisperPlugger` API against `audio_cleaner.py` consumer | done | BACKWARD COMPATIBLE. All 22 consumer params match. URL auto-prefix behavior preserved. Wall-clock timeout active (production-pinned). |
| 8  | Commit the merge | done | Merge commit c134df1. Both parents preserved (aa3a62c + ec822b2f). a59b820 reachable. |
| 9  | Push to origin | done | `aa3a62c..c134df1  feature/file-chunking -> feature/file-chunking`. Remote confirmed. |
| 10 | Update `openspec/specs/core/spec.md` with post-merge behaviors | done | Added 7 new requirements (verbose fix, URL auto-prefix, JSON swears, transcript save, transcript reuse, mmguero.remove_suffix, m4b format). Updated AudioChunker threshold to 150MB. Not committed (plan artifact). |
| 11 | Document adaptation notes in README (dedupe Transcript Workflow sections) | done | Commit 2d40cc0. README 17+/32- = -15 lines. Pushed c134df1..2d40cc0. |
