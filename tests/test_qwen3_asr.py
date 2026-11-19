"""Tests for the qwen3-asr remote backend.

Covers the job API contract (POST /v1/jobs, GET /v1/jobs/{id}), word mapping,
and the failure modes that must not silently produce uncensored audio.
No network: `requests` in the monkeyplug module is replaced with a stub.
"""
from unittest.mock import Mock

import pytest

import monkeyplug.monkeyplug as mp
from monkeyplug.monkeyplug import WhisperPlugger


def _plugger(**attrs):
    """WhisperPlugger without __init__ (no audio file, no model, no ffmpeg)."""
    p = object.__new__(WhisperPlugger)
    p.wordList = []
    p.debug = False
    p.remote_url = "http://qwen3:8081"
    p.remote_backend = mp.REMOTE_BACKEND_QWEN3_ASR
    p.remoteParams = {}
    p.api_timeout = 5
    p.poll_interval = 0
    p.inputFileSpec = __file__  # any readable file; the stub never sends it
    p.swearsMap = {"damn": True}
    p.confidenceThreshold = 0.65
    p.logger = Mock()
    for k, v in attrs.items():
        setattr(p, k, v)
    return p


def _response(payload, status_code=200):
    r = Mock()
    r.status_code = status_code
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def test_recognize_remote_dispatches_to_qwen3(monkeypatch):
    p = _plugger()
    p._RecognizeSpeechQwen3Asr = Mock()
    p._upload_audio_for_transcription = Mock()
    WhisperPlugger._RecognizeSpeechRemote(p)
    p._RecognizeSpeechQwen3Asr.assert_called_once_with()
    p._upload_audio_for_transcription.assert_not_called()


def test_upload_posts_to_jobs_and_sends_language(monkeypatch):
    p = _plugger(remoteParams={"lang": "en"})
    post = Mock(return_value=_response({"job_id": "abc", "status": "queued"}))
    monkeypatch.setattr(mp.requests, "post", post)

    assert p._upload_audio_for_qwen3_job() == "abc"

    args, kwargs = post.call_args
    assert args[0] == "http://qwen3:8081/v1/jobs"
    assert kwargs["data"] == {"language": "en"}
    assert "timeout" in kwargs


def test_upload_without_language_sends_no_form_field(monkeypatch):
    p = _plugger()
    post = Mock(return_value=_response({"job_id": "abc"}))
    monkeypatch.setattr(mp.requests, "post", post)
    p._upload_audio_for_qwen3_job()
    assert post.call_args[1]["data"] == {}


def test_poll_returns_words_on_done(monkeypatch):
    p = _plugger()
    get = Mock(side_effect=[
        _response({"status": "transcribing", "done": 1, "total": 2}),
        _response({
            "status": "done",
            "words": [{"word": "Hello", "start": 0.0, "end": 0.4}],
        }),
    ])
    monkeypatch.setattr(mp.requests, "get", get)

    words = p._poll_for_qwen3_job("abc")
    assert words == [{"word": "Hello", "start": 0.0, "end": 0.4}]
    assert get.call_args[0][0] == "http://qwen3:8081/v1/jobs/abc"


def test_poll_reports_404_as_restart(monkeypatch):
    p = _plugger()
    monkeypatch.setattr(mp.requests, "get", Mock(return_value=_response({}, 404)))
    with pytest.raises(RuntimeError, match="restarted"):
        p._poll_for_qwen3_job("abc")


def test_poll_raises_on_error_status(monkeypatch):
    p = _plugger()
    monkeypatch.setattr(
        mp.requests, "get",
        Mock(return_value=_response({"status": "error", "error": "no speech"})),
    )
    with pytest.raises(RuntimeError, match="no speech"):
        p._poll_for_qwen3_job("abc")


def test_poll_refuses_empty_word_list(monkeypatch):
    p = _plugger()
    monkeypatch.setattr(
        mp.requests, "get",
        Mock(return_value=_response({"status": "done", "words": []})),
    )
    with pytest.raises(RuntimeError, match="no word timings"):
        p._poll_for_qwen3_job("abc")


def test_extract_maps_and_flags_swears():
    p = _plugger()
    p._extract_words_from_qwen3_result([
        {"word": "Damn,", "start": 1.0, "end": 1.5},
        {"word": " fine ", "start": 1.5, "end": 2.0},
        {"word": "", "start": 2.0, "end": 2.1},
    ])
    assert [w["word"] for w in p.wordList] == ["Damn,", "fine"]
    assert p.wordList[0]["scrub"] is True
    assert p.wordList[0]["conf"] == 1.0
    assert p.wordList[1]["scrub"] is False
