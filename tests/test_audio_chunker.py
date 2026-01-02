"""
Functional tests for AudioChunker module.

This test suite validates core functionality:
- File size-based chunking decisions
- Audio splitting at silence points with chunk reuse
- Split point selection algorithm
- Chapter metadata extraction and restoration
- Chunk concatenation with error handling
- Serial vs parallel processing modes
- Plugger state management
- Full workflow integration
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import json

from monkeyplug.audio_chunker import AudioChunker, AudioChunkingError


class MockPlugger:
    """Mock plugger for testing."""
    
    def __init__(self):
        self.inputFileSpec = "/path/to/input.m4a"
        self.outputJson = "/path/to/output.json"
        self.inputTranscript = None
        self.wordList = []
        self.swearsMap = {"profanity": True}
        self.confidenceThreshold = 0.65
        self.debug = False
        self.muteTimeList = []
        self.sineTimeList = []
        self.beepDelayList = []
        self.beep = False
        self.beepMixNormalize = False
        self.beepAudioWeight = 1
        self.beepSineWeight = 1
        self.beepDropTransition = 0
        self.aParams = ["-c:a", "aac", "-b:a", "128K"]
    
    def CreateCleanMuteList(self):
        """Mock method that would normally transcribe and build mute list."""
        self.wordList = [
            {"word": "test", "start": 0.0, "end": 1.0, "conf": 1.0, "scrub": False},
            {"word": "profanity", "start": 1.5, "end": 2.0, "conf": 0.9, "scrub": True}
        ]
        self.muteTimeList = ["volume=enable='between(t,1.500,2.000)':volume=0"]
        return self.muteTimeList
    
    def RecognizeSpeech(self):
        """Mock speech recognition."""
        self.wordList = [
            {"word": "hello", "start": 0.0, "end": 0.5, "conf": 1.0, "scrub": False}
        ]
    
    def LoadTranscriptFromFile(self):
        """Mock transcript loading."""
        self.wordList = [
            {"word": "loaded", "start": 0.0, "end": 0.5, "conf": 1.0, "scrub": False}
        ]


class TestNeedsChunking:
    """Test file size-based chunking decisions."""
    
    def test_large_file_needs_chunking(self):
        """File larger than 150MB should require chunking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            large_file = os.path.join(tmpdir, "large.m4a")
            with open(large_file, 'wb') as f:
                # Write 200MB
                f.write(b'x' * (200 * 1024 * 1024))
            
            assert chunker.needs_chunking(large_file) is True
    
    def test_small_file_no_chunking(self):
        """File smaller than 150MB should not require chunking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            small_file = os.path.join(tmpdir, "small.m4a")
            with open(small_file, 'wb') as f:
                # Write 50MB
                f.write(b'x' * (50 * 1024 * 1024))
            
            assert chunker.needs_chunking(small_file) is False
    
    def test_exactly_at_threshold(self):
        """File exactly at 150MB threshold should not require chunking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            threshold_file = os.path.join(tmpdir, "threshold.m4a")
            with open(threshold_file, 'wb') as f:
                # Write exactly 150MB
                f.write(b'x' * (150 * 1024 * 1024))
            
            assert chunker.needs_chunking(threshold_file) is False


class TestSelectSplitPoints:
    """Test split point selection algorithm."""
    
    def test_selects_closest_silence_within_tolerance(self):
        """Should select silence points closest to target durations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            # Total 3600s, target 1800s chunks
            duration = 3600.0
            target_duration = 1800.0
            
            # Silence at good points: 1805s and 3595s
            silence_points = [1805.0, 3595.0]
            
            result = chunker._select_split_points(silence_points, duration, target_duration)
            
            # Should select 1805 (within tolerance of 1800)
            assert 1805.0 in result
    
    def test_falls_back_to_time_based_when_no_silence(self):
        """Should use time-based splits when no silence found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            duration = 3600.0
            target_duration = 1800.0
            silence_points = []
            
            result = chunker._select_split_points(silence_points, duration, target_duration)
            
            # Should have time-based splits
            assert len(result) > 0
            assert 1800.0 in result
    
    def test_respects_tolerance_setting(self):
        """Should only select silence within tolerance threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            duration = 3600.0
            target_duration = 1800.0
            
            # Silence too far from target (1800 +/- 360)
            silence_points = [1000.0, 2500.0]
            
            result = chunker._select_split_points(silence_points, duration, target_duration)
            
            # Should fall back to time-based since silence is outside tolerance
            assert 1800.0 in result or any(1500 <= p <= 2100 for p in result)
    
    def test_handles_short_duration(self):
        """Should handle audio shorter than target chunk duration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            duration = 600.0  # Only 10 minutes
            target_duration = 1800.0  # Target 30 minutes
            silence_points = [300.0]
            
            result = chunker._select_split_points(silence_points, duration, target_duration)
            
            # Should return empty or minimal splits
            assert len(result) == 0 or result[0] < duration


class TestSplitAudioAtSilence:
    """Test audio splitting with chunk reuse."""
    
    @patch('monkeyplug.utilities.FFmpegRunner.get_audio_duration')
    @patch('monkeyplug.utilities.FFmpegRunner.detect_silence_points')
    @patch('subprocess.run')
    def test_reuses_existing_chunks(self, mock_run, mock_detect, mock_duration):
        """Should reuse existing chunks instead of re-splitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            plugger.debug = True
            chunker = AudioChunker(tmpdir, plugger)
            
            # Create source file
            source_file = os.path.join(tmpdir, "source.m4a")
            with open(source_file, 'wb') as f:
                f.write(b'x' * (200 * 1024 * 1024))
            
            # Pre-create chunks
            file_id = "TEST123"
            chunk_dir = Path(tmpdir) / file_id / "chunks"
            chunk_dir.mkdir(parents=True)
            
            chunk_files = []
            for i in range(2):
                chunk_file = chunk_dir / f"source_chunk_{i:03d}.m4a"
                chunk_file.write_bytes(b'chunk data')
                chunk_files.append(str(chunk_file))
            
            mock_duration.return_value = 3600.0
            mock_detect.return_value = [1800.0]
            
            result = chunker._split_audio_at_silence(source_file, file_id)
            
            # Should return existing chunks
            assert len(result) == 2
            assert all(os.path.exists(f) for f in result)
            
            # Should NOT call ffmpeg split
            mock_run.assert_not_called()
    
    @patch('monkeyplug.utilities.FFmpegRunner.get_audio_duration')
    @patch('monkeyplug.utilities.FFmpegRunner.detect_silence_points')
    @patch('subprocess.run')
    def test_creates_new_chunks_when_none_exist(self, mock_run, mock_detect, mock_duration):
        """Should create new chunks when none exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            source_file = os.path.join(tmpdir, "source.m4a")
            with open(source_file, 'wb') as f:
                f.write(b'x' * (200 * 1024 * 1024))
            
            file_id = "TEST123"
            
            mock_duration.return_value = 3600.0
            mock_detect.return_value = [1800.0]
            
            # Mock ffmpeg split to create chunks
            def create_chunks(*args, **kwargs):
                chunk_dir = Path(tmpdir) / file_id / "chunks"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                for i in range(2):
                    chunk_file = chunk_dir / f"source_chunk_{i:03d}.m4a"
                    chunk_file.write_bytes(b'new chunk')
                return MagicMock(returncode=0)
            
            mock_run.side_effect = create_chunks
            
            result = chunker._split_audio_at_silence(source_file, file_id)
            
            # Should create new chunks
            assert len(result) == 2
            assert mock_run.called
    
    @patch('monkeyplug.utilities.FFmpegRunner.get_audio_duration')
    @patch('monkeyplug.utilities.FFmpegRunner.detect_silence_points')
    def test_handles_no_silence_found(self, mock_detect, mock_duration):
        """Should fall back to time-based splits when no silence found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            source_file = os.path.join(tmpdir, "source.m4a")
            with open(source_file, 'wb') as f:
                f.write(b'x' * (200 * 1024 * 1024))
            
            mock_duration.return_value = 3600.0
            mock_detect.return_value = []  # No silence
            
            with patch('subprocess.run') as mock_run:
                def create_chunks(*args, **kwargs):
                    chunk_dir = Path(tmpdir) / "TEST123" / "chunks"
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    chunk_file = chunk_dir / "source_chunk_000.m4a"
                    chunk_file.write_bytes(b'chunk')
                    return MagicMock(returncode=0)
                
                mock_run.side_effect = create_chunks
                
                result = chunker._split_audio_at_silence(source_file, "TEST123")
                
                # Should still create at least one chunk
                assert len(result) >= 1


class TestExtractChapters:
    """Test chapter metadata extraction."""
    
    @patch('subprocess.run')
    def test_extracts_chapters_successfully(self, mock_run):
        """Should extract chapters using ffmpeg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            source_file = "/path/to/source.m4b"
            file_id = "TEST123"
            
            mock_run.return_value = MagicMock(returncode=0)
            
            result = chunker._extract_chapters(source_file, file_id)
            
            assert result is not None
            assert result.name == "chapters.txt"
            mock_run.assert_called_once()
            
            # Verify ffmpeg command structure
            call_args = mock_run.call_args[0][0]
            assert 'ffmpeg' in call_args
            assert source_file in call_args
            assert 'ffmetadata' in call_args
    
    @patch('subprocess.run')
    def test_handles_chapter_extraction_failure(self, mock_run):
        """Should return None when chapter extraction fails."""
        import subprocess
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ffmpeg", stderr=b"Chapter extraction failed"
            )
            
            result = chunker._extract_chapters("/path/to/source.m4b", "TEST123")
            
            assert result is None


class TestConcatenateChunks:
    """Test chunk concatenation functionality."""
    
    @patch('subprocess.run')
    def test_concatenates_without_chapters(self, mock_run):
        """Should concatenate chunks without chapter restoration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            # Create file_id directory structure
            file_id = "TEST123"
            (Path(tmpdir) / file_id).mkdir(parents=True, exist_ok=True)
            
            # Create chunk files
            chunk_files = []
            for i in range(3):
                chunk_file = os.path.join(tmpdir, f"chunk_{i}.m4a")
                with open(chunk_file, 'wb') as f:
                    f.write(b'x' * (50 * 1024 * 1024))
                chunk_files.append(chunk_file)
            
            output_file = os.path.join(tmpdir, "output.m4a")
            
            # Mock ffmpeg to create output
            def create_output(*args, **kwargs):
                with open(output_file, 'wb') as f:
                    f.write(b'x' * (140 * 1024 * 1024))
                return MagicMock(returncode=0, stderr='')
            
            mock_run.side_effect = create_output
            
            chunker._concatenate_chunks(chunk_files, output_file, file_id, None)
            
            assert os.path.exists(output_file)
            assert mock_run.call_count == 1  # Only concat, no chapter restore
    
    @patch('subprocess.run')
    def test_concatenates_with_chapter_restoration(self, mock_run):
        """Should concatenate and restore chapters with correct ffmpeg commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            # Create file_id directory structure
            file_id = "TEST123"
            (Path(tmpdir) / file_id).mkdir(parents=True, exist_ok=True)
            
            # Create chunk files
            chunk_files = []
            for i in range(2):
                chunk_file = os.path.join(tmpdir, f"chunk_{i}.m4a")
                with open(chunk_file, 'wb') as f:
                    f.write(b'x' * (50 * 1024 * 1024))
                chunk_files.append(chunk_file)
            
            # Create chapter file
            chapter_file = Path(tmpdir) / "chapters.txt"
            chapter_file.write_text("[CHAPTER]\nTIMEBASE=1/1000\n")
            
            output_file = os.path.join(tmpdir, "output.m4a")
            
            # Mock ffmpeg calls and capture commands
            ffmpeg_calls = []
            def mock_ffmpeg(cmd, *args, **kwargs):
                ffmpeg_calls.append(cmd)
                
                if len(ffmpeg_calls) == 1:
                    # First call: concat - create temp file
                    temp_file = os.path.join(tmpdir, "output.temp.m4a")
                    with open(temp_file, 'wb') as f:
                        f.write(b'x' * (90 * 1024 * 1024))
                elif len(ffmpeg_calls) == 2:
                    # Second call: chapter restore - create final output
                    with open(output_file, 'wb') as f:
                        f.write(b'x' * (90 * 1024 * 1024))
                
                return MagicMock(returncode=0, stderr='')
            
            mock_run.side_effect = mock_ffmpeg
            
            chunker._concatenate_chunks(chunk_files, output_file, file_id, chapter_file)
            
            # Verify output file was created
            assert os.path.exists(output_file)
            
            # Verify two ffmpeg calls were made
            assert len(ffmpeg_calls) == 2
            
            # Verify first call is concatenation (creates temp file)
            concat_cmd = ffmpeg_calls[0]
            assert 'ffmpeg' in concat_cmd
            assert '-f' in concat_cmd and 'concat' in concat_cmd
            assert 'output.temp.m4a' in ' '.join(concat_cmd)
            
            # Verify second call is chapter restoration
            restore_cmd = ffmpeg_calls[1]
            assert 'ffmpeg' in restore_cmd
            assert str(chapter_file) in restore_cmd  # Chapter file is input
            assert '-map_metadata' in restore_cmd  # Metadata mapping
            assert output_file in restore_cmd  # Final output
    
    def test_raises_error_on_missing_chunks(self):
        """Should raise error if chunk files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            chunk_files = ["/nonexistent/chunk_0.m4a", "/nonexistent/chunk_1.m4a"]
            output_file = os.path.join(tmpdir, "output.m4a")
            
            with pytest.raises(AudioChunkingError) as exc_info:
                chunker._concatenate_chunks(chunk_files, output_file, "TEST123", None)
            
            assert "missing" in str(exc_info.value).lower()
    
    @patch('subprocess.run')
    def test_handles_concatenation_failure(self, mock_run):
        """Should raise AudioChunkingError when ffmpeg fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            # Create file_id directory structure
            file_id = "TEST123"
            (Path(tmpdir) / file_id).mkdir(parents=True, exist_ok=True)
            
            # Create chunk files
            chunk_files = []
            for i in range(2):
                chunk_file = os.path.join(tmpdir, f"chunk_{i}.m4a")
                with open(chunk_file, 'wb') as f:
                    f.write(b'chunk')
                chunk_files.append(chunk_file)
            
            output_file = os.path.join(tmpdir, "output.m4a")
            
            mock_run.return_value = MagicMock(returncode=1, stderr='FFmpeg error')
            
            with pytest.raises(AudioChunkingError) as exc_info:
                chunker._concatenate_chunks(chunk_files, output_file, file_id, None)
            
            assert "concatenation failed" in str(exc_info.value).lower()


class TestPluggerStateManagement:
    """Test plugger state save/restore context manager."""
    
    def test_restores_state_after_normal_execution(self):
        """Should restore plugger state after normal context exit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            plugger.inputFileSpec = "original_input.m4a"
            plugger.wordList = ["original", "words"]
            
            chunker = AudioChunker(tmpdir, plugger)
            
            with chunker._plugger_temp_state():
                # Modify state inside context
                plugger.inputFileSpec = "modified_input.m4a"
                plugger.wordList = ["modified", "words"]
            
            # State should be restored
            assert plugger.inputFileSpec == "original_input.m4a"
            assert plugger.wordList == ["original", "words"]
    
    def test_restores_state_after_exception(self):
        """Should restore plugger state even if exception occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            plugger.inputFileSpec = "original_input.m4a"
            plugger.outputJson = "original_output.json"
            
            chunker = AudioChunker(tmpdir, plugger)
            
            try:
                with chunker._plugger_temp_state():
                    plugger.inputFileSpec = "modified_input.m4a"
                    plugger.outputJson = "modified_output.json"
                    raise ValueError("Test exception")
            except ValueError:
                pass
            
            # State should still be restored
            assert plugger.inputFileSpec == "original_input.m4a"
            assert plugger.outputJson == "original_output.json"
    
    def test_saves_all_plugger_attributes(self):
        """Should save and restore all relevant plugger attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            plugger.inputFileSpec = "input.m4a"
            plugger.outputJson = "output.json"
            plugger.inputTranscript = "transcript.json"
            plugger.wordList = ["word1", "word2"]
            
            chunker = AudioChunker(tmpdir, plugger)
            
            with chunker._plugger_temp_state():
                plugger.inputFileSpec = "new_input.m4a"
                plugger.outputJson = "new_output.json"
                plugger.inputTranscript = "new_transcript.json"
                plugger.wordList = []
            
            assert plugger.inputFileSpec == "input.m4a"
            assert plugger.outputJson == "output.json"
            assert plugger.inputTranscript == "transcript.json"
            assert plugger.wordList == ["word1", "word2"]


class TestProcessChunkSerial:
    """Test serial chunk processing."""
    
    @patch('monkeyplug.utilities.FFmpegRunner.run_encode')
    @patch('monkeyplug.utilities.TranscriptManager.save_transcript')
    def test_processes_chunk_with_transcription(self, mock_save, mock_encode):
        """Should transcribe and encode a single chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            original_wordlist = plugger.wordList.copy()
            chunker = AudioChunker(tmpdir, plugger)
            
            chunk_file = os.path.join(tmpdir, "chunk_000.m4a")
            with open(chunk_file, 'wb') as f:
                f.write(b'chunk data')
            
            output_chunk = os.path.join(tmpdir, "chunk_000_cleaned.m4a")
            mock_encode.return_value = output_chunk
            
            with open(output_chunk, 'wb') as f:
                f.write(b'encoded data')
            
            result = chunker._process_chunk(chunk_file, 1)
            
            assert result == output_chunk
            # State should be restored after _process_chunk
            assert plugger.wordList == original_wordlist
            mock_encode.assert_called_once()
    
    @patch('monkeyplug.utilities.FFmpegRunner.run_encode')
    @patch('monkeyplug.utilities.TranscriptManager.load_or_create_transcript')
    def test_reuses_existing_transcript(self, mock_load, mock_encode):
        """Should reuse existing transcript when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            chunk_file = os.path.join(tmpdir, "chunk_000.m4a")
            with open(chunk_file, 'wb') as f:
                f.write(b'chunk data')
            
            # Create existing transcript
            transcript_file = os.path.join(tmpdir, "chunk_000_transcript.json")
            with open(transcript_file, 'w') as f:
                json.dump([{"word": "test", "start": 0.0, "end": 1.0}], f)
            
            mock_load.return_value = ([{"word": "test", "start": 0.0, "end": 1.0}], True)
            
            output_chunk = os.path.join(tmpdir, "chunk_000_cleaned.m4a")
            with open(output_chunk, 'wb') as f:
                f.write(b'encoded')
            
            mock_encode.return_value = output_chunk
            
            result = chunker._process_chunk(chunk_file, 1)
            
            assert result == output_chunk


class TestProcessChunksSerial:
    """Test serial chunk processing workflow."""
    
    @patch('monkeyplug.audio_chunker.AudioChunker._process_chunk')
    def test_processes_all_chunks_serially(self, mock_process):
        """Should process all chunks one at a time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            chunk_files = [
                os.path.join(tmpdir, f"chunk_{i}.m4a")
                for i in range(3)
            ]
            
            for chunk_file in chunk_files:
                with open(chunk_file, 'wb') as f:
                    f.write(b'chunk')
            
            mock_process.side_effect = [
                f"{chunk}_cleaned" for chunk in chunk_files
            ]
            
            result = chunker._process_chunks_serial(chunk_files)
            
            assert len(result) == 3
            assert mock_process.call_count == 3
    
    @patch('monkeyplug.audio_chunker.AudioChunker._process_chunk')
    def test_continues_on_chunk_failure(self, mock_process):
        """Should continue processing even if one chunk fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            chunk_files = [
                os.path.join(tmpdir, f"chunk_{i}.m4a")
                for i in range(3)
            ]
            
            for chunk_file in chunk_files:
                with open(chunk_file, 'wb') as f:
                    f.write(b'chunk')
            
            # Second chunk fails
            mock_process.side_effect = [
                f"{chunk_files[0]}_cleaned",
                Exception("Processing failed"),
                f"{chunk_files[2]}_cleaned"
            ]
            
            result = chunker._process_chunks_serial(chunk_files)
            
            assert len(result) == 3
            # Failed chunk should be original file
            assert result[1] == chunk_files[1]


class TestProcessChunksParallel:
    """Test parallel chunk processing workflow."""
    
    @patch('monkeyplug.audio_chunker.AudioChunker._encode_all_chunks_parallel')
    @patch('monkeyplug.audio_chunker.AudioChunker._transcribe_all_chunks')
    def test_runs_transcription_then_encoding(self, mock_transcribe, mock_encode):
        """Should run transcription serially, then encoding in parallel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger, parallel_encoding=True)
            
            chunk_files = [
                os.path.join(tmpdir, f"chunk_{i}.m4a")
                for i in range(3)
            ]
            
            transcripts = [f"{c}_transcript.json" for c in chunk_files]
            mock_transcribe.return_value = transcripts
            
            processed = [f"{c}_cleaned" for c in chunk_files]
            mock_encode.return_value = processed
            
            result = chunker._process_chunks_parallel(chunk_files)
            
            assert len(result) == 3
            mock_transcribe.assert_called_once_with(chunk_files)
            mock_encode.assert_called_once_with(chunk_files, transcripts)


class TestTranscribeChunk:
    """Test individual chunk transcription."""
    
    @patch('monkeyplug.utilities.TranscriptManager.load_or_create_transcript')
    def test_creates_transcript_for_chunk(self, mock_load):
        """Should create transcript and save it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            original_wordlist = plugger.wordList.copy()
            chunker = AudioChunker(tmpdir, plugger)
            
            chunk_file = os.path.join(tmpdir, "chunk_000.m4a")
            with open(chunk_file, 'wb') as f:
                f.write(b'chunk')
            
            word_list = [{"word": "test", "start": 0.0, "end": 1.0}]
            mock_load.return_value = (word_list, False)
            
            result = chunker._transcribe_chunk(chunk_file, 1)
            
            assert result.endswith("_transcript.json")
            # State should be restored after _transcribe_chunk
            assert plugger.wordList == original_wordlist


class TestEncodeChunkFromTranscript:
    """Test encoding chunk from existing transcript."""
    
    @patch('monkeyplug.utilities.FFmpegRunner.run_encode')
    def test_loads_transcript_and_encodes(self, mock_encode):
        """Should load transcript and encode chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            chunk_file = os.path.join(tmpdir, "chunk_000.m4a")
            with open(chunk_file, 'wb') as f:
                f.write(b'chunk')
            
            transcript_file = os.path.join(tmpdir, "chunk_000_transcript.json")
            with open(transcript_file, 'w') as f:
                json.dump([{"word": "test", "start": 0.0}], f)
            
            output_chunk = os.path.join(tmpdir, "chunk_000_cleaned.m4a")
            with open(output_chunk, 'wb') as f:
                f.write(b'encoded')
            
            mock_encode.return_value = output_chunk
            
            result = chunker._encode_chunk_from_transcript(
                chunk_file, transcript_file, 1
            )
            
            assert result == output_chunk
            mock_encode.assert_called_once()


class TestProcessWithChunking:
    """Test full chunking workflow integration."""
    
    @patch('monkeyplug.audio_chunker.AudioChunker._concatenate_chunks')
    @patch('monkeyplug.audio_chunker.AudioChunker._process_chunks_serial')
    @patch('monkeyplug.audio_chunker.AudioChunker._split_audio_at_silence')
    @patch('monkeyplug.audio_chunker.AudioChunker._extract_chapters')
    def test_complete_workflow_serial(self, mock_extract, mock_split, 
                                      mock_process, mock_concat):
        """Should execute complete chunking workflow with serial processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger, parallel_encoding=False)
            
            source_file = os.path.join(tmpdir, "source.m4a")
            output_file = os.path.join(tmpdir, "output.m4a")
            
            with open(source_file, 'wb') as f:
                f.write(b'source')
            
            chapter_file = Path(tmpdir) / "chapters.txt"
            mock_extract.return_value = chapter_file
            
            chunks = [f"{tmpdir}/chunk_{i}.m4a" for i in range(2)]
            mock_split.return_value = chunks
            
            processed = [f"{c}_cleaned" for c in chunks]
            mock_process.return_value = processed
            
            # Mock concat to create output file
            def create_output(*args, **kwargs):
                with open(output_file, 'wb') as f:
                    f.write(b'output')
            
            mock_concat.side_effect = create_output
            
            chunker.process_with_chunking(source_file, output_file)
            
            mock_extract.assert_called_once()
            mock_split.assert_called_once()
            mock_process.assert_called_once_with(chunks)
            mock_concat.assert_called_once()
    
    @patch('monkeyplug.audio_chunker.AudioChunker._concatenate_chunks')
    @patch('monkeyplug.audio_chunker.AudioChunker._process_chunks_parallel')
    @patch('monkeyplug.audio_chunker.AudioChunker._split_audio_at_silence')
    @patch('monkeyplug.audio_chunker.AudioChunker._extract_chapters')
    def test_complete_workflow_parallel(self, mock_extract, mock_split,
                                       mock_process, mock_concat):
        """Should execute complete chunking workflow with parallel processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger, parallel_encoding=True)
            
            source_file = os.path.join(tmpdir, "source.m4a")
            output_file = os.path.join(tmpdir, "output.m4a")
            
            with open(source_file, 'wb') as f:
                f.write(b'source')
            
            mock_extract.return_value = None
            
            chunks = [f"{tmpdir}/chunk_{i}.m4a" for i in range(2)]
            mock_split.return_value = chunks
            
            processed = [f"{c}_cleaned" for c in chunks]
            mock_process.return_value = processed
            
            def create_output(*args, **kwargs):
                with open(output_file, 'wb') as f:
                    f.write(b'output')
            
            mock_concat.side_effect = create_output
            
            chunker.process_with_chunking(source_file, output_file)
            
            # Should use parallel processing
            mock_process.assert_called_once_with(chunks)
    
    @patch('monkeyplug.audio_chunker.AudioChunker._extract_chapters')
    def test_handles_workflow_error(self, mock_extract):
        """Should raise AudioChunkingError on workflow failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            chunker = AudioChunker(tmpdir, plugger)
            
            source_file = os.path.join(tmpdir, "source.m4a")
            output_file = os.path.join(tmpdir, "output.m4a")
            
            mock_extract.side_effect = Exception("Chapter extraction failed")
            
            with pytest.raises(AudioChunkingError) as exc_info:
                chunker.process_with_chunking(source_file, output_file)
            
            assert "chunking failed" in str(exc_info.value).lower()


class TestLogging:
    """Test debug logging functionality."""
    
    def test_logs_when_debug_enabled(self):
        """Should log messages when debug is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            plugger.debug = True
            chunker = AudioChunker(tmpdir, plugger)
            
            # Test that _log doesn't raise errors
            chunker._log("Test message")
            chunker._log_section("Test Section")
            
            # No assertion needed - just verify no exceptions
    
    def test_no_logs_when_debug_disabled(self):
        """Should not log when debug is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugger = MockPlugger()
            plugger.debug = False
            chunker = AudioChunker(tmpdir, plugger)
            
            # Should not raise any errors
            chunker._log("Test message")
            chunker._log_section("Test Section")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
