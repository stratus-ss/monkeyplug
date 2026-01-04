"""
Audio file chunking with metadata preservation for monkeyplug.

This module provides the AudioChunker class which handles:
- Splitting large audio files at silence points
- Processing chunks (serial or parallel)
- Extracting and preserving metadata (chapters, tags)
- Reassembling chunks with metadata restoration
- Validating output file integrity
"""

import datetime
import json
import mmguero
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from monkeyplug.utilities import (
    MonkeyplugLogger,
    FFmpegCommandBuilder,
    AudioFilterBuilder,
    FFmpegRunner,
    TranscriptManager
)


class AudioChunkingError(Exception):
    """Base exception for audio chunking errors."""
    pass


class AudioChunker:
    """
    Manages chunking of large audio files with plugger integration.
    
    This class coordinates the entire chunking workflow:
    - Detects when files need chunking based on size thresholds
    - Splits audio at natural silence points
    - Calls plugger methods to transcribe and encode chunks
    - Supports both serial and parallel processing
    - Reassembles chunks and restores metadata
    """

    MAX_CHUNK_SIZE_MB = 150
    MAX_CHUNK_SIZE_BYTES = MAX_CHUNK_SIZE_MB * 1024 * 1024
    
    SILENCE_NOISE_THRESHOLD = "-40dB"  # Noise level below which is considered silence
    SILENCE_MIN_DURATION = 0.5  # Minimum silence duration in seconds
    SPLIT_POINT_TOLERANCE = 0.2  # Accept silence within 20% of target duration

    def __init__(
        self,
        working_dir: str,
        plugger,
        parallel_encoding: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the AudioChunker.

        Args:
            working_dir: Directory for temporary chunk files
            plugger: Plugger instance (VoskPlugger or WhisperPlugger)
            parallel_encoding: Enable two-phase processing (serial transcription, parallel encoding)
            max_workers: Maximum parallel workers for encoding phase
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.plugger = plugger
        self.parallel_encoding = parallel_encoding
        self.max_workers = max_workers
        self.debug = plugger.debug
        
        # Reuse plugger's logger but with [AudioChunker] prefix
        # This ensures all logs go to the same file
        if hasattr(plugger, 'logger') and plugger.logger:
            # Create a wrapper logger that adds the AudioChunker prefix
            self.logger = MonkeyplugLogger(
                output_file_path=None,  # Don't create a new log file
                enabled=self.debug,
                prefix="[AudioChunker]"
            )
            # Share the underlying Python logger instance
            self.logger.logger = plugger.logger.logger
            self.logger.log_file = plugger.logger.log_file
        else:
            # Fallback if plugger doesn't have a logger
            self.logger = MonkeyplugLogger(
                output_file_path=getattr(plugger, 'outputFileSpec', None),
                enabled=self.debug,
                prefix="[AudioChunker]"
            )

    @contextmanager
    def _plugger_temp_state(self):
        """
        Context manager for temporarily modifying plugger state.
        
        Automatically saves all plugger state on entry and restores it on exit,
        even if an exception occurs. This prevents state leakage between chunks.
        """
        # Save state
        saved_state = {
            'inputFileSpec': self.plugger.inputFileSpec,
            'outputJson': self.plugger.outputJson,
            'inputTranscript': self.plugger.inputTranscript,
            'wordList': self.plugger.wordList.copy()
        }
        
        try:
            yield self.plugger
        finally:
            # Restore state
            self.plugger.inputFileSpec = saved_state['inputFileSpec']
            self.plugger.outputJson = saved_state['outputJson']
            self.plugger.inputTranscript = saved_state['inputTranscript']
            self.plugger.wordList = saved_state['wordList']

    def needs_chunking(self, source_file: str) -> bool:
        """
        Check if a file needs to be chunked based on size.
        """
        file_size = os.path.getsize(source_file)
        return file_size > self.MAX_CHUNK_SIZE_BYTES

    def process_with_chunking(self, source_file: str, output_file: str):
        """
        Complete workflow: split, process chunks, reassemble with metadata.
        """
        try:
            overall_start = datetime.datetime.now()
            self.logger.log_section("Starting chunked processing")

            # Extract chapters before splitting
            step_start = datetime.datetime.now()
            file_id = os.path.splitext(os.path.basename(source_file))[0]
            chapter_file = self._extract_chapters(source_file, file_id)
            step_duration = (datetime.datetime.now() - step_start).total_seconds()
            self.logger.info(f"Chapter extraction completed in {step_duration:.1f}s")

            # Split at silence points
            step_start = datetime.datetime.now()
            chunks = self._split_audio_at_silence(source_file, file_id)
            step_duration = (datetime.datetime.now() - step_start).total_seconds()
            self.logger.info(f"Audio splitting completed in {step_duration:.1f}s")

            # Process chunks (serial or parallel based on configuration)
            if self.parallel_encoding:
                processed_chunks = self._process_chunks_parallel(chunks)
            else:
                processed_chunks = self._process_chunks_serial(chunks)

            # Concatenate chunks with chapter restoration
            step_start = datetime.datetime.now()
            self._concatenate_chunks(processed_chunks, output_file, file_id, chapter_file)
            step_duration = (datetime.datetime.now() - step_start).total_seconds()
            self.logger.info(f"Concatenation completed in {step_duration:.1f}s")

            overall_duration = (datetime.datetime.now() - overall_start).total_seconds()
            self.logger.log_section(f"Chunked processing complete - Total time: {overall_duration:.1f}s ({overall_duration/60:.1f} minutes)")
            
            # Aggregate all chunk transcripts into plugger's wordList for reporting
            self._aggregate_transcripts(chunks)

        except Exception as e:
            raise AudioChunkingError(f"Chunking failed: {e}") from e
    
    def _aggregate_transcripts(self, chunks: list):
        """
        Load and aggregate all chunk transcripts into the plugger's wordList.
        This enables censorship report generation after chunked processing.
        """
        self.logger.info("Aggregating transcripts from all chunks...")
        self.plugger.wordList = []
        
        for chunk_file in chunks:
            transcript_path = TranscriptManager.get_transcript_path(chunk_file)
            if os.path.exists(transcript_path):
                try:
                    with open(transcript_path, 'r') as f:
                        chunk_words = json.load(f)
                        self.plugger.wordList.extend(chunk_words)
                except Exception as e:
                    self.logger.info(f"Warning: Could not load transcript {transcript_path}: {e}")
        
        # Recalculate scrub flags for aggregated list
        for word in self.plugger.wordList:
            word_text = word.get('word', '')
            word_conf = word.get('conf', 1.0)
            import string
            scrubbed = str(word_text).lower().strip().translate(
                str.maketrans('', '', string.punctuation)
            )
            # Don't censor empty strings (e.g., punctuation-only words like "%", "!", etc.)
            word['scrub'] = (scrubbed and 
                           scrubbed in self.plugger.swearsMap and 
                           word_conf >= self.plugger.confidenceThreshold)
        
        self.logger.info(f"Aggregated {len(self.plugger.wordList)} words from {len(chunks)} chunks")

    def _process_chunks_serial(self, chunks: list) -> list:
        """
        Process chunks serially: transcribe + encode each chunk before moving to next.
        """
        cpu_count = os.cpu_count() or 1
        serial_start = datetime.datetime.now()
        self.logger.info(f"Processing {len(chunks)} chunks serially (detected {cpu_count} CPUs, using 1 worker)...")
        
        processed_chunks = []
        for i, chunk_file in enumerate(chunks, 1):
            chunk_start = datetime.datetime.now()
            self.logger.info(f"Processing chunk {i}/{len(chunks)}")
            try:
                processed_chunk = self._process_chunk(chunk_file, i)
                processed_chunks.append(processed_chunk)
                chunk_duration = (datetime.datetime.now() - chunk_start).total_seconds()
                self.logger.info(f"Chunk {i} completed in {chunk_duration:.1f}s")
            except Exception as e:
                self.logger.info(f"ERROR: Failed to process chunk {i}: {e}")
                self.logger.info(f"WARNING: Using original chunk {i}")
                processed_chunks.append(chunk_file)
        
        serial_duration = (datetime.datetime.now() - serial_start).total_seconds()
        self.logger.info(f"Serial processing complete - {len(processed_chunks)} chunks in {serial_duration:.1f}s")
        return processed_chunks

    def _process_chunks_parallel(self, chunks: list) -> list:
        """
        Process chunks in two phases:
        1. Transcribe all chunks serially (don't overload API)
        2. Encode all chunks in parallel (CPU-bound work)
        """
        # Phase 1: Transcription
        transcripts = self._transcribe_all_chunks(chunks)
        
        # Phase 2: Parallel Encoding
        processed_chunks = self._encode_all_chunks_parallel(chunks, transcripts)
        
        return processed_chunks
    
    def _transcribe_all_chunks(self, chunks: list) -> list:
        """Transcribe all chunks serially and return transcript paths."""
        self.logger.log_section("PHASE 1: Serial Transcription")
        start_time = datetime.datetime.now()
        
        transcripts = []
        for i, chunk_file in enumerate(chunks, 1):
            self.logger.info(f"Transcribing chunk {i}/{len(chunks)}")
            transcript_path = self._transcribe_chunk(chunk_file, i)
            transcripts.append(transcript_path)
        
        duration = (datetime.datetime.now() - start_time).total_seconds()
        self.logger.info(f"Transcription complete - {len(transcripts)} transcripts in {duration:.1f}s")
        return transcripts
    
    def _encode_all_chunks_parallel(self, chunks: list, transcripts: list) -> list:
        """Encode all chunks in parallel using up to half of available CPUs."""
        self.logger.log_section("PHASE 2: Parallel Encoding")
        start_time = datetime.datetime.now()
        
        # Determine worker count
        if self.max_workers is None:
            cpu_count = os.cpu_count() or 1
            self.max_workers = max(1, cpu_count // 2)
            self.logger.info(f"Detected {cpu_count} CPUs, using {self.max_workers} workers")
        
        self.logger.info(f"Encoding {len(chunks)} chunks using {self.max_workers} parallel workers...")
        
        # Launch parallel encoding tasks
        processed_chunks = [None] * len(chunks)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit each chunk for encoding
            futures = {}
            for i in range(len(chunks)):
                chunk_file = chunks[i]
                transcript_path = transcripts[i]
                future = executor.submit(self._encode_chunk_from_transcript, chunk_file, transcript_path, i+1)
                futures[future] = i  # Remember which chunk this future is processing
            
            # Process results as they complete
            completed_count = 0
            for future in as_completed(futures):
                chunk_idx = futures[future]
                completed_count += 1
                try:
                    processed_chunks[chunk_idx] = future.result()
                    self.logger.info(f"Encoded chunk {chunk_idx+1} ({completed_count}/{len(chunks)})")
                except Exception as e:
                    self.logger.info(f"ERROR encoding chunk {chunk_idx+1}: {e}")
                    self.logger.info(f"Using original chunk {chunk_idx+1}")
                    processed_chunks[chunk_idx] = chunks[chunk_idx]
        
        duration = (datetime.datetime.now() - start_time).total_seconds()
        self.logger.info(f"Encoding complete - {len(chunks)} chunks in {duration:.1f}s")
        return processed_chunks

    def _process_chunk(self, chunk_file: str, chunk_index: int) -> str:
        """
        Process a single chunk: transcribe + encode.
        
        Temporarily swaps plugger's input file to the chunk, calls plugger methods,
        then restores original state.
        """
        with self._plugger_temp_state():
            # Configure plugger for this chunk
            self.plugger.inputFileSpec = chunk_file
            self.plugger.wordList = []
            
            # Set up transcript path
            transcript_path = TranscriptManager.get_transcript_path(chunk_file)
            self.plugger.outputJson = transcript_path
            
            # Load existing transcript or create new one
            if os.path.exists(transcript_path):
                self.logger.info(f"Reusing existing transcript for chunk {chunk_index}")
                self.plugger.inputTranscript = transcript_path
            
            # CreateCleanMuteList will either load the transcript (if inputTranscript is set)
            # or perform speech recognition, then create the mute list
            self.plugger.CreateCleanMuteList()
            
            # Save transcript if we just created it
            if not os.path.exists(transcript_path) and self.plugger.wordList:
                TranscriptManager.save_transcript(
                    transcript_path=transcript_path,
                    word_list=self.plugger.wordList,
                    debug=self.debug,
                    logger=self.logger.info
                )
            
            # Encode chunk
            chunk_name = os.path.splitext(os.path.basename(chunk_file))[0]
            output_chunk = os.path.join(
                os.path.dirname(chunk_file),
                f"{chunk_name}_cleaned{os.path.splitext(chunk_file)[1]}"
            )
            
            self._run_ffmpeg_encode(chunk_file, output_chunk)
            
            return output_chunk

    def _transcribe_chunk(self, chunk_file: str, chunk_index: int) -> str:
        """
        Phase 1: Transcribe a chunk and save transcript (no encoding).
        """
        with self._plugger_temp_state():
            # Configure plugger for this chunk
            transcript_path = TranscriptManager.get_transcript_path(chunk_file)
            
            self.plugger.inputFileSpec = chunk_file
            self.plugger.wordList = []
            self.plugger.outputJson = transcript_path
            
            # Load existing or create new transcript
            def create_transcript():
                self.plugger.RecognizeSpeech()
                return self.plugger.wordList
            
            word_list, was_loaded = TranscriptManager.load_or_create_transcript(
                audio_file=chunk_file,
                transcript_path=transcript_path,
                create_callback=create_transcript,
                swears_map=self.plugger.swearsMap,
                confidence_threshold=self.plugger.confidenceThreshold,
                debug=self.debug,
                logger=self.logger.info
            )
            self.plugger.wordList = word_list
            
            return transcript_path

    def _encode_chunk_from_transcript(self, chunk_file: str, transcript_path: str, chunk_index: int) -> str:
        """
        Phase 2: Load transcript and encode chunk with ffmpeg.
        
        This method is called in parallel workers, so it must not modify plugger state.
        Note: Verbose debug output is suppressed during parallel execution to avoid log interleaving.
        """
        with self._plugger_temp_state():
            # Temporarily suppress verbose debug output during parallel execution
            original_debug = self.plugger.debug
            self.plugger.debug = False
            
            try:
                # Load transcript and build mute list
                self.plugger.inputFileSpec = chunk_file
                self.plugger.inputTranscript = transcript_path
                self.plugger.wordList = []
                self.plugger.LoadTranscriptFromFile()
                self.plugger.CreateCleanMuteList()
                
                # Generate output filename
                chunk_name = os.path.splitext(os.path.basename(chunk_file))[0]
                output_chunk = os.path.join(
                    os.path.dirname(chunk_file),
                    f"{chunk_name}_cleaned{os.path.splitext(chunk_file)[1]}"
                )
                
                # Build ffmpeg command with audio filters (reuse plugger's logic)
                self._run_ffmpeg_encode(chunk_file, output_chunk)
                
                return output_chunk
            finally:
                self.plugger.debug = original_debug

    def _run_ffmpeg_encode(self, input_file: str, output_file: str):
        """
        Encode chunk with ffmpeg using plugger's filter lists.
        
        Uses utilities module for consistent ffmpeg command construction.
        """
        # Build audio filter arguments using utilities module
        if len(self.plugger.muteTimeList) > 0:
            if self.plugger.beep:
                audioArgs = AudioFilterBuilder.build_beep_filters(
                    self.plugger.muteTimeList,
                    self.plugger.sineTimeList,
                    self.plugger.beepDelayList,
                    mix_normalize=self.plugger.beepMixNormalize,
                    audio_weight=self.plugger.beepAudioWeight,
                    sine_weight=self.plugger.beepSineWeight,
                    dropout_transition=self.plugger.beepDropTransition
                )
            else:
                audioArgs = AudioFilterBuilder.build_mute_filters(self.plugger.muteTimeList)
        else:
            audioArgs = []
        
        # Use utilities module to run ffmpeg command
        FFmpegRunner.run_encode(
            input_file=input_file,
            output_file=output_file,
            audio_params=self.plugger.aParams,
            audio_args=audioArgs,
            video_mode=False,
            debug=self.debug
        )

    def _split_audio_at_silence(self, source_file: str, file_id: str) -> list:
        """
        Split large audio file into chunks at silence points.
        """
        chunk_dir = self.working_dir / file_id / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        file_ext = os.path.splitext(source_file)[1].lstrip(".") or "m4a"
        
        # Check for existing chunks
        pattern = re.compile(rf"^{re.escape(base_name)}_chunk_\d{{3}}\.{re.escape(file_ext)}$")
        existing_chunks = sorted([
            str(p) for p in chunk_dir.glob(f"{base_name}_chunk_*.{file_ext}")
            if pattern.match(p.name) and p.is_file()
        ])
        
        if existing_chunks:
            self.logger.info(f"Reusing {len(existing_chunks)} existing chunks")
            return existing_chunks
        
        # Calculate target duration
        file_size = os.path.getsize(source_file)
        num_chunks = (file_size // self.MAX_CHUNK_SIZE_BYTES) + 1
        duration = self._get_audio_duration(source_file)
        target_duration = duration / num_chunks
        
        self.logger.info(f"Source file: {source_file}")
        self.logger.info(f"File size on disk: {file_size / (1024*1024):.1f} MB ({file_size} bytes)")
        self.logger.info(f"Splitting into ~{num_chunks} chunks of ~{self.MAX_CHUNK_SIZE_MB} MB each")
        self.logger.info(f"Total duration: {duration:.1f}s, Target chunk: {target_duration:.1f}s")
        
        # Detect silence and split
        silence_points = self._detect_silence_points(source_file)
        split_times = self._select_split_points(silence_points, duration, target_duration)
        
        if not split_times:
            self.logger.info("WARNING: No split points found, file may be too short")
            return [source_file]
        
        # Perform split using utilities module
        chunk_pattern = str(chunk_dir / f"{base_name}_chunk_%03d.{file_ext}")
        split_cmd = FFmpegCommandBuilder.build_split_command(
            source_file,
            split_times,
            chunk_pattern
        )
        subprocess.run(split_cmd, check=True, capture_output=True)
        
        # Collect created chunks
        chunks = sorted(chunk_dir.glob(f"{base_name}_chunk_*.{file_ext}"))
        chunk_paths = [str(chunk) for chunk in chunks]
        
        if not chunk_paths:
            self.logger.info("WARNING: No chunks created, using original")
            return [source_file]
        
        self.logger.info(f"Created {len(chunk_paths)} chunks")
        return chunk_paths

    def _get_audio_duration(self, source_file: str) -> float:
        """Get audio file duration using ffprobe."""
        return FFmpegRunner.get_audio_duration(source_file)

    def _detect_silence_points(self, source_file: str) -> list:
        """Detect silence points in audio using ffmpeg silencedetect."""
        silence_points = FFmpegRunner.detect_silence_points(
            source_file,
            self.SILENCE_NOISE_THRESHOLD,
            self.SILENCE_MIN_DURATION
        )
        self.logger.info(f"Found {len(silence_points)} silence points")
        return silence_points

    def _select_split_points(self, silence_points: list, duration: float, target_duration: float) -> list:
        """Select optimal silence points for splitting."""
        if not silence_points:
            # Fallback to time-based splits
            num_chunks = int(duration / target_duration) + 1
            return [i * target_duration for i in range(1, num_chunks) if i * target_duration < duration]
        
        split_points = []
        current_target = target_duration
        tolerance = target_duration * self.SPLIT_POINT_TOLERANCE
        
        while current_target < duration:
            # Find nearest silence within tolerance
            closest = min(silence_points, key=lambda x: abs(x - current_target), default=None)
            if closest and abs(closest - current_target) <= tolerance:
                split_points.append(closest)
                current_target = closest + target_duration
            else:
                split_points.append(current_target)
                current_target += target_duration
        
        return split_points

    def _extract_chapters(self, source_file: str, file_id: str) -> Optional[Path]:
        """Extract chapter metadata from source file."""
        metadata_file = self.working_dir / file_id / "chapters.txt"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        extract_cmd = FFmpegCommandBuilder.build_extract_chapters_command(
            source_file,
            str(metadata_file)
        )
        
        try:
            subprocess.run(extract_cmd, check=True, capture_output=True)
            self.logger.info(f"Extracted chapters to {metadata_file}")
            return metadata_file
        except subprocess.CalledProcessError:
            self.logger.info("WARNING: Chapter extraction failed")
            return None

    def _concatenate_chunks(self, chunk_files: list, output_file: str, file_id: str, chapter_file: Optional[Path]):
        """Concatenate audio chunks into final output."""
        self.logger.info(f"Concatenating {len(chunk_files)} chunks...")
        
        # Verify all chunk files exist
        missing_chunks = [f for f in chunk_files if not os.path.exists(f)]
        if missing_chunks:
            raise AudioChunkingError(
                f"Cannot concatenate: {len(missing_chunks)} chunk(s) missing: {missing_chunks[:3]}"
            )
        
        # Create concat list
        concat_list_file = self.working_dir / file_id / "concat_list.txt"
        with open(concat_list_file, "w") as f:
            for chunk_file in chunk_files:
                abs_path = os.path.abspath(chunk_file)
                f.write(f"file '{abs_path}'\n")
        
        if self.debug:
            self.logger.info(f"Concat list saved to: {concat_list_file}")
            with open(concat_list_file, 'r') as f:
                self.logger.info(f"Concat list contents:\n{f.read()}")
        
        # Concatenate using utilities module
        cmd_result = FFmpegCommandBuilder.build_concat_command(
            str(concat_list_file),
            output_file,
            str(chapter_file) if chapter_file and chapter_file.exists() else None
        )
        
        if chapter_file and chapter_file.exists():
            # Unpack tuple of (concat_cmd, restore_cmd, temp_output)
            concat_cmd, restore_cmd, temp_output = cmd_result
            
            try:
                result = subprocess.run(concat_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.info(f"FFmpeg concat error: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, concat_cmd, result.stdout, result.stderr)
            except subprocess.CalledProcessError as e:
                raise AudioChunkingError(
                    f"Concatenation failed (exit code {e.returncode}). "
                    f"This may indicate format mismatches between chunks. "
                    f"FFmpeg output: {e.stderr[:500] if e.stderr else 'N/A'}"
                ) from e
            
            # Restore chapters
            try:
                result = subprocess.run(restore_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.info(f"FFmpeg chapter restore error: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, restore_cmd, result.stdout, result.stderr)
            except subprocess.CalledProcessError as e:
                raise AudioChunkingError(
                    f"Chapter restoration failed (exit code {e.returncode}). "
                    f"FFmpeg output: {e.stderr[:500] if e.stderr else 'N/A'}"
                ) from e
            
            if os.path.exists(temp_output):
                os.remove(temp_output)
        else:
            # Simple concat without chapter restoration
            concat_cmd = cmd_result
            try:
                result = subprocess.run(concat_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.info(f"FFmpeg concat error: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, concat_cmd, result.stdout, result.stderr)
            except subprocess.CalledProcessError as e:
                raise AudioChunkingError(
                    f"Concatenation failed (exit code {e.returncode}). "
                    f"This may indicate format mismatches between chunks. "
                    f"FFmpeg output: {e.stderr[:500] if e.stderr else 'N/A'}"
                ) from e
        
        output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        self.logger.info(f"Concatenation complete - Final output: {output_size_mb:.1f} MB")
        
        # Clean up processed chunk files (keep original chunks for potential reuse)
        self._cleanup_processed_chunks(chunk_files)

    def _cleanup_processed_chunks(self, chunk_files: list):
        """
        Clean up processed (_cleaned) chunk files after successful concatenation.
        
        Keeps original chunks (e.g., _chunk_000.m4b) for potential reuse,
        but removes intermediate processed chunks (e.g., _chunk_000_cleaned.m4b).
        """
        cleaned_count = 0
        for chunk_file in chunk_files:
            if '_cleaned' in chunk_file and os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                    cleaned_count += 1
                except Exception as e:
                    self.logger.info(f"WARNING: Could not delete {chunk_file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} processed chunk files")
