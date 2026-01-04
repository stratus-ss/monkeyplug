#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FFmpeg command construction and audio processing utilities for monkeyplug."""

import datetime
import json
import logging
import mmguero
import os
import subprocess
import sys


class MonkeyplugLogger:
    """Centralized logging for monkeyplug - handles console and file output."""
    
    def __init__(self, output_file_path=None, enabled=False, prefix=None):
        """
        Initialize logger.
        
        Args:
            output_file_path: Path to output file (log will be saved alongside)
            enabled: Whether logging is enabled (verbose mode)
            prefix: Optional prefix for log messages (e.g., "[AudioChunker]")
        """
        self.enabled = enabled
        self.prefix = prefix
        self.log_file = None
        self.logger = None
        
        if enabled and output_file_path:
            self._setup_file_logging(output_file_path)
    
    def _setup_file_logging(self, output_file_path):
        """Setup file logging based on output file path."""
        output_base = os.path.splitext(output_file_path)[0]
        self.log_file = f"{output_base}.log"
        
        # Create a unique logger for this instance
        logger_name = f'monkeyplug_{id(self)}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Log file created: {self.log_file}")
        mmguero.eprint(f"Log file: {self.log_file}")
    
    def log(self, message, level='info'):
        """
        Log a message to console and file.
        
        Args:
            message: Message to log
            level: Log level ('info', 'warning', 'error', 'debug')
        """
        if not self.enabled:
            return
        
        # Build formatted message
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console_prefix = f"[{timestamp}]"
        if self.prefix:
            console_prefix += f" {self.prefix}"
        console_message = f"{console_prefix} {message}"
        
        # Console output
        mmguero.eprint(console_message)
        
        # File output (without timestamp - logger adds it)
        if self.logger:
            file_message = f"{self.prefix} {message}" if self.prefix else message
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(file_message)
    
    def log_section(self, title):
        """Log a section header with separator lines."""
        separator = "=" * 60
        self.log(separator)
        self.log(title)
        self.log(separator)
    
    def info(self, message):
        """Log info message."""
        self.log(message, level='info')
    
    def warning(self, message):
        """Log warning message."""
        self.log(message, level='warning')
    
    def error(self, message):
        """Log error message."""
        self.log(message, level='error')
    
    def debug(self, message):
        """Log debug message."""
        self.log(message, level='debug')


class FFmpegCommandBuilder:
    """Builds FFmpeg commands for audio encoding and processing."""
    
    @staticmethod
    def build_encode_command(input_file, output_file, audio_params, audio_args=None, video_mode=False):
        """Build FFmpeg command for encoding audio with optional filters."""
        base = ['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-loglevel', 'error', '-y', '-i', input_file]
        if video_mode:
            base.extend(['-c:v', 'copy', '-sn', '-dn'])
        else:
            base.extend(['-vn', '-sn', '-dn'])
        if audio_args:
            base.extend(audio_args)
        base.extend(audio_params)
        base.append(output_file)
        return base
    
    @staticmethod
    def build_intermediate_wav_command(input_file, output_file, intermediate_params):
        """Build FFmpeg command to create intermediate WAV file."""
        return ['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-loglevel', 'error', '-y',
                '-i', input_file, '-vn', '-sn', '-dn', *intermediate_params, output_file]
    
    @staticmethod
    def build_probe_command(input_file):
        """Build ffprobe command to get codec/format information."""
        return ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_file]
    
    @staticmethod
    def build_duration_probe_command(input_file):
        """Build ffprobe command to get audio duration."""
        return ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
    
    @staticmethod
    def build_silence_detect_command(input_file, noise_threshold="-40dB", min_duration=0.5):
        """Build ffmpeg command to detect silence points."""
        return ['ffmpeg', '-i', input_file, '-af', f'silencedetect=noise={noise_threshold}:d={min_duration}',
                '-f', 'null', '-']
    
    @staticmethod
    def build_split_command(input_file, split_times, output_pattern):
        """Build ffmpeg command to split audio at timestamps."""
        times_str = ",".join(str(t) for t in split_times)
        return ['ffmpeg', '-i', input_file, '-f', 'segment', '-segment_times', times_str,
                '-c', 'copy', '-map', '0:a', '-map', '0:v?', '-reset_timestamps', '1', output_pattern]
    
    @staticmethod
    def build_concat_command(concat_list_file, output_file, chapter_file=None):
        """Build ffmpeg command to concatenate audio chunks."""
        if chapter_file:
            base, ext = os.path.splitext(output_file)
            temp_output = f"{base}.temp{ext}"
            concat_cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list_file,
                         '-c', 'copy', '-map', '0:a', '-map', '0:v?', '-y', temp_output]
            restore_cmd = ['ffmpeg', '-i', temp_output, '-i', chapter_file,
                          '-map_metadata', '1', '-c', 'copy', '-y', output_file]
            return (concat_cmd, restore_cmd, temp_output)
        return ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list_file,
                '-c', 'copy', '-map', '0:a', '-map', '0:v?', '-y', output_file]
    
    @staticmethod
    def build_extract_chapters_command(input_file, metadata_file):
        """Build ffmpeg command to extract chapter metadata."""
        return ['ffmpeg', '-i', input_file, '-f', 'ffmetadata', '-y', metadata_file]


class AudioFilterBuilder:
    """Builds audio filter strings for muting/beeping profanity."""
    
    @staticmethod
    def build_mute_filters(mute_time_list):
        """Build FFmpeg audio filter arguments for muting."""
        return ['-af', ','.join(mute_time_list)] if mute_time_list else []
    
    @staticmethod
    def build_beep_filters(mute_time_list, sine_time_list, beep_delay_list, 
                          mix_normalize=False, audio_weight=1, sine_weight=1, dropout_transition=0):
        """Build FFmpeg audio filter arguments for beeping."""
        if not mute_time_list:
            return []
        
        mute_str = ','.join(mute_time_list)
        sine_str = ';'.join([f'{val}[beep{i+1}]' for i, val in enumerate(sine_time_list)])
        delay_str = ';'.join([f'[beep{i+1}]{val}[beep{i+1}_delayed]' for i, val in enumerate(beep_delay_list)])
        mix_list = ''.join([f'[beep{i+1}_delayed]' for i in range(len(beep_delay_list))])
        
        weights = f"{audio_weight} {' '.join([str(sine_weight)] * len(beep_delay_list))}"
        filter_str = (f"[0:a]{mute_str}[mute];{sine_str};{delay_str};"
                     f"[mute]{mix_list}amix=inputs={len(beep_delay_list)+1}:"
                     f"normalize={str(mix_normalize).lower()}:dropout_transition={dropout_transition}:weights={weights}")
        return ['-filter_complex', filter_str]
    
    @staticmethod
    def create_mute_time_entry(start_time, end_time, peek_start_time):
        """Create mute filter entries for a single word (fade out + fade in)."""
        start_str = format(start_time, ".3f")
        end_str = format(end_time, ".3f")
        peek_str = format(peek_start_time, ".3f")
        return [
            f"afade=enable='between(t,{start_str},{end_str})':t=out:st={start_str}:d=5ms",
            f"afade=enable='between(t,{end_str},{peek_str})':t=in:st={end_str}:d=5ms"
        ]
    
    @staticmethod
    def create_beep_entries(start_time, end_time, beep_hertz=1000):
        """Create beep filter entries for a single word."""
        start_str = format(start_time, ".3f")
        end_str = format(end_time, ".3f")
        duration = format(end_time - start_time, ".3f")
        delay_ms = int(start_time * 1000)
        
        mute_entry = f"volume=enable='between(t,{start_str},{end_str})':volume=0"
        sine_entry = f"sine=f={beep_hertz}:duration={duration}"
        delay_entry = f"atrim=0:{duration},adelay={'|'.join([str(delay_ms)] * 2)}"
        return (mute_entry, sine_entry, delay_entry)


class FFmpegRunner:
    """Executes FFmpeg commands and handles results."""
    
    @staticmethod
    def run_command(cmd, debug=False):
        """Execute an FFmpeg or ffprobe command."""
        return mmguero.run_process(cmd, stdout=True, stderr=True, debug=debug)
    
    @staticmethod
    def run_encode(input_file, output_file, audio_params, audio_args=None, video_mode=False, debug=False):
        """Execute FFmpeg encoding command and validate output."""
        cmd = FFmpegCommandBuilder.build_encode_command(input_file, output_file, audio_params, 
                                                         audio_args, video_mode)
        result, output = FFmpegRunner.run_command(cmd, debug=debug)
        
        if result != 0 or not os.path.isfile(output_file):
            # Capture error details for exception message
            error_msg = f"Could not encode {input_file}"
            if result != 0:
                error_msg += f" (exit code: {result})"
            if not os.path.isfile(output_file):
                error_msg += " (output file not created)"
            
            # Always show error details regardless of debug mode
            if debug:
                mmguero.eprint(' '.join(mmguero.flatten(cmd)))
                mmguero.eprint(f"Return code: {result}")
                mmguero.eprint(output)
            
            # Include output in exception for better error reporting
            if output:
                error_msg += f" - FFmpeg output: {output[:500]}"  # Limit to 500 chars
            
            raise ValueError(error_msg)
        return output_file
    
    @staticmethod
    def run_probe(input_file, debug=False):
        """Execute ffprobe and return parsed JSON result."""
        cmd = FFmpegCommandBuilder.build_probe_command(input_file)
        result, output = FFmpegRunner.run_command(cmd, debug=debug)
        
        if result != 0:
            if debug:
                mmguero.eprint(' '.join(mmguero.flatten(cmd)))
                mmguero.eprint(f"Return code: {result}")
                mmguero.eprint(output)
            raise ValueError(f"Could not analyze {input_file}")
        return mmguero.load_str_if_json(' '.join(output))
    
    @staticmethod
    def get_audio_duration(input_file):
        """Get duration of audio file in seconds."""
        cmd = FFmpegCommandBuilder.build_duration_probe_command(input_file)
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    
    @staticmethod
    def detect_silence_points(input_file, noise_threshold="-40dB", min_duration=0.5):
        """Detect silence points in audio file."""
        cmd = FFmpegCommandBuilder.build_silence_detect_command(input_file, noise_threshold, min_duration)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        silence_points = []
        for line in result.stderr.split("\n"):
            if "silence_end:" in line:
                try:
                    timestamp = float(line.split("silence_end:")[1].split("|")[0].strip())
                    silence_points.append(timestamp)
                except (ValueError, IndexError):
                    continue
        return silence_points


class CensorshipReport:
    """Generates reports of censored words and silence insertions."""
    
    @staticmethod
    def generate_report(word_list: list, output_path: str, format: str = 'json'):
        """
        Generate a censorship report showing all words being censored.
        
        Args:
            word_list: List of word dictionaries with 'scrub' flags
            output_path: Path to save the report
            format: Report format ('txt' or 'json', default: 'json')
        """
        censored_words = [w for w in word_list if w.get('scrub', False)]
        
        if format == 'txt':
            CensorshipReport._generate_text_report(censored_words, output_path)
        elif format == 'json':
            CensorshipReport._generate_json_report(censored_words, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'txt' or 'json'")
    
    @staticmethod
    def _generate_text_report(censored_words: list, output_path: str):
        """Generate plain text censorship report."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MONKEYPLUG CENSORSHIP REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            if not censored_words:
                f.write("No words were censored.\n")
                return
            
            f.write(f"Total words censored: {len(censored_words)}\n\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'#':<6} {'Word':<20} {'Start Time':<12} {'End Time':<12} {'Confidence':<12}\n")
            f.write("-" * 80 + "\n")
            
            for idx, word in enumerate(censored_words, 1):
                word_text = word.get('word', 'N/A')
                start = word.get('start', 0)
                end = word.get('end', 0)
                conf = word.get('conf', 1.0)
                
                f.write(f"{idx:<6} {word_text:<20} {start:<12.2f} {end:<12.2f} {conf:<12.3f}\n")
            
            f.write("-" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total censored words: {len(censored_words)}\n")
            
            if censored_words:
                total_duration = sum(w.get('end', 0) - w.get('start', 0) for w in censored_words)
                f.write(f"Total silence duration: {total_duration:.2f} seconds\n")
                
                first_censor = min(w.get('start', 0) for w in censored_words)
                last_censor = max(w.get('end', 0) for w in censored_words)
                f.write(f"First censorship at: {first_censor:.2f}s\n")
                f.write(f"Last censorship at: {last_censor:.2f}s\n")
                
                # Word frequency
                from collections import Counter
                word_counts = Counter(w.get('word', 'N/A') for w in censored_words)
                f.write("\nMost censored words:\n")
                for word, count in word_counts.most_common(10):
                    f.write(f"  {word}: {count} occurrence(s)\n")
    
    @staticmethod
    def _generate_json_report(censored_words: list, output_path: str):
        """Generate JSON censorship report."""
        report = {
            'total_censored': len(censored_words),
            'censored_words': censored_words
        }
        
        if censored_words:
            report['summary'] = {
                'total_silence_duration': sum(w.get('end', 0) - w.get('start', 0) for w in censored_words),
                'first_censorship_time': min(w.get('start', 0) for w in censored_words),
                'last_censorship_time': max(w.get('end', 0) for w in censored_words)
            }
            
            # Word frequency
            from collections import Counter
            word_counts = Counter(w.get('word', 'N/A') for w in censored_words)
            report['word_frequency'] = dict(word_counts)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    @staticmethod
    def print_summary(word_list: list):
        """
        Print censorship summary to CLI.
        
        Args:
            word_list: List of word dictionaries with 'scrub' flags
        """
        censored_words = [w for w in word_list if w.get('scrub', False)]
        
        if not censored_words:
            mmguero.eprint("\nNo words were censored.")
            return
        
        from collections import Counter
        
        total_duration = sum(w.get('end', 0) - w.get('start', 0) for w in censored_words)
        word_counts = Counter(w.get('word', 'N/A') for w in censored_words)
        
        mmguero.eprint("\n" + "=" * 70)
        mmguero.eprint("CENSORSHIP SUMMARY")
        mmguero.eprint("=" * 70)
        mmguero.eprint(f"Total words censored: {len(censored_words)}")
        mmguero.eprint(f"Total silence duration: {total_duration:.2f} seconds")
        mmguero.eprint(f"\nMost censored words:")
        for word, count in word_counts.most_common(10):
            mmguero.eprint(f"  - {word}: {count} occurrence(s)")
        mmguero.eprint("=" * 70 + "\n")
    
    @staticmethod
    def save_json_report(word_list: list, output_path: str):
        """
        Save detailed censorship report as pretty-printed JSON.
        
        Args:
            word_list: List of word dictionaries with 'scrub' flags
            output_path: Path to save JSON report
        """
        from collections import Counter
        
        censored_words = [w for w in word_list if w.get('scrub', False)]
        
        report = {
            'total_censored': len(censored_words),
            'censored_words': censored_words
        }
        
        if censored_words:
            report['summary'] = {
                'total_silence_duration': sum(w.get('end', 0) - w.get('start', 0) for w in censored_words),
                'first_censorship_time': min(w.get('start', 0) for w in censored_words),
                'last_censorship_time': max(w.get('end', 0) for w in censored_words)
            }
            
            # Word frequency
            word_counts = Counter(w.get('word', 'N/A') for w in censored_words)
            report['word_frequency'] = dict(word_counts)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        mmguero.eprint(f"Censorship report saved: {output_path}")


class TranscriptManager:
    """Manages transcript file operations for speech recognition results."""
    
    @staticmethod
    def get_transcript_path(audio_file: str) -> str:
        """
        Generate standard transcript path for an audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Path to corresponding transcript JSON file
        """
        audio_basename = os.path.basename(audio_file)
        audio_name = os.path.splitext(audio_basename)[0]
        return os.path.join(
            os.path.dirname(audio_file),
            f"{audio_name}_transcript.json"
        )
    
    @staticmethod
    def save_transcript(transcript_path: str, word_list: list, debug: bool = False, logger=None):
        """
        Save transcript word list to JSON file.
        
        Args:
            transcript_path: Path to save transcript JSON
            word_list: List of word dictionaries to save
            debug: Enable debug logging
            logger: Optional logging function (defaults to mmguero.eprint)
        """
        log_fn = logger if logger else mmguero.eprint
        
        if not word_list:
            if debug:
                log_fn("Warning: Empty word list, not saving transcript")
            return
        
        with open(transcript_path, "w") as f:
            json.dump(word_list, f)
        
        if debug:
            log_fn(f"Saved transcript: {os.path.basename(transcript_path)} ({len(word_list)} words)")
    
    @staticmethod
    def load_transcript(transcript_path: str, swears_map: dict, 
                       confidence_threshold: float = 0.65, debug: bool = False, logger=None) -> list:
        """
        Load transcript from JSON file and recalculate scrub flags.
        
        Args:
            transcript_path: Path to transcript JSON file
            swears_map: Dictionary of profanity words (already scrubbed/lowercased)
            confidence_threshold: Minimum confidence to scrub a word
            debug: Enable debug logging
            logger: Optional logging function (defaults to mmguero.eprint)
            
        Returns:
            List of word dictionaries with updated 'scrub' flags
            
        Raises:
            FileNotFoundError: If transcript file doesn't exist
        """
        log_fn = logger if logger else mmguero.eprint
        
        if not os.path.isfile(transcript_path):
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
        with open(transcript_path, 'r') as f:
            word_list = json.load(f)
        
        # Recalculate scrub flags with current swears list and confidence threshold
        import string
        for word in word_list:
            word_text = word.get('word', '')
            word_conf = word.get('conf', 1.0)
            
            # Scrub word (remove punctuation, lowercase)
            scrubbed = str(word_text).lower().strip().translate(
                str.maketrans('', '', string.punctuation)
            )
            
            # Don't censor empty strings (e.g., punctuation-only words like "%", "!", etc.)
            word['scrub'] = (scrubbed and 
                           scrubbed in swears_map and 
                           word_conf >= confidence_threshold)
        
        if debug:
            scrubbed_count = sum(1 for w in word_list if w.get('scrub', False))
            log_fn(f"Loaded transcript: {os.path.basename(transcript_path)} ({len(word_list)} words, {scrubbed_count} to censor)")
        
        return word_list
    
    @staticmethod
    def load_or_create_transcript(audio_file: str, transcript_path: str,
                                  create_callback, swears_map: dict = None,
                                  confidence_threshold: float = 0.65,
                                  debug: bool = False, logger=None) -> tuple:
        """
        Load existing transcript or create new one via callback.
        
        Args:
            audio_file: Path to audio file
            transcript_path: Path to transcript file
            create_callback: Function to call to create transcript (should return word_list)
            swears_map: Dictionary of profanity words for loading existing transcript
            confidence_threshold: Minimum confidence to scrub a word
            debug: Enable debug logging
            logger: Optional logging function (defaults to mmguero.eprint)
            
        Returns:
            Tuple of (word_list, was_loaded_from_file)
        """
        log_fn = logger if logger else mmguero.eprint
        
        if os.path.exists(transcript_path):
            if debug:
                log_fn(f"Reusing existing transcript: {os.path.basename(transcript_path)}")
            
            word_list = TranscriptManager.load_transcript(
                transcript_path, swears_map, confidence_threshold, debug, logger
            )
            return (word_list, True)
        else:
            if debug:
                log_fn(f"Creating new transcript for: {os.path.basename(audio_file)}")
            
            word_list = create_callback()
            
            if word_list:
                TranscriptManager.save_transcript(transcript_path, word_list, debug, logger)
            
            return (word_list, False)


def get_codecs(local_filename, debug=False):
    """Get stream codecs from an input filename.
    
    Returns dict like: {'video': {'h264'}, 'audio': {'eac3'}, 'subtitle': {'subrip'}, 'format': ['mp4', 'mov']}
    """
    result = {}
    if not os.path.isfile(local_filename):
        return result
    
    probe_output = FFmpegRunner.run_probe(local_filename, debug=debug)
    
    if 'streams' in probe_output:
        for stream in probe_output['streams']:
            if 'codec_name' in stream and 'codec_type' in stream:
                codec_type = stream['codec_type'].lower()
                codec_value = stream['codec_name'].lower()
                if codec_type in result:
                    result[codec_type].add(codec_value)
                else:
                    result[codec_type] = set([codec_value])
    
    result['format'] = mmguero.deep_get(probe_output, ['format', 'format_name'])
    if isinstance(result['format'], str):
        result['format'] = result['format'].split(',')
    return result
