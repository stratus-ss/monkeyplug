#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import errno
import json
import mmguero
import mutagen
import os
import pathlib
import requests
import shutil
import string
import sys
import tempfile
import wave

from urllib.parse import urlparse
from itertools import tee

try:
    from monkeyplug.audio_chunker import AudioChunker
except ImportError:
    AudioChunker = None

from monkeyplug.utilities import (
    FFmpegCommandBuilder,
    AudioFilterBuilder,
    FFmpegRunner,
    TranscriptManager,
    get_codecs as get_codecs_util
)

###################################################################################################
CHANNELS_REPLACER = 'CHANNELS'
SAMPLE_RATE_REPLACER = 'SAMPLE'
AUDIO_DEFAULT_PARAMS_BY_FORMAT = {
    "flac": ["-c:a", "flac", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
    "m4a": ["-c:a", "aac", "-b:a", "128K", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
    "aac": ["-c:a", "aac", "-b:a", "128K", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128K", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
    "ogg": ["-c:a", "libvorbis", "-qscale:a", "5", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
    "opus": ["-c:a", "libopus", "-b:a", "128K", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
    "ac3": ["-c:a", "ac3", "-b:a", "128K", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
    "wav": ["-c:a", "pcm_s16le", "-ar", SAMPLE_RATE_REPLACER, "-ac", CHANNELS_REPLACER],
}
AUDIO_CODEC_TO_FORMAT = {
    "aac": "m4a",
    "ac3": "ac3",
    "flac": "flac",
    "mp3": "mp3",
    "opus": "opus",
    "vorbis": "ogg",
    "pcm_s16le": "wav",
}

AUDIO_DEFAULT_FORMAT = "mp3"
AUDIO_DEFAULT_CHANNELS = 2
AUDIO_DEFAULT_SAMPLE_RATE = 48000
AUDIO_MATCH_FORMAT = "MATCH"
AUDIO_INTERMEDIATE_PARAMS = ["-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000"]
AUDIO_DEFAULT_WAV_FRAMES_CHUNK = 8000
BEEP_HERTZ_DEFAULT = 1000
BEEP_MIX_NORMALIZE_DEFAULT = False
BEEP_AUDIO_WEIGHT_DEFAULT = 1
BEEP_SINE_WEIGHT_DEFAULT = 1
BEEP_DROPOUT_TRANSITION_DEFAULT = 0
CONFIDENCE_THRESHOLD_DEFAULT = 0.65
SWEARS_FILENAME_DEFAULT = 'swears.txt'
MUTAGEN_METADATA_TAGS = ['encodedby', 'comment']
MUTAGEN_METADATA_TAG_VALUE = u'monkeyplug'
SPEECH_REC_MODE_VOSK = "vosk"
SPEECH_REC_MODE_WHISPER = "whisper"
SPEECH_REC_MODE_REMOTE_WHISPER = "remote-whisper"
DEFAULT_SPEECH_REC_MODE = os.getenv("MONKEYPLUG_MODE", SPEECH_REC_MODE_WHISPER)
DEFAULT_VOSK_MODEL_DIR = os.getenv(
    "VOSK_MODEL_DIR", os.path.join(os.path.join(os.path.join(os.path.expanduser("~"), '.cache'), 'vosk'))
)
DEFAULT_WHISPER_MODEL_DIR = os.getenv(
    "WHISPER_MODEL_DIR", os.path.join(os.path.join(os.path.join(os.path.expanduser("~"), '.cache'), 'whisper'))
)
DEFAULT_WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "small.en")
DEFAULT_TORCH_THREADS = 0

###################################################################################################
script_name = os.path.basename(__file__)
script_path = os.path.dirname(os.path.realpath(__file__))


# thanks https://docs.python.org/3/library/itertools.html#recipes
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def scrubword(value):
    return str(value).lower().strip().translate(str.maketrans('', '', string.punctuation))


###################################################################################################
# download to file
def DownloadToFile(url, local_filename=None, chunk_bytes=4096, debug=False):
    tmpDownloadedFileSpec = local_filename if local_filename else os.path.basename(urlparse(url).path)
    r = requests.get(url, stream=True, allow_redirects=True)
    with open(tmpDownloadedFileSpec, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_bytes):
            if chunk:
                f.write(chunk)
    fExists = os.path.isfile(tmpDownloadedFileSpec)
    fSize = os.path.getsize(tmpDownloadedFileSpec)
    if debug:
        mmguero.eprint(
            f"Download of {url} to {tmpDownloadedFileSpec} {'succeeded' if fExists else 'failed'} ({mmguero.size_human_format(fSize)})"
        )

    if fExists and (fSize > 0):
        return tmpDownloadedFileSpec
    else:
        if fExists:
            os.remove(tmpDownloadedFileSpec)
        return None


###################################################################################################
# Get tag from file to indicate monkeyplug has already been set
def GetMonkeyplugTagged(local_filename, debug=False):
    result = False
    if os.path.isfile(local_filename):
        mut = mutagen.File(local_filename, easy=True)
        if debug:
            mmguero.eprint(f'Tags of {local_filename}: {mut}')
        if hasattr(mut, 'get'):
            for tag in MUTAGEN_METADATA_TAGS:
                try:
                    if MUTAGEN_METADATA_TAG_VALUE in mmguero.get_iterable(mut.get(tag, default=())):
                        result = True
                        break
                except Exception as e:
                    if debug:
                        mmguero.eprint(e)
    return result


###################################################################################################
# Set tag to file to indicate monkeyplug has worked its magic
def SetMonkeyplugTag(local_filename, debug=False):
    result = False
    if os.path.isfile(local_filename):
        mut = mutagen.File(local_filename, easy=True)
        if debug:
            mmguero.eprint(f'Tags of {local_filename} before: {mut}')
        if hasattr(mut, '__setitem__'):
            for tag in MUTAGEN_METADATA_TAGS:
                try:
                    mut[tag] = MUTAGEN_METADATA_TAG_VALUE
                    result = True
                    break
                except Exception as e:
                    if debug:
                        mmguero.eprint(e)
            if result:
                try:
                    mut.save(local_filename)
                except Exception as e:
                    result = False
                    mmguero.eprint(e)
            if debug:
                mmguero.eprint(f'Tags of {local_filename} after: {mut}')

    return result


###################################################################################################
# get stream codecs from an input filename
# e.g. result: {'video': {'h264'}, 'audio': {'eac3'}, 'subtitle': {'subrip'}}
def GetCodecs(local_filename, debug=False):
    """Wrapper for get_codecs from utilities module."""
    return get_codecs_util(local_filename, debug=debug)


#################################################################################
class Plugger(object):
    debug = False
    inputFileSpec = ""
    inputCodecs = {}
    inputFileParts = None
    outputFileSpec = ""
    outputAudioFileFormat = ""
    outputVideoFileFormat = ""
    outputJson = ""
    tmpDownloadedFileSpec = ""
    swearsFileSpec = ""
    swearsMap = {}
    wordList = []
    naughtyWordList = []
    # for beep and mute
    muteTimeList = []
    # for beep only
    sineTimeList = []
    beepDelayList = []
    padSecPre = 0.0
    padSecPost = 0.0
    beep = False
    beepHertz = BEEP_HERTZ_DEFAULT
    beepMixNormalize = BEEP_MIX_NORMALIZE_DEFAULT
    beepAudioWeight = BEEP_AUDIO_WEIGHT_DEFAULT
    beepSineWeight = BEEP_SINE_WEIGHT_DEFAULT
    beepDropTransition = BEEP_DROPOUT_TRANSITION_DEFAULT
    forceDespiteTag = False
    aParams = None
    tags = None

    ######## init #################################################################
    def __init__(
        self,
        iFileSpec,
        oFileSpec,
        oAudioFileFormat,
        iSwearsFileSpec,
        outputJson,
        inputTranscript=None,
        saveTranscript=False,
        aParams=None,
        aChannels=AUDIO_DEFAULT_CHANNELS,
        aSampleRate=AUDIO_DEFAULT_SAMPLE_RATE,
        padMsecPre=0,
        padMsecPost=0,
        beep=False,
        beepHertz=BEEP_HERTZ_DEFAULT,
        beepMixNormalize=BEEP_MIX_NORMALIZE_DEFAULT,
        beepAudioWeight=BEEP_AUDIO_WEIGHT_DEFAULT,
        beepSineWeight=BEEP_SINE_WEIGHT_DEFAULT,
        beepDropTransition=BEEP_DROPOUT_TRANSITION_DEFAULT,
        confidenceThreshold=CONFIDENCE_THRESHOLD_DEFAULT,
        force=False,
        useChunking=False,
        chunkingWorkDir=None,
        parallelEncoding=False,
        maxWorkers=None,
        dbug=False,
    ):
        self.padSecPre = padMsecPre / 1000.0
        self.padSecPost = padMsecPost / 1000.0
        self.beep = beep
        self.beepHertz = beepHertz
        self.beepMixNormalize = beepMixNormalize
        self.beepAudioWeight = beepAudioWeight
        self.beepSineWeight = beepSineWeight
        self.beepDropTransition = beepDropTransition
        self.confidenceThreshold = confidenceThreshold
        self.forceDespiteTag = force
        self.debug = dbug
        self.outputJson = outputJson
        self.inputTranscript = inputTranscript
        self.saveTranscript = saveTranscript
        self.useChunking = useChunking
        self.chunkingWorkDir = chunkingWorkDir if chunkingWorkDir else tempfile.gettempdir()
        self.parallelEncoding = parallelEncoding
        self.maxWorkers = maxWorkers

        # determine input file name, or download and save file
        if (iFileSpec is not None) and os.path.isfile(iFileSpec):
            self.inputFileSpec = iFileSpec
        elif iFileSpec.lower().startswith("http"):
            self.tmpDownloadedFileSpec = DownloadToFile(iFileSpec)
            if (self.tmpDownloadedFileSpec is not None) and os.path.isfile(self.tmpDownloadedFileSpec):
                self.inputFileSpec = self.tmpDownloadedFileSpec
            else:
                raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), iFileSpec)
        else:
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), iFileSpec)

        # input file should exist locally by now
        if os.path.isfile(self.inputFileSpec):
            self.inputFileParts = os.path.splitext(self.inputFileSpec)
            self.inputCodecs = GetCodecs(self.inputFileSpec)
            inputFormat = next(
                iter([x for x in self.inputCodecs.get('format', None) if x in AUDIO_DEFAULT_PARAMS_BY_FORMAT]), None
            )
        else:
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), self.inputFileSpec)

        # determine output file name (either specified or based on input filename)
        self.outputFileSpec = oFileSpec if oFileSpec else self.inputFileParts[0] + "_clean"
        if self.outputFileSpec:
            outParts = os.path.splitext(self.outputFileSpec)
            # Only use output file's extension if one exists and no format was specified
            if not oAudioFileFormat and outParts[1]:
                oAudioFileFormat = outParts[1]

        if str(oAudioFileFormat).upper() == AUDIO_MATCH_FORMAT:
            # output format not specified, base on input filename matching extension (or codec)
            if self.inputFileParts[1] in AUDIO_DEFAULT_PARAMS_BY_FORMAT:
                self.outputFileSpec = outParts[0] + self.inputFileParts[1]
            elif str(inputFormat).lower() in AUDIO_DEFAULT_PARAMS_BY_FORMAT:
                self.outputFileSpec = outParts[0] + '.' + inputFormat.lower()
            else:
                for codec in mmguero.get_iterable(self.inputCodecs.get('audio', [])):
                    if codec.lower() in AUDIO_CODEC_TO_FORMAT:
                        self.outputFileSpec = outParts[0] + '.' + AUDIO_CODEC_TO_FORMAT[codec.lower()]
                        break

        elif oAudioFileFormat:
            # output filename specified with extension, use base name to avoid duplication
            self.outputFileSpec = outParts[0] + '.' + oAudioFileFormat.lower().lstrip('.')

        else:
            # can't determine what output file audio format should be
            raise ValueError("Output file audio format unspecified")

        # determine output file extension if it's not already obvious
        outParts = os.path.splitext(self.outputFileSpec)
        self.outputAudioFileFormat = outParts[1].lower().lstrip('.')

        if (not self.outputAudioFileFormat) or (
            (not aParams) and (self.outputAudioFileFormat not in AUDIO_DEFAULT_PARAMS_BY_FORMAT)
        ):
            raise ValueError("Output file audio format unspecified or unsupported")
        elif not aParams:
            # we're using ffmpeg encoding params based on output file format
            self.aParams = AUDIO_DEFAULT_PARAMS_BY_FORMAT[self.outputAudioFileFormat]
        else:
            # they specified custom ffmpeg encoding params
            self.aParams = aParams
            if self.aParams.startswith("base64:"):
                self.aParams = base64.b64decode(self.aParams[7:]).decode("utf-8")
            self.aParams = self.aParams.split(' ')
        self.aParams = [
            {
                CHANNELS_REPLACER: str(aChannels),
                SAMPLE_RATE_REPLACER: str(aSampleRate),
            }.get(aParam, aParam)
            for aParam in self.aParams
        ]

        # if we're actually just replacing the audio stream(s) inside a video file, the actual output file is still a video file
        self.outputVideoFileFormat = (
            self.inputFileParts[1]
            if (
                (len(mmguero.get_iterable(self.inputCodecs.get('video', []))) > 0)
                and (str(oAudioFileFormat).upper() == AUDIO_MATCH_FORMAT)
            )
            else ''
        )
        if self.outputVideoFileFormat:
            self.outputFileSpec = outParts[0] + self.outputVideoFileFormat

        # create output directory if it doesn't exist
        self._ensure_directory_exists(self.outputFileSpec, "output directory")

        # if output file already exists, remove as we'll be overwriting it anyway
        if os.path.isfile(self.outputFileSpec):
            if self.debug:
                mmguero.eprint(f'Removing existing destination file {self.outputFileSpec}')
            os.remove(self.outputFileSpec)

        # If save-transcript is enabled and no explicit JSON output path, auto-generate one
        if self.saveTranscript and not self.outputJson:
            outputBaseName = os.path.splitext(self.outputFileSpec)[0]
            self.outputJson = outputBaseName + '_transcript.json'
            if self.debug:
                mmguero.eprint(f'Auto-generated transcript output: {self.outputJson}')
        
        # If JSON output is specified, ensure its directory exists too
        if self.outputJson:
            self._ensure_directory_exists(self.outputJson, "JSON output directory")

        # load the swears file (not actually mapping right now, but who knows, speech synthesis maybe someday?)
        if (iSwearsFileSpec is not None) and os.path.isfile(iSwearsFileSpec):
            self.swearsFileSpec = iSwearsFileSpec
        else:
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), iSwearsFileSpec)
        
        self._load_swears_file()

        if self.debug:
            mmguero.eprint(f'Input: {self.inputFileSpec}')
            mmguero.eprint(f'Input codec: {self.inputCodecs}')
            mmguero.eprint(f'Output: {self.outputFileSpec}')
            mmguero.eprint(f'Output audio format: {self.outputAudioFileFormat}')
            mmguero.eprint(f'Encode parameters: {self.aParams}')
            mmguero.eprint(f'Profanity file: {self.swearsFileSpec}')
            mmguero.eprint(f'Intermediate downloaded file: {self.tmpDownloadedFileSpec}')
            if self.outputJson:
                mmguero.eprint(f'Transcript output: {self.outputJson}')
            mmguero.eprint(f'Beep instead of mute: {self.beep}')
            if self.beep:
                mmguero.eprint(f'Beep hertz: {self.beepHertz}')
                mmguero.eprint(f'Beep mix normalization: {self.beepMixNormalize}')
                mmguero.eprint(f'Beep audio weight: {self.beepAudioWeight}')
                mmguero.eprint(f'Beep sine weight: {self.beepSineWeight}')
                mmguero.eprint(f'Beep dropout transition: {self.beepDropTransition}')
            mmguero.eprint(f'Force despite tags: {self.forceDespiteTag}')

    ######## del ##################################################################
    def __del__(self):
        # if we downloaded the input file, remove it as well
        if os.path.isfile(self.tmpDownloadedFileSpec):
            os.remove(self.tmpDownloadedFileSpec)

    ######## _ensure_directory_exists #############################################
    def _ensure_directory_exists(self, filepath, description="directory"):
        """Ensure the directory for a file path exists, creating it if necessary"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            if self.debug:
                mmguero.eprint(f'Creating {description}: {directory}')
            os.makedirs(directory, exist_ok=True)
        return directory

    ######## _load_swears_file ####################################################
    def _load_swears_file(self):
        """Load swears from text or JSON format"""
        # Try to detect and parse JSON first
        is_json = False
        if self.swearsFileSpec.lower().endswith('.json'):
            is_json = True
        else:
            # Try to parse as JSON even without .json extension
            try:
                with open(self.swearsFileSpec, 'r') as f:
                    content = f.read()
                    json.loads(content)
                    is_json = True
            except (json.JSONDecodeError, ValueError):
                pass
        
        if is_json:
            self._load_swears_from_json()
        else:
            self._load_swears_from_text()
        
        if self.debug:
            mmguero.eprint(f'Loaded {len(self.swearsMap)} profanity entries from {self.swearsFileSpec}')
    
    def _load_swears_from_json(self):
        """Load swears from JSON format - simple array of strings
        
        Format: ["word1", "word2", "word3", ...]
        Example: https://github.com/zautumnz/profane-words/blob/master/words.json
        """
        with open(self.swearsFileSpec, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"JSON swears file must contain an array of strings, got {type(data).__name__}")
        
        for item in data:
            if isinstance(item, str) and item.strip():
                self.swearsMap[scrubword(item)] = "*****"
    
    def _load_swears_from_text(self):
        """Load swears from pipe-delimited text format (legacy)"""
        lines = []
        with open(self.swearsFileSpec) as f:
            lines = [line.rstrip("\n") for line in f]
        for line in lines:
            lineMap = line.split("|")
            self.swearsMap[scrubword(lineMap[0])] = lineMap[1] if len(lineMap) > 1 else "*****"

    ######## _should_scrub_word ##################################################
    def _should_scrub_word(self, word_text, confidence=1.0):
        """Check if a word should be scrubbed based on swears list and confidence threshold
        
        Args:
            word_text: The word to check
            confidence: Confidence score (0.0 to 1.0), defaults to 1.0
            
        Returns:
            bool: True if word should be scrubbed, False otherwise
        """
        return (scrubword(word_text) in self.swearsMap and 
                confidence >= self.confidenceThreshold)

    ######## LoadTranscriptFromFile ##############################################
    def LoadTranscriptFromFile(self):
        """Load pre-generated transcript from JSON file"""
        if not self.inputTranscript:
            return False
        
        if not os.path.isfile(self.inputTranscript):
            raise IOError(errno.ENOENT, os.strerror(errno.ENOENT), self.inputTranscript)
        
        self.wordList = TranscriptManager.load_transcript(
            transcript_path=self.inputTranscript,
            swears_map=self.swearsMap,
            confidence_threshold=self.confidenceThreshold,
            debug=self.debug
        )
        
        return True

    ######## CreateCleanMuteList #################################################
    def CreateCleanMuteList(self):
        # Try to load existing transcript first, otherwise perform speech recognition
        if not self.LoadTranscriptFromFile():
            self.RecognizeSpeech()

        self.naughtyWordList = [word for word in self.wordList if word["scrub"] is True]
        if len(self.naughtyWordList) > 0:
            # append a dummy word at the very end so that pairwise can peek then ignore it
            self.naughtyWordList.extend(
                [
                    {
                        "conf": 1,
                        "end": self.naughtyWordList[-1]["end"] + 2.0,
                        "start": self.naughtyWordList[-1]["end"] + 1.0,
                        "word": "mothaflippin",
                        "scrub": True,
                    }
                ]
            )
        if self.debug:
            mmguero.eprint(self.naughtyWordList)

        self.muteTimeList = []
        self.sineTimeList = []
        self.beepDelayList = []
        for word, wordPeek in pairwise(self.naughtyWordList):
            wordStart = word["start"] - self.padSecPre
            wordEnd = word["end"] + self.padSecPost
            wordPeekStart = wordPeek["start"] - self.padSecPre
            
            if self.beep:
                # Use utilities module to build beep entries
                mute_entry, sine_entry, delay_entry = AudioFilterBuilder.create_beep_entries(
                    wordStart, wordEnd, self.beepHertz
                )
                self.muteTimeList.append(mute_entry)
                self.sineTimeList.append(sine_entry)
                self.beepDelayList.append(delay_entry)
            else:
                # Use utilities module to build mute entries
                fade_out, fade_in = AudioFilterBuilder.create_mute_time_entry(
                    wordStart, wordEnd, wordPeekStart
                )
                self.muteTimeList.append(fade_out)
                self.muteTimeList.append(fade_in)

        if self.debug:
            mmguero.eprint(self.muteTimeList)
            if self.beep:
                mmguero.eprint(self.sineTimeList)
                mmguero.eprint(self.beepDelayList)

        return self.muteTimeList

    ######## EncodeCleanAudio ####################################################
    def EncodeCleanAudio(self):
        if (self.forceDespiteTag is True) or (GetMonkeyplugTagged(self.inputFileSpec, debug=self.debug) is False):
            # Check if we should use chunking for large files
            if self.useChunking and AudioChunker:
                chunker = AudioChunker(
                    working_dir=self.chunkingWorkDir,
                    plugger=self,
                    parallel_encoding=self.parallelEncoding,
                    max_workers=self.maxWorkers
                )
                if chunker.needs_chunking(self.inputFileSpec):
                    if self.debug:
                        mmguero.eprint("File exceeds size threshold, using chunked processing")
                    chunker.process_with_chunking(
                        source_file=self.inputFileSpec,
                        output_file=self.outputFileSpec
                    )
                    SetMonkeyplugTag(self.outputFileSpec, debug=self.debug)
                    return self.outputFileSpec
            
            # Normal (non-chunked) processing
            self.CreateCleanMuteList()

            # Build audio filter arguments using utilities module
            if len(self.muteTimeList) > 0:
                if self.beep:
                    audioArgs = AudioFilterBuilder.build_beep_filters(
                        self.muteTimeList,
                        self.sineTimeList,
                        self.beepDelayList,
                        mix_normalize=self.beepMixNormalize,
                        audio_weight=self.beepAudioWeight,
                        sine_weight=self.beepSineWeight,
                        dropout_transition=self.beepDropTransition
                    )
                else:
                    audioArgs = AudioFilterBuilder.build_mute_filters(self.muteTimeList)
            else:
                audioArgs = []

            # Use utilities module to run ffmpeg command
            FFmpegRunner.run_encode(
                input_file=self.inputFileSpec,
                output_file=self.outputFileSpec,
                audio_params=self.aParams,
                audio_args=audioArgs,
                video_mode=bool(self.outputVideoFileFormat),
                debug=self.debug
            )

            SetMonkeyplugTag(self.outputFileSpec, debug=self.debug)

        else:
            shutil.copyfile(self.inputFileSpec, self.outputFileSpec)

        return self.outputFileSpec


#################################################################################


#################################################################################
class VoskPlugger(Plugger):
    tmpWavFileSpec = ""
    modelPath = ""
    wavReadFramesChunk = AUDIO_DEFAULT_WAV_FRAMES_CHUNK
    vosk = None

    def __init__(
        self,
        iFileSpec,
        oFileSpec,
        oAudioFileFormat,
        iSwearsFileSpec,
        mDir,
        outputJson,
        inputTranscript=None,
        saveTranscript=False,
        aParams=None,
        aChannels=AUDIO_DEFAULT_CHANNELS,
        aSampleRate=AUDIO_DEFAULT_SAMPLE_RATE,
        wChunk=AUDIO_DEFAULT_WAV_FRAMES_CHUNK,
        padMsecPre=0,
        padMsecPost=0,
        beep=False,
        beepHertz=BEEP_HERTZ_DEFAULT,
        beepMixNormalize=BEEP_MIX_NORMALIZE_DEFAULT,
        beepAudioWeight=BEEP_AUDIO_WEIGHT_DEFAULT,
        beepSineWeight=BEEP_SINE_WEIGHT_DEFAULT,
        beepDropTransition=BEEP_DROPOUT_TRANSITION_DEFAULT,
        confidenceThreshold=CONFIDENCE_THRESHOLD_DEFAULT,
        force=False,
        useChunking=False,
        chunkingWorkDir=None,
        parallelEncoding=False,
        maxWorkers=None,
        dbug=False,
    ):
        self.wavReadFramesChunk = wChunk
        self.modelPath = None
        self.vosk = None

        # Only load model if we're actually going to transcribe (no input transcript provided)
        if not inputTranscript:
            # make sure the VOSK model path exists
            if (mDir is not None) and os.path.isdir(mDir):
                self.modelPath = mDir
            else:
                raise IOError(
                    errno.ENOENT,
                    os.strerror(errno.ENOENT) + " (see https://alphacephei.com/vosk/models)",
                    mDir,
                )

            self.vosk = mmguero.dynamic_import("vosk", "vosk", debug=dbug)
            if not self.vosk:
                raise Exception(f"Unable to initialize VOSK API")
            if not dbug:
                self.vosk.SetLogLevel(-1)

        super().__init__(
            iFileSpec=iFileSpec,
            oFileSpec=oFileSpec,
            oAudioFileFormat=oAudioFileFormat,
            iSwearsFileSpec=iSwearsFileSpec,
            outputJson=outputJson,
            inputTranscript=inputTranscript,
            saveTranscript=saveTranscript,
            aParams=aParams,
            aChannels=aChannels,
            aSampleRate=aSampleRate,
            padMsecPre=padMsecPre,
            padMsecPost=padMsecPost,
            beep=beep,
            beepHertz=beepHertz,
            beepMixNormalize=beepMixNormalize,
            beepAudioWeight=beepAudioWeight,
            beepSineWeight=beepSineWeight,
            beepDropTransition=beepDropTransition,
            confidenceThreshold=confidenceThreshold,
            force=force,
            useChunking=useChunking,
            chunkingWorkDir=chunkingWorkDir,
            parallelEncoding=parallelEncoding,
            maxWorkers=maxWorkers,
            dbug=dbug,
        )

        self.tmpWavFileSpec = self.inputFileParts[0] + ".wav"

        if self.debug:
            if inputTranscript:
                mmguero.eprint(f'Using input transcript (skipping speech recognition)')
            else:
                mmguero.eprint(f'Model directory: {self.modelPath}')
                mmguero.eprint(f'Intermediate audio file: {self.tmpWavFileSpec}')
                mmguero.eprint(f'Read frames: {self.wavReadFramesChunk}')

    def __del__(self):
        super().__del__()
        # clean up intermediate WAV file used for speech recognition
        if os.path.isfile(self.tmpWavFileSpec):
            os.remove(self.tmpWavFileSpec)

    def CreateIntermediateWAV(self):
        """Create intermediate WAV file for VOSK speech recognition."""
        ffmpegCmd = FFmpegCommandBuilder.build_intermediate_wav_command(
            self.inputFileSpec,
            self.tmpWavFileSpec,
            AUDIO_INTERMEDIATE_PARAMS
        )
        ffmpegResult, ffmpegOutput = FFmpegRunner.run_command(ffmpegCmd, debug=self.debug)
        if (ffmpegResult != 0) or (not os.path.isfile(self.tmpWavFileSpec)):
            mmguero.eprint(' '.join(mmguero.flatten(ffmpegCmd)))
            mmguero.eprint(ffmpegResult)
            mmguero.eprint(ffmpegOutput)
            raise ValueError(
                f"Could not convert {self.inputFileSpec} to {self.tmpWavFileSpec} (16 kHz, mono, s16 PCM WAV)"
            )
        return self.inputFileSpec

    def RecognizeSpeech(self):
        self.CreateIntermediateWAV()
        self.wordList.clear()
        with wave.open(self.tmpWavFileSpec, "rb") as wf:
            if (
                (wf.getnchannels() != 1)
                or (wf.getframerate() != 16000)
                or (wf.getsampwidth() != 2)
                or (wf.getcomptype() != "NONE")
            ):
                raise Exception(f"Audio file ({self.tmpWavFileSpec}) must be 16 kHz, mono, s16 PCM WAV")

            rec = self.vosk.KaldiRecognizer(self.vosk.Model(self.modelPath), wf.getframerate())
            rec.SetWords(True)
            while True:
                data = wf.readframes(self.wavReadFramesChunk)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if "result" in res:
                        self.wordList.extend(
                            [
                                dict(r, **{'scrub': self._should_scrub_word(
                                    mmguero.deep_get(r, ["word"]), 
                                    mmguero.deep_get(r, ["conf"], 1.0)
                                )})
                                for r in res["result"]
                            ]
                        )
            res = json.loads(rec.FinalResult())
            if "result" in res:
                self.wordList.extend(
                    [
                        dict(r, **{'scrub': self._should_scrub_word(
                            mmguero.deep_get(r, ["word"]), 
                            mmguero.deep_get(r, ["conf"], 1.0)
                        )})
                        for r in res["result"]
                    ]
                )

            if self.debug:
                mmguero.eprint(json.dumps(self.wordList))

            if self.outputJson:
                TranscriptManager.save_transcript(
                    transcript_path=self.outputJson,
                    word_list=self.wordList,
                    debug=self.debug
                )

        return self.wordList


#################################################################################


#################################################################################
class WhisperPlugger(Plugger):
    debug = False
    model = None
    torch = None
    whisper = None
    transcript = None
    remote_url = None
    api_timeout = 600
    poll_interval = 5

    def __init__(
        self,
        iFileSpec,
        oFileSpec,
        oAudioFileFormat,
        iSwearsFileSpec,
        mDir,
        mName,
        torchThreads,
        outputJson,
        inputTranscript=None,
        saveTranscript=False,
        remoteUrl=None,
        apiTimeout=600,
        pollInterval=5,
        aParams=None,
        aChannels=AUDIO_DEFAULT_CHANNELS,
        aSampleRate=AUDIO_DEFAULT_SAMPLE_RATE,
        padMsecPre=0,
        padMsecPost=0,
        beep=False,
        beepHertz=BEEP_HERTZ_DEFAULT,
        beepMixNormalize=BEEP_MIX_NORMALIZE_DEFAULT,
        beepAudioWeight=BEEP_AUDIO_WEIGHT_DEFAULT,
        beepSineWeight=BEEP_SINE_WEIGHT_DEFAULT,
        beepDropTransition=BEEP_DROPOUT_TRANSITION_DEFAULT,
        confidenceThreshold=CONFIDENCE_THRESHOLD_DEFAULT,
        force=False,
        useChunking=False,
        chunkingWorkDir=None,
        parallelEncoding=False,
        maxWorkers=None,
        dbug=False,
    ):
        # Handle remote URL - add http:// if no scheme provided
        if remoteUrl:
            original_url = remoteUrl
            remoteUrl = remoteUrl.rstrip('/')
            if not remoteUrl.startswith(('http://', 'https://')):
                remoteUrl = f'http://{remoteUrl}'
                if dbug:
                    mmguero.eprint(f'Adding http:// to URL: {original_url} -> {remoteUrl}')
        self.remote_url = remoteUrl
        self.api_timeout = apiTimeout
        self.poll_interval = pollInterval
        self.whisper = None
        self.model = None
        self.torch = None

        # Only load model if we're actually going to transcribe (no input transcript provided)
        if not inputTranscript:
            # Only load local model if not using remote
            if not self.remote_url:
                if torchThreads > 0:
                    self.torch = mmguero.dynamic_import("torch", "torch", debug=dbug)
                    if self.torch:
                        self.torch.set_num_threads(torchThreads)

                self.whisper = mmguero.dynamic_import("whisper", "openai-whisper", debug=dbug)
                if not self.whisper:
                    raise Exception("Unable to initialize Whisper API")

                self.model = self.whisper.load_model(mName, download_root=mDir)
                if not self.model:
                    raise Exception(f"Unable to load Whisper model {mName} in {mDir}")

        super().__init__(
            iFileSpec=iFileSpec,
            oFileSpec=oFileSpec,
            oAudioFileFormat=oAudioFileFormat,
            iSwearsFileSpec=iSwearsFileSpec,
            outputJson=outputJson,
            inputTranscript=inputTranscript,
            saveTranscript=saveTranscript,
            aParams=aParams,
            aChannels=aChannels,
            aSampleRate=aSampleRate,
            padMsecPre=padMsecPre,
            padMsecPost=padMsecPost,
            beep=beep,
            beepHertz=beepHertz,
            beepMixNormalize=beepMixNormalize,
            beepAudioWeight=beepAudioWeight,
            beepSineWeight=beepSineWeight,
            beepDropTransition=beepDropTransition,
            confidenceThreshold=confidenceThreshold,
            force=force,
            useChunking=useChunking,
            chunkingWorkDir=chunkingWorkDir,
            parallelEncoding=parallelEncoding,
            maxWorkers=maxWorkers,
            dbug=dbug,
        )

        if self.debug:
            if inputTranscript:
                mmguero.eprint(f'Using input transcript (skipping speech recognition)')
            elif self.remote_url:
                mmguero.eprint(f'Remote Whisper URL: {self.remote_url}')
                mmguero.eprint(f'API Timeout: {self.api_timeout}')
                mmguero.eprint(f'Poll Interval: {self.poll_interval}')
            else:
                mmguero.eprint(f'Model directory: {mDir}')
                mmguero.eprint(f'Model name: {mName}')

    def __del__(self):
        super().__del__()

    def RecognizeSpeech(self):
        self.wordList.clear()

        if self.remote_url:
            self._RecognizeSpeechRemote()
        else:
            self._RecognizeSpeechLocal()

        if self.debug:
            mmguero.eprint(json.dumps(self.wordList))

        if self.outputJson:
            TranscriptManager.save_transcript(
                transcript_path=self.outputJson,
                word_list=self.wordList,
                debug=self.debug
            )

        return self.wordList

    def _RecognizeSpeechLocal(self):
        """Local Whisper transcription"""
        self.transcript = self.model.transcribe(word_timestamps=True, audio=self.inputFileSpec)
        if self.transcript and ('segments' in self.transcript):
            for segment in self.transcript['segments']:
                if 'words' in segment:
                    for word in segment['words']:
                        word['word'] = word['word'].strip()
                        # Whisper provides 'probability' field for confidence
                        word['scrub'] = self._should_scrub_word(
                            word['word'], 
                            word.get('probability', 1.0)
                        )
                        self.wordList.append(word)

    def _RecognizeSpeechRemote(self):
        """Remote Whisper transcription using async task pattern"""
        try:
            task_id = self._upload_audio_for_transcription()
            result = self._poll_for_transcription_result(task_id)
            self._extract_words_from_result(result)
        except requests.exceptions.RequestException as e:
            mmguero.eprint(f"Error communicating with remote service: {e}")
            raise

    def _upload_audio_for_transcription(self):
        """Upload audio file to remote service and get task ID"""
        with open(self.inputFileSpec, 'rb') as f:
            files = {'file': f}
            data = {'word_timestamps': True}
            
            if self.debug:
                mmguero.eprint(f'Uploading to {self.remote_url}/transcription/')
            
            response = requests.post(
                f'{self.remote_url}/transcription/',
                files=files,
                data=data,
                timeout=self.api_timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            task_id = response_data.get('identifier')
            if not task_id:
                raise ValueError(f"No task identifier in response: {response_data}")
            
            if self.debug:
                mmguero.eprint(f'Task ID: {task_id}')
            
            return task_id

    def _poll_for_transcription_result(self, task_id):
        """Poll remote service until transcription completes and return result"""
        import time
        
        while True:
            response = requests.get(f'{self.remote_url}/task/{task_id}')
            response.raise_for_status()
            data = response.json()
            
            status = data.get('status')
            
            if self.debug:
                progress = data.get('progress', 0)
                mmguero.eprint(f'Status: {status}, Progress: {progress:.1%}')
            
            if status == 'completed':
                return data.get('result', [])
            elif status in ['failed', 'error']:
                error_msg = data.get('error', 'Unknown error')
                raise RuntimeError(f'Transcription failed: {error_msg}')
            
            time.sleep(self.poll_interval)

    def _extract_words_from_result(self, result):
        """Extract words from transcription result and add to word list"""
        if not isinstance(result, list) or len(result) == 0:
            return
        
        # Check if result has word-level timestamps (native Whisper format)
        has_word_timestamps = 'words' in result[0] if result else False
        
        if has_word_timestamps:
            if self.debug:
                mmguero.eprint('Using native Whisper word-level timestamps')
            for segment in result:
                if 'words' in segment:
                    for word in segment['words']:
                        word_text = word.get('word', '').strip()
                        if word_text:
                            word_conf = word.get('probability', 1.0)
                            self.wordList.append({
                                'word': word_text,
                                'start': word.get('start', 0),
                                'end': word.get('end', 0),
                                'conf': word_conf,
                                'scrub': self._should_scrub_word(word_text, word_conf)
                            })
        else:
            # Fallback: estimate word timing from segment-level timestamps
            if self.debug:
                mmguero.eprint('Word timestamps not available, estimating from segments')
            for segment in result:
                text = segment.get('text', '').strip()
                if not text:
                    continue
                
                words = text.split()
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                duration = end - start
                word_duration = duration / len(words) if words else 0
                
                for idx, word_text in enumerate(words):
                    word_start = start + (idx * word_duration)
                    word_end = word_start + word_duration
                    self.wordList.append({
                        'word': word_text,
                        'start': word_start,
                        'end': word_end,
                        'conf': 1.0,
                        'scrub': self._should_scrub_word(word_text, 1.0)
                    })

# RunMonkeyPlug
def RunMonkeyPlug():
    parser = argparse.ArgumentParser(
        description=script_name,
        add_help=False,
        usage="{} <arguments>".format(script_name),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="debug",
        type=mmguero.str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help="Verbose/debug output",
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="speechRecMode",
        metavar="<string>",
        type=str,
        default=DEFAULT_SPEECH_REC_MODE,
        help=f"Speech recognition engine ({SPEECH_REC_MODE_WHISPER}|{SPEECH_REC_MODE_VOSK}|{SPEECH_REC_MODE_REMOTE_WHISPER}) (default: {DEFAULT_SPEECH_REC_MODE})",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        type=str,
        default=None,
        required=True,
        metavar="<string>",
        help="Input file (or URL)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default=None,
        required=False,
        metavar="<string>",
        help="Output file",
    )
    parser.add_argument(
        "--output-json",
        dest="outputJson",
        type=str,
        default=None,
        required=False,
        metavar="<string>",
        help="Output file to store transcript JSON",
    )
    parser.add_argument(
        "--input-transcript",
        dest="inputTranscript",
        type=str,
        default=None,
        required=False,
        metavar="<string>",
        help="Load existing transcript JSON instead of performing speech recognition",
    )
    parser.add_argument(
        "--save-transcript",
        dest="saveTranscript",
        action="store_true",
        default=False,
        help="Automatically save transcript JSON alongside output audio file (default: true)",
    )
    parser.add_argument(
        "-w",
        "--swears",
        help=f"text file containing profanity (default: \"{SWEARS_FILENAME_DEFAULT}\")",
        default=os.path.join(script_path, SWEARS_FILENAME_DEFAULT),
        metavar="<profanity file>",
    )
    parser.add_argument(
        "--confidence-threshold",
        dest="confidenceThreshold",
        metavar="<float>",
        type=float,
        default=CONFIDENCE_THRESHOLD_DEFAULT,
        help=f"Minimum confidence level (0.0-1.0) required to censor a word (default: {CONFIDENCE_THRESHOLD_DEFAULT})",
    )
    parser.add_argument(
        "-a",
        "--audio-params",
        help=f"Audio parameters for ffmpeg (default depends on output audio codec)",
        dest="aParams",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--channels",
        dest="aChannels",
        metavar="<int>",
        type=int,
        default=AUDIO_DEFAULT_CHANNELS,
        help=f"Audio output channels (default: {AUDIO_DEFAULT_CHANNELS})",
    )
    parser.add_argument(
        "-s",
        "--sample-rate",
        dest="aSampleRate",
        metavar="<int>",
        type=int,
        default=AUDIO_DEFAULT_SAMPLE_RATE,
        help=f"Audio output sample rate (default: {AUDIO_DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="outputFormat",
        type=str,
        default=AUDIO_MATCH_FORMAT,
        required=False,
        metavar="<string>",
        help=f"Output file format (default: inferred from extension of --output, or \"{AUDIO_MATCH_FORMAT}\")",
    )
    parser.add_argument(
        "--pad-milliseconds",
        dest="padMsec",
        metavar="<int>",
        type=int,
        default=0,
        help=f"Milliseconds to pad on either side of muted segments (default: 0)",
    )
    parser.add_argument(
        "--pad-milliseconds-pre",
        dest="padMsecPre",
        metavar="<int>",
        type=int,
        default=0,
        help=f"Milliseconds to pad before muted segments (default: 0)",
    )
    parser.add_argument(
        "--pad-milliseconds-post",
        dest="padMsecPost",
        metavar="<int>",
        type=int,
        default=0,
        help=f"Milliseconds to pad after muted segments (default: 0)",
    )
    parser.add_argument(
        "-b",
        "--beep",
        dest="beep",
        type=mmguero.str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help="Beep instead of silence",
    )
    parser.add_argument(
        "-h",
        "--beep-hertz",
        dest="beepHertz",
        metavar="<int>",
        type=int,
        default=BEEP_HERTZ_DEFAULT,
        help=f"Beep frequency hertz (default: {BEEP_HERTZ_DEFAULT})",
    )
    parser.add_argument(
        "--beep-mix-normalize",
        dest="beepMixNormalize",
        type=mmguero.str2bool,
        nargs="?",
        const=True,
        default=BEEP_MIX_NORMALIZE_DEFAULT,
        metavar="true|false",
        help=f"Normalize mix of audio and beeps (default: {BEEP_MIX_NORMALIZE_DEFAULT})",
    )
    parser.add_argument(
        "--beep-audio-weight",
        dest="beepAudioWeight",
        metavar="<int>",
        type=int,
        default=BEEP_AUDIO_WEIGHT_DEFAULT,
        help=f"Mix weight for non-beeped audio (default: {BEEP_AUDIO_WEIGHT_DEFAULT})",
    )
    parser.add_argument(
        "--beep-sine-weight",
        dest="beepSineWeight",
        metavar="<int>",
        type=int,
        default=BEEP_SINE_WEIGHT_DEFAULT,
        help=f"Mix weight for beep (default: {BEEP_SINE_WEIGHT_DEFAULT})",
    )
    parser.add_argument(
        "--beep-dropout-transition",
        dest="beepDropTransition",
        metavar="<int>",
        type=int,
        default=BEEP_DROPOUT_TRANSITION_DEFAULT,
        help=f"Dropout transition for beep (default: {BEEP_DROPOUT_TRANSITION_DEFAULT})",
    )

    parser.add_argument(
        "--force",
        dest="forceDespiteTag",
        type=mmguero.str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help="Process file despite existence of embedded tag",
    )

    chunkingArgGroup = parser.add_argument_group('Chunking Options')
    chunkingArgGroup.add_argument(
        "--use-chunking",
        dest="useChunking",
        type=mmguero.str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help="Enable audio chunking for large files (>150MB)",
    )
    chunkingArgGroup.add_argument(
        "--chunking-work-dir",
        dest="chunkingWorkDir",
        metavar="<string>",
        type=str,
        default=None,
        help="Working directory for audio chunks (default: same as input file)",
    )
    chunkingArgGroup.add_argument(
        "--parallel-encoding",
        dest="parallelEncoding",
        type=mmguero.str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help="Enable parallel encoding after serial transcription (requires --use-chunking)",
    )
    chunkingArgGroup.add_argument(
        "--max-workers",
        dest="maxWorkers",
        metavar="<int>",
        type=int,
        default=None,
        help="Maximum number of parallel workers for encoding (default: CPU count, only with --parallel-encoding)",
    )

    voskArgGroup = parser.add_argument_group('VOSK Options')
    voskArgGroup.add_argument(
        "--vosk-model-dir",
        dest="voskModelDir",
        metavar="<string>",
        type=str,
        default=DEFAULT_VOSK_MODEL_DIR,
        help=f"VOSK model directory (default: {DEFAULT_VOSK_MODEL_DIR})",
    )
    voskArgGroup.add_argument(
        "--vosk-read-frames-chunk",
        dest="voskReadFramesChunk",
        metavar="<int>",
        type=int,
        default=os.getenv("VOSK_READ_FRAMES", AUDIO_DEFAULT_WAV_FRAMES_CHUNK),
        help=f"WAV frame chunk (default: {AUDIO_DEFAULT_WAV_FRAMES_CHUNK})",
    )

    whisperArgGroup = parser.add_argument_group('Whisper Options')
    whisperArgGroup.add_argument(
        "--whisper-model-dir",
        dest="whisperModelDir",
        metavar="<string>",
        type=str,
        default=DEFAULT_WHISPER_MODEL_DIR,
        help=f"Whisper model directory ({DEFAULT_WHISPER_MODEL_DIR})",
    )
    whisperArgGroup.add_argument(
        "--whisper-model-name",
        dest="whisperModelName",
        metavar="<string>",
        type=str,
        default=DEFAULT_WHISPER_MODEL_NAME,
        help=f"Whisper model name ({DEFAULT_WHISPER_MODEL_NAME})",
    )
    whisperArgGroup.add_argument(
        "--torch-threads",
        dest="torchThreads",
        metavar="<int>",
        type=int,
        default=DEFAULT_TORCH_THREADS,
        help=f"Number of threads used by torch for CPU inference ({DEFAULT_TORCH_THREADS})",
    )

    remoteWhisperArgGroup = parser.add_argument_group('Remote Whisper Options')
    remoteWhisperArgGroup.add_argument(
        "--remote-whisper-url",
        dest="remoteWhisperUrl",
        metavar="<string>",
        type=str,
        default=os.getenv("REMOTE_WHISPER_URL", None),
        help="Remote Whisper service URL (e.g., http://localhost:8000)",
    )
    remoteWhisperArgGroup.add_argument(
        "--remote-whisper-timeout",
        dest="remoteWhisperTimeout",
        metavar="<int>",
        type=int,
        default=600,
        help="Timeout for remote API requests in seconds (default: 600)",
    )
    remoteWhisperArgGroup.add_argument(
        "--remote-whisper-poll-interval",
        dest="remoteWhisperPollInterval",
        metavar="<int>",
        type=int,
        default=5,
        help="Poll interval for checking remote task status in seconds (default: 5)",
    )

    try:
        parser.error = parser.exit
        args = parser.parse_args()
    except SystemExit as sy:
        mmguero.eprint(sy)
        parser.print_help()
        exit(2)

    if args.debug:
        mmguero.eprint(os.path.join(script_path, script_name))
        mmguero.eprint("Arguments: {}".format(sys.argv[1:]))
        mmguero.eprint("Arguments: {}".format(args))
    else:
        sys.tracebacklimit = 0

    if args.speechRecMode == SPEECH_REC_MODE_VOSK:
        pathlib.Path(args.voskModelDir).mkdir(parents=True, exist_ok=True)
        plug = VoskPlugger(
            args.input,
            args.output,
            args.outputFormat,
            args.swears,
            args.voskModelDir,
            args.outputJson,
            inputTranscript=args.inputTranscript,
            saveTranscript=args.saveTranscript,
            aParams=args.aParams,
            aChannels=args.aChannels,
            aSampleRate=args.aSampleRate,
            wChunk=args.voskReadFramesChunk,
            padMsecPre=args.padMsecPre if args.padMsecPre > 0 else args.padMsec,
            padMsecPost=args.padMsecPost if args.padMsecPost > 0 else args.padMsec,
            beep=args.beep,
            beepHertz=args.beepHertz,
            beepMixNormalize=args.beepMixNormalize,
            beepAudioWeight=args.beepAudioWeight,
            beepSineWeight=args.beepSineWeight,
            beepDropTransition=args.beepDropTransition,
            confidenceThreshold=args.confidenceThreshold,
            force=args.forceDespiteTag,
            useChunking=args.useChunking,
            chunkingWorkDir=args.chunkingWorkDir,
            parallelEncoding=args.parallelEncoding,
            maxWorkers=args.maxWorkers,
            dbug=args.debug,
        )

    elif args.speechRecMode in [SPEECH_REC_MODE_WHISPER, SPEECH_REC_MODE_REMOTE_WHISPER]:
        # Use remote if mode is remote-whisper OR if remote URL is provided
        use_remote = (args.speechRecMode == SPEECH_REC_MODE_REMOTE_WHISPER) or args.remoteWhisperUrl
        
        if use_remote:
            if not args.remoteWhisperUrl:
                raise ValueError("Remote Whisper URL must be specified with --remote-whisper-url or REMOTE_WHISPER_URL environment variable")
            remote_url = args.remoteWhisperUrl
        else:
            pathlib.Path(args.whisperModelDir).mkdir(parents=True, exist_ok=True)
            remote_url = None
        
        plug = WhisperPlugger(
            args.input,
            args.output,
            args.outputFormat,
            args.swears,
            args.whisperModelDir,
            args.whisperModelName,
            args.torchThreads,
            args.outputJson,
            inputTranscript=args.inputTranscript,
            saveTranscript=args.saveTranscript,
            remoteUrl=remote_url,
            apiTimeout=args.remoteWhisperTimeout,
            pollInterval=args.remoteWhisperPollInterval,
            aParams=args.aParams,
            aChannels=args.aChannels,
            aSampleRate=args.aSampleRate,
            padMsecPre=args.padMsecPre if args.padMsecPre > 0 else args.padMsec,
            padMsecPost=args.padMsecPost if args.padMsecPost > 0 else args.padMsec,
            beep=args.beep,
            beepHertz=args.beepHertz,
            beepMixNormalize=args.beepMixNormalize,
            beepAudioWeight=args.beepAudioWeight,
            beepSineWeight=args.beepSineWeight,
            beepDropTransition=args.beepDropTransition,
            confidenceThreshold=args.confidenceThreshold,
            force=args.forceDespiteTag,
            useChunking=args.useChunking,
            chunkingWorkDir=args.chunkingWorkDir,
            parallelEncoding=args.parallelEncoding,
            maxWorkers=args.maxWorkers,
            dbug=args.debug,
        )
    
    else:
        raise ValueError(f"Unsupported speech recognition engine {args.speechRecMode}")

    print(plug.EncodeCleanAudio())


###################################################################################################
if __name__ == "__main__":
    RunMonkeyPlug()
