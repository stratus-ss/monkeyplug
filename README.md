# monkeyplug

[![Latest Version](https://img.shields.io/pypi/v/monkeyplug)](https://pypi.python.org/pypi/monkeyplug/) [![VOSK Docker Images](https://github.com/mmguero/monkeyplug/workflows/monkeyplug-build-push-vosk-ghcr/badge.svg)](https://github.com/mmguero/monkeyplug/pkgs/container/monkeyplug) [![Whisper Docker Images](https://github.com/mmguero/monkeyplug/workflows/monkeyplug-build-push-whisper-ghcr/badge.svg)](https://github.com/mmguero/monkeyplug/pkgs/container/monkeyplug)

**monkeyplug** is a little script to censor profanity in audio files (intended for podcasts, but YMMV) in a few simple steps:

1. The user provides a local audio file (or a URL pointing to an audio file which is downloaded)
2. Speech recognition is performed using one of the following engines:
   - [Whisper](https://openai.com/research/whisper) ([GitHub](https://github.com/openai/whisper)) - Local Whisper processing
   - [Vosk](https://alphacephei.com/vosk/)-[API](https://github.com/alphacep/vosk-api) - Local Vosk processing
   - [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI) - Remote transcription service (tested and supported)
3. Each recognized word is checked against a [list](./src/monkeyplug/swears.txt) of profanity or other words you'd like muted (supports text or [JSON format](./SWEARS_JSON_FORMAT.md))
4. Words are only censored if the speech recognition confidence level meets or exceeds the threshold (default: 65%, configurable via `--confidence-threshold`)
5. [`ffmpeg`](https://www.ffmpeg.org/) is used to create a cleaned audio file, muting or "bleeping" the objectional words

You can then use your favorite media player to play the cleaned audio file.

If provided a video file for input, **monkeyplug** will attempt to process the audio stream from the file and remultiplex it, copying the original video stream. 

**monkeyplug** is part of a family of projects with similar goals:

* 📼 [cleanvid](https://github.com/mmguero/cleanvid) for video files (using [SRT-formatted](https://en.wikipedia.org/wiki/SubRip#Format) subtitles)
* 🎤 [monkeyplug](https://github.com/mmguero/monkeyplug) for audio and video files (using either [Whisper](https://openai.com/research/whisper) or the [Vosk](https://alphacephei.com/vosk/)-[API](https://github.com/alphacep/vosk-api) for speech recognition)
* 📕 [montag](https://github.com/mmguero/montag) for ebooks

## Installation

Using `pip`, to install the latest [release from PyPI](https://pypi.org/project/monkeyplug/):

```
python3 -m pip install -U monkeyplug
```

Or to install directly from GitHub:


```
python3 -m pip install -U 'git+https://github.com/mmguero/monkeyplug'
```

## Prerequisites

[monkeyplug](./src/monkeyplug/monkeyplug.py) requires:

* [FFmpeg](https://www.ffmpeg.org)
* Python 3
    - [mutagen](https://github.com/quodlibet/mutagen)
    - a speech recognition library, either of:
        + [Whisper](https://github.com/openai/whisper)
        + [vosk-api](https://github.com/alphacep/vosk-api) with a VOSK [compatible model](https://alphacephei.com/vosk/models)

To install FFmpeg, use your operating system's package manager or install binaries from [ffmpeg.org](https://www.ffmpeg.org/download.html). The Python dependencies will be installed automatically if you are using `pip` to install monkeyplug, except for [`vosk`](https://pypi.org/project/vosk/) or [`openai-whisper`](https://pypi.org/project/openai-whisper/); as monkeyplug can work with both speech recognition engines, there is not a hard installation requirement for either until runtime.

## usage

```
usage: monkeyplug.py <arguments>

monkeyplug.py

options:
  -v, --verbose [true|false]
                        Verbose/debug output
  -m, --mode <string>   Speech recognition engine (whisper|vosk|remote-whisper) (default: whisper)
  -i, --input <string>  Input file (or URL)
  -o, --output <string>
                        Output file
  --output-json <string>
                        Output file to store transcript JSON
  --input-transcript <string>
                        Load existing transcript JSON instead of performing speech recognition
  --save-transcript     Automatically save transcript JSON alongside output audio file
  -w, --swears <profanity file>
                        text or JSON file containing profanity (default: "swears.txt")
  --confidence-threshold <float>
                        Minimum confidence level (0.0-1.0) required to censor a word (default: 0.65)
  -a, --audio-params APARAMS
                        Audio parameters for ffmpeg (default depends on output audio codec)
  -c, --channels <int>  Audio output channels (default: 2)
  -s, --sample-rate <int>
                        Audio output sample rate (default: 48000)
  -f, --format <string>
                        Output file format (default: inferred from extension of --output, or "MATCH")
  --pad-milliseconds <int>
                        Milliseconds to pad on either side of muted segments (default: 0)
  --pad-milliseconds-pre <int>
                        Milliseconds to pad before muted segments (default: 0)
  --pad-milliseconds-post <int>
                        Milliseconds to pad after muted segments (default: 0)
  -b, --beep [true|false]
                        Beep instead of silence
  -h, --beep-hertz <int>
                        Beep frequency hertz (default: 1000)
  --beep-mix-normalize [true|false]
                        Normalize mix of audio and beeps (default: False)
  --beep-audio-weight <int>
                        Mix weight for non-beeped audio (default: 1)
  --beep-sine-weight <int>
                        Mix weight for beep (default: 1)
  --beep-dropout-transition <int>
                        Dropout transition for beep (default: 0)
  --force [true|false]  Process file despite existence of embedded tag

Chunking Options:
  --use-chunking [true|false]
                        Enable audio chunking for large files (>150MB)
  --chunking-work-dir <string>
                        Working directory for audio chunks (default: same as input file)
  --parallel-encoding [true|false]
                        Enable parallel encoding after serial transcription (requires --use-chunking)
  --max-workers <int>   Maximum number of parallel workers for encoding (default: CPU count, only with --parallel-encoding)

VOSK Options:
  --vosk-model-dir <string>
                        VOSK model directory (default: ~/.cache/vosk)
  --vosk-read-frames-chunk <int>
                        WAV frame chunk (default: 8000)

Whisper Options:
  --whisper-model-dir <string>
                        Whisper model directory (~/.cache/whisper)
  --whisper-model-name <string>
                        Whisper model name (base.en)
  --torch-threads <int>
                        Number of threads used by torch for CPU inference (0)

Remote Whisper Options (for Whisper-WebUI):
  --remote-whisper-url <string>
                        Remote Whisper-WebUI service URL (e.g., http://localhost:8000)
  --remote-whisper-timeout <int>
                        Timeout for remote API requests in seconds (default: 600)
  --remote-whisper-poll-interval <int>
                        Poll interval for checking remote task status in seconds (default: 5)
```

### Using Remote Transcription with Whisper-WebUI

You can use a remote [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI) service instead of running Whisper locally. This is useful if:

- You want to offload processing to a more powerful machine
- You want to avoid installing large ML models locally
- You have a centralized transcription service in your infrastructure

**Note**: This feature is specifically designed for and tested with [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI). While it may work with other Whisper backends that provide compatible REST API endpoints, those have not been tested.

#### Using Remote Mode

Use the `--mode remote-whisper` flag along with `--remote-whisper-url`:

```bash
# Using command-line arguments (http:// is optional, will be added automatically)
monkeyplug.py --mode remote-whisper --remote-whisper-url http://localhost:8000 -i input.mp3 -o output.mp3

# Or without protocol (http:// will be added automatically)
monkeyplug.py --mode remote-whisper --remote-whisper-url localhost:8000 -i input.mp3 -o output.mp3

# Using environment variable
export REMOTE_WHISPER_URL=http://localhost:8000
monkeyplug.py --mode remote-whisper -i input.mp3 -o output.mp3
```

**Note**: If you don't specify `http://` or `https://`, the tool will automatically add `http://` to the URL.

### Transcript Workflow

Monkeyplug supports saving and reusing transcripts, which is useful for:

- **Faster reprocessing**: Transcribe once, then quickly test different swear lists or confidence thresholds
- **Iterative refinement**: Adjust your profanity list without waiting for re-transcription
- **Manual review**: Export transcripts to review and modify before processing

#### Saving Transcripts

```bash
# Automatically save transcript alongside output file
monkeyplug.py -i input.mp3 -o output.mp3 --save-transcript
# Creates: output_transcript.json

# Save transcript to specific location
monkeyplug.py -i input.mp3 -o output.mp3 --output-json my_transcript.json
```

#### Loading Pre-existing Transcripts

```bash
# Skip speech recognition and use existing transcript
monkeyplug.py -i input.mp3 -o output.mp3 --input-transcript my_transcript.json

# The transcript will be re-evaluated against your current swear list
# This is useful for:
# - Testing different swear lists
# - Adjusting confidence thresholds
# - Quick reprocessing without re-transcribing
```

**Note**: When loading a transcript, the words are re-evaluated against your current swear list and confidence threshold settings, so you can experiment with different filtering options without re-transcribing the audio.

### Large File Handling

Monkeyplug automatically handles large audio files (>150MB) using intelligent chunking:

- **Smart Splitting**: Automatically splits files at natural silence points rather than arbitrary timestamps
- **Metadata Preservation**: Extracts and restores chapter markers and other metadata
- **Chunk Reuse**: Caches split chunks and transcripts to avoid redundant processing during reruns
- **Processing Modes**:
  - **Serial** (default): Processes one chunk at a time (lower memory, slower)
  - **Parallel**: Transcribes serially, then encodes chunks in parallel (faster, higher memory)

The chunking system operates transparently—if your file exceeds 150MB, monkeyplug will automatically split, process, and reassemble it while preserving all metadata.

### Docker

Alternately, a [Dockerfile](./docker/Dockerfile) is provided to allow you to run monkeyplug in Docker. You can pull one of the following images:

* [VOSK](https://alphacephei.com/vosk/models)
    - oci.guero.org/monkeyplug:vosk-small
    - oci.guero.org/monkeyplug:vosk-large
* [Whisper](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages)
    - oci.guero.org/monkeyplug:whisper-tiny.en
    - oci.guero.org/monkeyplug:whisper-tiny
    - oci.guero.org/monkeyplug:whisper-base.en
    - oci.guero.org/monkeyplug:whisper-base
    - oci.guero.org/monkeyplug:whisper-small.en
    - oci.guero.org/monkeyplug:whisper-small
    - oci.guero.org/monkeyplug:whisper-medium.en
    - oci.guero.org/monkeyplug:whisper-medium
    - oci.guero.org/monkeyplug:whisper-large-v1
    - oci.guero.org/monkeyplug:whisper-large-v2
    - oci.guero.org/monkeyplug:whisper-large-v3
    - oci.guero.org/monkeyplug:whisper-large

then run [`monkeyplug-docker.sh`](./docker/monkeyplug-docker.sh) inside the directory where your audio files are located.

## Contributing

If you'd like to help improve monkeyplug, pull requests will be welcomed!

## Authors

* **Seth Grover** - *Initial work* - [mmguero](https://github.com/mmguero)

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
