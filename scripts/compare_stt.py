"""Compare plusieurs réglages du STT sur les mêmes fichiers WAV.

Usage :
    uv run python -m scripts.compare_stt recordings/ --models base,small --beams 1,5
    uv run python -m scripts.compare_stt recordings/x.wav --prompt "Rue des Lilas, Villeurbanne."

Les WAV doivent être au format du pipeline (16 kHz, mono, 16 bits) : c'est le cas de ceux
écrits par le serveur avec VOC_RECORD_DIR=recordings.
"""

import argparse
import time
import wave
from pathlib import Path

from voc_ai.audio_format import BYTES_PER_SAMPLE, SAMPLE_RATE
from voc_ai.transcriber import WhisperTranscriber


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, BYTES_PER_SAMPLE):
            raise ValueError(f"{path} : format attendu {SAMPLE_RATE} Hz mono 16 bits")
        return wav.readframes(wav.getnframes())


def list_wavs(target: Path) -> list[Path]:
    return sorted(target.glob("*.wav")) if target.is_dir() else [target]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="fichier WAV ou dossier de WAV")
    parser.add_argument("--models", default="base", help="tailles Whisper séparées par des virgules")
    parser.add_argument("--beams", default="5", help="valeurs de beam_size séparées par des virgules")
    parser.add_argument("--prompt", default=None, help="vocabulaire attendu (initial_prompt)")
    args = parser.parse_args()

    recordings = [(path, read_pcm(path)) for path in list_wavs(args.target)]
    if not recordings:
        parser.error(f"aucun WAV dans {args.target}")

    for model_size in args.models.split(","):
        for beam_size in map(int, args.beams.split(",")):
            transcriber = WhisperTranscriber(model_size, beam_size=beam_size, initial_prompt=args.prompt)
            print(f"\n=== {model_size}, beam={beam_size}, prompt={'oui' if args.prompt else 'non'} ===")

            for path, pcm in recordings:
                started = time.perf_counter()
                transcript = transcriber.transcribe(pcm)
                elapsed = time.perf_counter() - started

                print(f"  {path.name}  {elapsed:5.2f} s | {transcript.text}")
                for rejected in transcript.rejected:
                    print(f"  {'':{len(path.name)}}  écarté | {rejected}")


if __name__ == "__main__":
    main()
