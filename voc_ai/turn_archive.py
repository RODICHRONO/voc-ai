import wave
from datetime import datetime
from pathlib import Path

from voc_ai.audio_format import BYTES_PER_SAMPLE, SAMPLE_RATE
from voc_ai.turn_recorder import CompletedTurn


class TurnArchive:
    """Enregistre l'audio de chaque tour en WAV, pour le diagnostic.

    Écouter ce que le serveur a VRAIMENT reçu tranche la question "est-ce le son ou le STT ?".
    Les fichiers se rejouent ensuite avec scripts/compare_stt.py pour comparer des réglages
    sur exactement le même audio. C'est la base d'un banc de test sur appels réels.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        # Préfixe commun aux tours d'une même connexion : start_ms repart de 0 à chaque fois.
        self._session_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")

    def save(self, turn: CompletedTurn) -> Path:
        path = self._directory / f"{self._session_prefix}-{turn.start_ms:07d}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(BYTES_PER_SAMPLE)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(turn.audio)
        return path
