from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel

LANGUAGE = "fr"
INT16_MAX = 32768.0


class SpeechTranscriber(Protocol):
    """Tout ce qui transforme de l'audio PCM (voir audio_format.py) en texte.

    Whisper local aujourd'hui ; Deepgram ou Speechmatics demain, sans toucher au reste.
    L'appel est bloquant : c'est TranscriptionService qui le sort de la boucle asyncio.
    """

    def transcribe(self, pcm: bytes) -> str: ...


class WhisperTranscriber:
    """STT local avec faster-whisper (Whisper compilé pour CPU via CTranslate2)."""

    def __init__(self, model_size: str) -> None:
        # int8 : poids quantifiés en entiers 8 bits. 2 à 4 fois plus rapide sur CPU qu'en
        # float32 pour une perte de qualité faible. Sans GPU, c'est le bon compromis.
        # Le chargement télécharge le modèle au premier lancement (puis cache local).
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, pcm: bytes) -> str:
        # Whisper attend des float32 dans [-1, 1], alors que notre pipeline est en int16.
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / INT16_MAX

        segments, _info = self._model.transcribe(
            samples,
            language=LANGUAGE,  # évite la détection de langue : plus rapide et plus fiable
            beam_size=1,  # décodage glouton : le plus rapide, qualité correcte
            vad_filter=False,  # notre propre VAD a déjà découpé le tour
            # Chaque tour est indépendant : réutiliser le texte précédent comme contexte
            # propage les erreurs et favorise les hallucinations.
            condition_on_previous_text=False,
        )
        # `segments` est un générateur PARESSEUX : le calcul n'a lieu qu'à l'itération.
        return " ".join(segment.text.strip() for segment in segments).strip()
