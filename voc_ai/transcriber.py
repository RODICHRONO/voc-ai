from dataclasses import dataclass
from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel

from voc_ai.hallucination_filter import rejection_reason

LANGUAGE = "fr"
INT16_MAX = 32768.0


@dataclass(frozen=True)
class Transcript:
    text: str
    # Segments écartés, sous la forme "texte (raison)". Gardés pour les afficher : voir ce
    # que le filtre retire est le seul moyen de savoir s'il est trop strict ou pas assez.
    rejected: tuple[str, ...] = ()


class SpeechTranscriber(Protocol):
    """Tout ce qui transforme de l'audio PCM (voir audio_format.py) en texte.

    Whisper local aujourd'hui ; Deepgram ou Speechmatics demain, sans toucher au reste.
    L'appel est bloquant : c'est TranscriptionService qui le sort de la boucle asyncio.
    """

    def transcribe(self, pcm: bytes) -> Transcript: ...


class WhisperTranscriber:
    """STT local avec faster-whisper (Whisper compilé pour CPU via CTranslate2)."""

    def __init__(self, model_size: str, beam_size: int = 5, initial_prompt: str | None = None) -> None:
        # int8 : poids quantifiés en entiers 8 bits. 2 à 4 fois plus rapide sur CPU qu'en
        # float32 pour une perte de qualité faible. Sans GPU, c'est le bon compromis.
        # Le chargement télécharge le modèle au premier lancement (puis cache local).
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        # Beam search : explore plusieurs hypothèses au lieu de prendre le mot le plus
        # probable à chaque pas. Mesuré : coût négligeable sur de la parole (l'encodeur
        # domine), et quelques homophones en moins sur de l'audio bruité.
        self._beam_size = beam_size
        # Texte présenté à Whisper comme "ce qui précédait" : il oriente le vocabulaire.
        # Mesuré : corrige les noms propres qu'il contient (Lilas, Villeurbanne), mais peut
        # attirer des mots voisins ("mardi" -> "Marty" avec "Martin" dans le prompt).
        self._initial_prompt = initial_prompt

    def transcribe(self, pcm: bytes) -> Transcript:
        # Whisper attend des float32 dans [-1, 1], alors que notre pipeline est en int16.
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / INT16_MAX

        segments, _info = self._model.transcribe(
            samples,
            language=LANGUAGE,  # évite la détection de langue : plus rapide et plus fiable
            beam_size=self._beam_size,
            initial_prompt=self._initial_prompt,
            vad_filter=False,  # notre propre VAD a déjà découpé le tour
            # Chaque tour est indépendant : réutiliser le texte précédent comme contexte
            # propage les erreurs et favorise les hallucinations.
            condition_on_previous_text=False,
            # On garde volontairement les réessais à température croissante (défaut) :
            # mesuré, les supprimer fait halluciner Whisper sur chaque silence.
        )

        kept: list[str] = []
        rejected: list[str] = []
        # `segments` est un générateur PARESSEUX : le calcul n'a lieu qu'à l'itération.
        for segment in segments:
            text = segment.text.strip()
            reason = rejection_reason(segment)
            if reason is None:
                kept.append(text)
            else:
                rejected.append(f"{text} ({reason})")

        return Transcript(text=" ".join(kept), rejected=tuple(rejected))
