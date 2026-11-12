from collections import deque
from dataclasses import dataclass

from voc_ai.audio_format import BYTES_PER_SAMPLE, FRAME_MS, SAMPLE_RATE

# Le VAD ne confirme un début de parole qu'après min_speech_ms : à ce moment, les premières
# trames sont déjà passées. On les récupère dans la mémoire tampon, plus un peu de marge
# AVANT pour ne pas amputer l'attaque du premier mot (Whisper transcrit mal "...onjour").
PREROLL_FRAMES = 6  # ~190 ms
# Un peu de souffle APRÈS la dernière trame vocale, pour ne pas couper la fin des mots.
TAIL_FRAMES = 3  # ~95 ms
# Taille de la mémoire tampon tant qu'aucune parole n'est détectée. Doit couvrir
# PREROLL_FRAMES + (min_speech_ms / FRAME_MS) ; 20 trames = 640 ms laissent de la marge.
IDLE_RETAINED_FRAMES = 20


@dataclass(frozen=True)
class CompletedTurn:
    start_ms: int
    end_ms: int
    audio: bytes  # PCM au format de audio_format.py

    @property
    def audio_ms(self) -> int:
        return len(self.audio) // BYTES_PER_SAMPLE * 1000 // SAMPLE_RATE


class TurnRecorder:
    """Garde l'audio d'un tour de parole pour le transmettre au STT.

    Le VAD décide QUAND (début/fin) ; ce recorder garde QUOI (les octets). Les deux sont
    séparés : le détecteur de tour reste une machine à états pure, sans audio.

    Usage : push() à chaque trame, puis start() / finish() sur les événements du détecteur.
    """

    def __init__(self) -> None:
        self._frames: deque[bytes] = deque()
        self._next_index = 0  # index absolu de la prochaine trame à arriver
        self._recording = False
        self._start_ms = 0

    @property
    def _first_index(self) -> int:
        return self._next_index - len(self._frames)

    def push(self, frame: bytes) -> None:
        self._frames.append(frame)
        self._next_index += 1

        if not self._recording:
            while len(self._frames) > IDLE_RETAINED_FRAMES:
                self._frames.popleft()

    def start(self, start_ms: int) -> None:
        """Démarre l'enregistrement au début de parole, avec la marge avant."""
        wanted_first = max(0, start_ms // FRAME_MS - PREROLL_FRAMES)
        # Si la mémoire tampon ne remonte pas assez loin, on prend ce qu'il y a.
        drop = max(0, wanted_first - self._first_index)
        for _ in range(drop):
            self._frames.popleft()

        self._recording = True
        self._start_ms = start_ms

    def finish(self, end_ms: int) -> CompletedTurn:
        """Termine l'enregistrement et rend l'audio du tour, sans le silence final."""
        end_index = min(end_ms // FRAME_MS + TAIL_FRAMES, self._next_index)
        frame_count = end_index - self._first_index
        audio = b"".join(list(self._frames)[:frame_count])

        # Le silence qui a servi à confirmer la fin est jeté : il ne fait pas partie du tour.
        self._frames.clear()
        self._recording = False
        return CompletedTurn(start_ms=self._start_ms, end_ms=end_ms, audio=audio)
