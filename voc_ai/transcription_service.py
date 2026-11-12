import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from voc_ai.transcriber import SpeechTranscriber
from voc_ai.turn_recorder import CompletedTurn


@dataclass(frozen=True)
class TurnTranscribed:
    start_ms: int
    end_ms: int
    text: str
    audio_ms: int
    # Temps de calcul du STT. processing_ms / audio_ms = "facteur temps réel" : au-dessus
    # de 1, le STT est plus lent que la parole et on ne pourra pas tenir une conversation.
    processing_ms: int
    type: str = "transcript"


class TranscriptionService:
    """Exécute le STT (bloquant, plusieurs secondes sur CPU) sans geler la boucle asyncio.

    Si on appelait Whisper directement dans le handler WebSocket, tout s'arrêterait
    pendant le calcul : plus de VAD, plus de réponse au client.
    """

    def __init__(self, transcriber: SpeechTranscriber) -> None:
        self._transcriber = transcriber
        # Un seul thread : le STT utilise déjà tous les cœurs ; en lancer deux en parallèle
        # ralentirait les deux. Les tours en attente passent les uns après les autres.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    async def transcribe(self, turn: CompletedTurn) -> dict:
        started = time.perf_counter()
        text = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._transcriber.transcribe, turn.audio
        )
        processing_ms = round((time.perf_counter() - started) * 1000)

        return asdict(
            TurnTranscribed(
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                text=text,
                audio_ms=turn.audio_ms,
                processing_ms=processing_ms,
            )
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
