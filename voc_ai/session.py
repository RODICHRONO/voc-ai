from dataclasses import asdict, dataclass, field
from typing import Protocol

from voc_ai.audio_format import FRAME_MS
from voc_ai.frame_assembler import FrameAssembler
from voc_ai.turn_detector import SpeechEnded, SpeechStarted, TurnDetector
from voc_ai.turn_recorder import CompletedTurn, TurnRecorder


class SpeechProbabilityModel(Protocol):
    """Tout ce qui sait dire "quelle probabilité que cette trame soit de la parole".

    Silero l'implémente tel quel. Dans les tests, on met une fonction bidon : c'est le
    même principe qui sert pour le STT, et qui servira pour le LLM et le TTS
    (interfaces interchangeables).
    """

    def __call__(self, frame: bytes) -> float: ...


@dataclass(frozen=True)
class FrameProbability:
    at_ms: int
    probability: float
    type: str = "vad"


@dataclass(frozen=True)
class FeedResult:
    # À envoyer tout de suite au client (état du VAD, début/fin de tour).
    messages: list[dict] = field(default_factory=list)
    # Tours terminés dont l'audio attend d'être transcrit. Séparés des messages parce que
    # leur traitement est lent : c'est à l'appelant de décider comment l'exécuter.
    completed_turns: list[CompletedTurn] = field(default_factory=list)


class AudioSession:
    """Une conversation = une session. Elle possède l'état audio de UNE connexion.

    Volontairement indépendante de FastAPI/WebSocket : elle prend des octets et rend
    des résultats. Le transport (WebSocket aujourd'hui, Twilio demain) est un détail
    qui reste dans main.py.
    """

    def __init__(self, vad: SpeechProbabilityModel, turn_detector: TurnDetector) -> None:
        self._vad = vad
        self._turn_detector = turn_detector
        self._assembler = FrameAssembler()
        self._recorder = TurnRecorder()
        self._frame_index = 0

    def feed(self, chunk: bytes) -> FeedResult:
        """Consomme un paquet audio et renvoie ce qui en résulte."""
        messages: list[dict] = []
        completed_turns: list[CompletedTurn] = []

        for frame in self._assembler.push(chunk):
            at_ms = self._frame_index * FRAME_MS
            self._frame_index += 1

            # Le recorder reçoit la trame AVANT la décision du VAD : au moment où le début
            # de parole est confirmé, la trame courante doit déjà être en mémoire.
            self._recorder.push(frame)

            probability = self._vad(frame)
            messages.append(asdict(FrameProbability(at_ms=at_ms, probability=probability)))

            event = self._turn_detector.process(probability)
            if event is None:
                continue

            if isinstance(event, SpeechStarted):
                self._recorder.start(event.start_ms)
            elif isinstance(event, SpeechEnded):
                completed_turns.append(self._recorder.finish(event.end_ms))
            messages.append(asdict(event))

        return FeedResult(messages=messages, completed_turns=completed_turns)
