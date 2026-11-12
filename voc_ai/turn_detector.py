from dataclasses import dataclass

from voc_ai.audio_format import FRAME_MS


@dataclass(frozen=True)
class TurnConfig:
    """Réglages de la détection de tour de parole.

    Ce sont LES paramètres qui font la qualité perçue d'un agent vocal :
    - end_silence_ms trop court : l'IA coupe la parole à l'utilisateur qui réfléchit.
    - end_silence_ms trop long : l'IA met trop de temps à répondre.
    Ce délai s'ajoute directement à la latence totale, il ne se compresse pas.
    """

    # Au-dessus : on considère que la trame contient de la parole.
    speech_threshold: float = 0.5
    # En dessous : la trame compte comme du silence. Deux seuils (hystérésis) pour
    # éviter qu'une probabilité qui oscille autour de 0.5 ne fasse clignoter l'état.
    silence_threshold: float = 0.35
    # Durée minimale de parole continue avant de déclarer un début : filtre les clics
    # et les bruits brefs.
    min_speech_ms: int = 96
    # Silence continu nécessaire pour déclarer la fin du tour.
    end_silence_ms: int = 700


@dataclass(frozen=True)
class SpeechStarted:
    start_ms: int
    type: str = "speech_started"


@dataclass(frozen=True)
class SpeechEnded:
    start_ms: int
    # Fin de la dernière trame vocale (et non le moment où on l'a détectée).
    end_ms: int
    # Moment où on a pu conclure. detected_ms - end_ms = délai imposé par end_silence_ms.
    detected_ms: int
    type: str = "speech_ended"


TurnEvent = SpeechStarted | SpeechEnded


class TurnDetector:
    """Machine à états : transforme des probabilités de parole en événements de tour.

    Elle ne connaît ni Silero ni l'audio : elle reçoit un nombre par trame. C'est ce
    qui permet de la tester avec des probabilités fabriquées, sans modèle ni micro.
    """

    def __init__(self, config: TurnConfig | None = None) -> None:
        self._config = config or TurnConfig()
        self._frame_index = 0
        self._is_speaking = False
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._speech_start_ms = 0
        self._last_speech_end_ms = 0

    def process(self, speech_probability: float) -> TurnEvent | None:
        """Traite une trame de FRAME_MS ms. Renvoie un événement si l'état change."""
        frame_start_ms = self._frame_index * FRAME_MS
        self._frame_index += 1

        if self._is_speaking:
            return self._process_while_speaking(speech_probability, frame_start_ms)
        return self._process_while_idle(speech_probability, frame_start_ms)

    def _process_while_idle(self, probability: float, frame_start_ms: int) -> SpeechStarted | None:
        if probability < self._config.speech_threshold:
            self._consecutive_speech_frames = 0
            return None

        self._consecutive_speech_frames += 1
        if self._consecutive_speech_frames * FRAME_MS < self._config.min_speech_ms:
            return None

        # Le début réel de la parole est la première trame de la série, pas la trame
        # courante : on remonte dans le temps.
        self._speech_start_ms = frame_start_ms - (self._consecutive_speech_frames - 1) * FRAME_MS
        self._last_speech_end_ms = frame_start_ms + FRAME_MS
        self._is_speaking = True
        self._consecutive_silence_frames = 0
        return SpeechStarted(start_ms=self._speech_start_ms)

    def _process_while_speaking(self, probability: float, frame_start_ms: int) -> SpeechEnded | None:
        frame_end_ms = frame_start_ms + FRAME_MS

        if probability >= self._config.silence_threshold:
            # Entre les deux seuils on reste du côté "parole" : une hésitation du
            # modèle ne doit pas démarrer le compte à rebours de fin de tour.
            self._consecutive_silence_frames = 0
            self._last_speech_end_ms = frame_end_ms
            return None

        self._consecutive_silence_frames += 1
        if self._consecutive_silence_frames * FRAME_MS < self._config.end_silence_ms:
            return None

        self._is_speaking = False
        self._consecutive_speech_frames = 0
        return SpeechEnded(
            start_ms=self._speech_start_ms,
            end_ms=self._last_speech_end_ms,
            detected_ms=frame_end_ms,
        )
