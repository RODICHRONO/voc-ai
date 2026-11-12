import unittest

from voc_ai.audio_format import FRAME_MS
from voc_ai.turn_detector import SpeechEnded, SpeechStarted, TurnConfig, TurnDetector

SPEECH = 0.9
SILENCE = 0.05


def run(detector: TurnDetector, probabilities: list[float]) -> list:
    events = [detector.process(p) for p in probabilities]
    return [event for event in events if event is not None]


class TurnDetectorTest(unittest.TestCase):
    def setUp(self):
        # 3 trames de parole pour démarrer, 5 trames de silence pour finir : tests courts.
        self.config = TurnConfig(min_speech_ms=3 * FRAME_MS, end_silence_ms=5 * FRAME_MS)
        self.detector = TurnDetector(self.config)

    def test_silence_only_emits_nothing(self):
        self.assertEqual(run(self.detector, [SILENCE] * 50), [])

    def test_click_shorter_than_min_speech_is_ignored(self):
        self.assertEqual(run(self.detector, [SPEECH, SPEECH, SILENCE] * 5), [])

    def test_speech_start_is_reported_at_first_speech_frame(self):
        events = run(self.detector, [SILENCE] * 4 + [SPEECH] * 3)

        self.assertEqual(events, [SpeechStarted(start_ms=4 * FRAME_MS)])

    def test_speech_end_waits_for_full_silence_duration(self):
        probabilities = [SPEECH] * 10 + [SILENCE] * 4
        self.assertEqual(len(run(self.detector, probabilities)), 1)  # début seulement

        events = run(self.detector, [SILENCE])  # 5e trame de silence

        self.assertEqual(
            events, [SpeechEnded(start_ms=0, end_ms=10 * FRAME_MS, detected_ms=15 * FRAME_MS)]
        )

    def test_short_pause_inside_speech_does_not_end_the_turn(self):
        probabilities = [SPEECH] * 10 + [SILENCE] * 4 + [SPEECH] * 10
        events = run(self.detector, probabilities)

        self.assertEqual([type(e) for e in events], [SpeechStarted])

    def test_probability_between_thresholds_keeps_speaking(self):
        # 0.4 est sous speech_threshold mais au-dessus de silence_threshold : toujours "parole".
        events = run(self.detector, [SPEECH] * 5 + [0.4] * 30)

        self.assertEqual([type(e) for e in events], [SpeechStarted])

    def test_two_turns_are_reported_separately(self):
        turn = [SPEECH] * 6 + [SILENCE] * 5
        events = run(self.detector, turn + turn)

        self.assertEqual(
            [type(e) for e in events], [SpeechStarted, SpeechEnded, SpeechStarted, SpeechEnded]
        )


if __name__ == "__main__":
    unittest.main()
