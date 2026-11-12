import unittest

from voc_ai.audio_format import FRAME_BYTES, FRAME_MS
from voc_ai.session import AudioSession
from voc_ai.turn_detector import TurnConfig, TurnDetector


class ScriptedVad:
    """Faux VAD : rend les probabilités d'une liste, une par trame reçue."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = iter(probabilities)

    def __call__(self, frame: bytes) -> float:
        return next(self._probabilities)


def make_session(probabilities: list[float]) -> AudioSession:
    config = TurnConfig(min_speech_ms=2 * FRAME_MS, end_silence_ms=2 * FRAME_MS)
    return AudioSession(ScriptedVad(probabilities), TurnDetector(config))


class AudioSessionTest(unittest.TestCase):
    def test_each_frame_produces_a_vad_message_with_its_timestamp(self):
        session = make_session([0.1, 0.1])

        messages = session.feed(b"\x00" * (FRAME_BYTES * 2)).messages

        self.assertEqual(
            messages,
            [
                {"type": "vad", "at_ms": 0, "probability": 0.1},
                {"type": "vad", "at_ms": FRAME_MS, "probability": 0.1},
            ],
        )

    def test_audio_split_across_packets_is_processed_once_complete(self):
        session = make_session([0.1])

        self.assertEqual(session.feed(b"\x00" * (FRAME_BYTES // 2)).messages, [])
        self.assertEqual(len(session.feed(b"\x00" * (FRAME_BYTES // 2)).messages), 1)

    def test_turn_events_are_emitted_after_the_frame_that_triggers_them(self):
        session = make_session([0.9, 0.9, 0.0, 0.0])

        messages = session.feed(b"\x00" * (FRAME_BYTES * 4)).messages

        types = [message["type"] for message in messages]
        self.assertEqual(
            types, ["vad", "vad", "speech_started", "vad", "vad", "speech_ended"]
        )

    def test_finished_turn_is_returned_with_its_audio(self):
        session = make_session([0.9, 0.9, 0.0, 0.0])

        result = session.feed(b"\x00" * (FRAME_BYTES * 4))

        self.assertEqual(len(result.completed_turns), 1)
        turn = result.completed_turns[0]
        self.assertEqual((turn.start_ms, turn.end_ms), (0, 2 * FRAME_MS))
        self.assertGreater(len(turn.audio), 0)

    def test_no_completed_turn_while_speech_is_ongoing(self):
        session = make_session([0.9, 0.9, 0.9])

        self.assertEqual(session.feed(b"\x00" * (FRAME_BYTES * 3)).completed_turns, [])


if __name__ == "__main__":
    unittest.main()
