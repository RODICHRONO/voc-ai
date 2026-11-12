import unittest

from voc_ai.audio_format import FRAME_BYTES, FRAME_MS
from voc_ai.turn_recorder import IDLE_RETAINED_FRAMES, PREROLL_FRAMES, TAIL_FRAMES, TurnRecorder


def frame(index: int) -> bytes:
    """Trame dont tous les octets valent son index : on retrouve d'où vient chaque morceau."""
    return bytes([index]) * FRAME_BYTES


def frame_indexes(audio: bytes) -> list[int]:
    return [audio[i] for i in range(0, len(audio), FRAME_BYTES)]


def record(recorder: TurnRecorder, frame_count: int) -> None:
    for index in range(frame_count):
        recorder.push(frame(index))


class TurnRecorderTest(unittest.TestCase):
    def test_turn_includes_preroll_before_and_tail_after_the_speech(self):
        recorder = TurnRecorder()
        record(recorder, 14)  # le début de parole (trame 10) est confirmé à la trame 13
        recorder.start(start_ms=10 * FRAME_MS)
        for index in range(14, 31):
            recorder.push(frame(index))

        turn = recorder.finish(end_ms=20 * FRAME_MS)

        expected = list(range(10 - PREROLL_FRAMES, 20 + TAIL_FRAMES))
        self.assertEqual(frame_indexes(turn.audio), expected)

    def test_turn_keeps_its_boundaries(self):
        recorder = TurnRecorder()
        record(recorder, 12)
        recorder.start(start_ms=8 * FRAME_MS)

        turn = recorder.finish(end_ms=11 * FRAME_MS)

        self.assertEqual((turn.start_ms, turn.end_ms), (8 * FRAME_MS, 11 * FRAME_MS))

    def test_preroll_is_clamped_when_speech_starts_at_the_very_beginning(self):
        recorder = TurnRecorder()
        record(recorder, 5)
        recorder.start(start_ms=0)

        turn = recorder.finish(end_ms=3 * FRAME_MS)

        self.assertEqual(frame_indexes(turn.audio)[0], 0)

    def test_idle_buffer_does_not_grow_without_speech(self):
        recorder = TurnRecorder()
        record(recorder, IDLE_RETAINED_FRAMES * 5)
        recorder.start(start_ms=(IDLE_RETAINED_FRAMES * 5 - 3) * FRAME_MS)

        turn = recorder.finish(end_ms=IDLE_RETAINED_FRAMES * 5 * FRAME_MS)

        self.assertLessEqual(len(turn.audio) // FRAME_BYTES, IDLE_RETAINED_FRAMES)

    def test_second_turn_does_not_contain_audio_of_the_first(self):
        recorder = TurnRecorder()
        record(recorder, 12)
        recorder.start(start_ms=4 * FRAME_MS)
        recorder.finish(end_ms=8 * FRAME_MS)
        for index in range(12, 40):
            recorder.push(frame(index))
        recorder.start(start_ms=30 * FRAME_MS)

        turn = recorder.finish(end_ms=35 * FRAME_MS)

        self.assertGreaterEqual(frame_indexes(turn.audio)[0], 30 - PREROLL_FRAMES)

    def test_audio_duration_is_derived_from_the_bytes(self):
        recorder = TurnRecorder()
        record(recorder, 12)
        recorder.start(start_ms=6 * FRAME_MS)

        turn = recorder.finish(end_ms=9 * FRAME_MS)

        self.assertEqual(turn.audio_ms, len(turn.audio) // FRAME_BYTES * FRAME_MS)


if __name__ == "__main__":
    unittest.main()
