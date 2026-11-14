import tempfile
import unittest
import wave
from pathlib import Path

from voc_ai.audio_format import SAMPLE_RATE
from voc_ai.turn_archive import TurnArchive
from voc_ai.turn_recorder import CompletedTurn


class TurnArchiveTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def test_saved_file_is_a_pipeline_format_wav_with_the_turn_audio(self):
        audio = bytes(range(256)) * 8
        path = TurnArchive(self.directory).save(CompletedTurn(start_ms=320, end_ms=640, audio=audio))

        with wave.open(str(path), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (SAMPLE_RATE, 1, 2))
            self.assertEqual(wav.readframes(wav.getnframes()), audio)

    def test_turns_of_one_session_get_distinct_files(self):
        archive = TurnArchive(self.directory)

        first = archive.save(CompletedTurn(start_ms=0, end_ms=320, audio=b"\x00\x00"))
        second = archive.save(CompletedTurn(start_ms=960, end_ms=1280, audio=b"\x00\x00"))

        self.assertNotEqual(first, second)

    def test_missing_directory_is_created(self):
        target = self.directory / "sub" / "dir"

        TurnArchive(target).save(CompletedTurn(start_ms=0, end_ms=32, audio=b"\x00\x00"))

        self.assertEqual(len(list(target.glob("*.wav"))), 1)


if __name__ == "__main__":
    unittest.main()
