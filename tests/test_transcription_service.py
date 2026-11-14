import threading
import unittest

from voc_ai.transcriber import Transcript
from voc_ai.transcription_service import TranscriptionService
from voc_ai.turn_recorder import CompletedTurn


class FakeTranscriber:
    def __init__(self) -> None:
        self.received: list[bytes] = []
        self.thread_name = ""

    def transcribe(self, pcm: bytes) -> Transcript:
        self.received.append(pcm)
        self.thread_name = threading.current_thread().name
        return Transcript(text="bonjour", rejected=("Merci d'avoir regardé (hallucination connue)",))


class FailingTranscriber:
    def transcribe(self, pcm: bytes) -> Transcript:
        raise RuntimeError("modèle indisponible")


def make_turn() -> CompletedTurn:
    return CompletedTurn(start_ms=320, end_ms=1280, audio=b"\x00\x00" * 16_000)  # 1 s


class TranscriptionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_transcript_message_carries_text_and_timings(self):
        service = TranscriptionService(FakeTranscriber())
        self.addCleanup(service.close)

        message = await service.transcribe(make_turn())

        self.assertEqual(message["type"], "transcript")
        self.assertEqual((message["start_ms"], message["end_ms"]), (320, 1280))
        self.assertEqual(message["text"], "bonjour")
        self.assertEqual(message["rejected"], ["Merci d'avoir regardé (hallucination connue)"])
        self.assertEqual(message["audio_ms"], 1000)
        self.assertGreaterEqual(message["processing_ms"], 0)

    async def test_transcriber_receives_the_turn_audio(self):
        transcriber = FakeTranscriber()
        service = TranscriptionService(transcriber)
        self.addCleanup(service.close)
        turn = make_turn()

        await service.transcribe(turn)

        self.assertEqual(transcriber.received, [turn.audio])

    async def test_blocking_transcriber_runs_outside_the_event_loop_thread(self):
        transcriber = FakeTranscriber()
        service = TranscriptionService(transcriber)
        self.addCleanup(service.close)

        await service.transcribe(make_turn())

        self.assertTrue(transcriber.thread_name.startswith("stt"))

    async def test_transcriber_error_is_propagated_to_the_caller(self):
        service = TranscriptionService(FailingTranscriber())
        self.addCleanup(service.close)

        with self.assertRaisesRegex(RuntimeError, "modèle indisponible"):
            await service.transcribe(make_turn())


if __name__ == "__main__":
    unittest.main()
