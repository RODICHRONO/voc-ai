import unittest

from voc_ai.audio_format import FRAME_BYTES
from voc_ai.frame_assembler import FrameAssembler


class FrameAssemblerTest(unittest.TestCase):
    def test_incomplete_chunk_yields_no_frame(self):
        self.assertEqual(FrameAssembler().push(b"\x01" * (FRAME_BYTES - 1)), [])

    def test_exact_frame_is_returned_as_is(self):
        frame = bytes(range(256)) * (FRAME_BYTES // 256)

        self.assertEqual(FrameAssembler().push(frame), [frame])

    def test_leftover_bytes_are_kept_for_next_chunk(self):
        assembler = FrameAssembler()
        assembler.push(b"\x01" * 100)

        frames = assembler.push(b"\x02" * (FRAME_BYTES - 100))

        self.assertEqual(frames, [b"\x01" * 100 + b"\x02" * (FRAME_BYTES - 100)])

    def test_large_chunk_is_split_into_several_frames(self):
        frames = FrameAssembler().push(b"\x00" * (FRAME_BYTES * 3 + 10))

        self.assertEqual(len(frames), 3)
        self.assertTrue(all(len(frame) == FRAME_BYTES for frame in frames))


if __name__ == "__main__":
    unittest.main()
