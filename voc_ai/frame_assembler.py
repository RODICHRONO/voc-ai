from voc_ai.audio_format import FRAME_BYTES


class FrameAssembler:
    """Redécoupe un flux d'octets arbitraire en trames de taille fixe.

    Pourquoi : le réseau livre l'audio par paquets de taille quelconque (le navigateur,
    plus tard Twilio avec ses 20 ms). Le VAD, lui, exige exactement FRAME_BYTES.
    Cette classe garde le reste en mémoire jusqu'au prochain paquet, sans rien perdre.
    """

    def __init__(self) -> None:
        self._pending = bytearray()

    def push(self, chunk: bytes) -> list[bytes]:
        """Ajoute des octets et renvoie toutes les trames complètes disponibles."""
        self._pending.extend(chunk)

        frames = []
        while len(self._pending) >= FRAME_BYTES:
            frames.append(bytes(self._pending[:FRAME_BYTES]))
            del self._pending[:FRAME_BYTES]
        return frames
