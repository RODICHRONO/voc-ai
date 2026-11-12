"""Format audio unique utilisé dans tout le pipeline.

Pourquoi un format figé ? Chaque brique (VAD, STT, TTS, téléphonie) a son propre
format attendu. Si on ne fixe pas un format "maison" au centre, on finit par
convertir dans tous les sens. Ici : PCM 16 bits, mono, 16 kHz.

- 16 kHz : suffisant pour la voix (la parole utile est sous 8 kHz) et c'est le format
  attendu par Silero VAD et par Whisper.
- Le téléphone (Twilio) envoie du µ-law 8 kHz : ce sera une conversion à l'entrée
  (étape Twilio), pas un changement de format interne.
"""

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2  # PCM int16 little-endian

# Silero VAD v5 impose des fenêtres de 512 échantillons à 16 kHz.
# 512 / 16000 = 32 ms : c'est la "résolution temporelle" de toute notre détection.
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
