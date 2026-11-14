"""Écarte les segments que Whisper a probablement inventés.

Whisper a appris sur des vidéos sous-titrées : face à du silence ou du bruit, il "complète"
avec ce qu'il a vu le plus souvent (génériques de sous-titres, "abonnez-vous", texte qui
boucle). Pire : il le fait avec une confiance élevée, donc avg_logprob ne suffit pas.

Seuils choisis sur mesures (voix française de synthèse, propre et bruitée à 10 dB, contre
silence / bruit / souffle) :
- vraie parole       : no_speech_prob <= 0.37, compression_ratio <= 1.06
- hallucinations     : no_speech_prob >= 0.70, compression_ratio >= 5.7
"""

from typing import Protocol

# Probabilité, estimée par Whisper, que le segment ne contienne pas de parole. Un "oui"
# isolé monte déjà à 0.35 : ne pas descendre ce seuil sans le remesurer.
MAX_NO_SPEECH_PROB = 0.6
# Texte qui se répète = texte très compressible. 2.4 est le seuil utilisé par Whisper
# lui-même pour déclencher ses réessais.
MAX_COMPRESSION_RATIO = 2.4
# Phrases bien connues des hallucinations de Whisper en français (génériques de sous-titres).
# Dernier filet : elles peuvent passer les seuils ci-dessus.
KNOWN_HALLUCINATIONS = (
    "sous-titr",
    "amara.org",
    "merci d'avoir regardé",
    "abonnez-vous",
)


class WhisperSegment(Protocol):
    text: str
    no_speech_prob: float
    compression_ratio: float


def rejection_reason(segment: WhisperSegment) -> str | None:
    """Renvoie pourquoi le segment est écarté, ou None s'il est gardé."""
    if segment.no_speech_prob > MAX_NO_SPEECH_PROB:
        return "probablement pas de parole"
    if segment.compression_ratio > MAX_COMPRESSION_RATIO:
        return "texte qui se répète"

    normalized = segment.text.lower().replace("’", "'")
    if any(phrase in normalized for phrase in KNOWN_HALLUCINATIONS):
        return "hallucination connue"
    return None
