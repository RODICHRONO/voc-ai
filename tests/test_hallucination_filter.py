import unittest
from types import SimpleNamespace

from voc_ai.hallucination_filter import rejection_reason


def segment(text: str, no_speech_prob: float = 0.02, compression_ratio: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob, compression_ratio=compression_ratio)


class HallucinationFilterTest(unittest.TestCase):
    # Valeurs reprises des mesures faites avec Whisper base (voir hallucination_filter.py).

    def test_clear_speech_is_kept(self):
        self.assertIsNone(rejection_reason(segment("Mon numéro de dossier est le 423.", 0.01, 0.89)))

    def test_noisy_speech_is_kept(self):
        self.assertIsNone(rejection_reason(segment("Bonjour, je voudrais un rendez-vous.", 0.11, 1.06)))

    def test_short_word_with_moderate_no_speech_prob_is_kept(self):
        # Un "oui" isolé a mesuré 0.35 : le seuil ne doit pas le couper.
        self.assertIsNone(rejection_reason(segment("Oui.", 0.37, 0.33)))

    def test_text_invented_on_silence_is_rejected(self):
        reason = rejection_reason(segment("Je vous invite à faire une autre vidéo", 0.70, 1.2))

        self.assertEqual(reason, "probablement pas de parole")

    def test_looping_text_is_rejected(self):
        reason = rejection_reason(segment("de l'écrité de l'écrité de l'écrité", 0.10, 7.29))

        self.assertEqual(reason, "texte qui se répète")

    def test_known_subtitle_credit_is_rejected_even_with_good_scores(self):
        reason = rejection_reason(segment("Sous-titrage ST' 501"))

        self.assertEqual(reason, "hallucination connue")

    def test_known_phrase_matches_typographic_apostrophe(self):
        self.assertEqual(rejection_reason(segment("Merci d’avoir regardé !")), "hallucination connue")


if __name__ == "__main__":
    unittest.main()
