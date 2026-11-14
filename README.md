# voc-ai

Un agent vocal construit pièce par pièce, pour apprendre comment fonctionne une chaîne
**voix → reconnaissance vocale → LLM → synthèse vocale**, et ce qui la rend difficile :
latence, détection de tour de parole, qualité de transcription, tests sur de vrais appels.

Tout tourne **en local, sans fournisseur payant**. Chaque étage est derrière une interface :
on pourra brancher plus tard Twilio, Speechmatics/Deepgram ou ElevenLabs pour comparer.

## Pourquoi ce projet

Point de départ : un retour d'entretien pour un poste « voice AI » qui demandait une
expérience concrète de la chaîne téléphonie → STT → LLM → TTS (Twilio, Speechmatics,
ElevenLabs), avec gestion de la latence, détection de tour de parole (VAD) et tests sur des
appels réels. Ce dépôt sert à apprendre ces mécanismes en les construisant, pas en assemblant
un framework qui les cache.

## Où on en est

| Étape | État |
|---|---|
| 1. Capture micro (navigateur) → serveur en WebSocket | ✅ |
| 2. Détection de parole (Silero VAD) et de fin de tour | ✅ |
| 3. Transcription locale (Whisper) + filtre anti-hallucinations | ✅ |
| 4. Réponse d'un LLM local (Ollama) | à faire |
| 5. Synthèse vocale locale (Piper) | à faire |
| 6. Latence mesurée par étage, de bout en bout | partiel (VAD, STT) |
| 7. Barge-in (l'utilisateur coupe l'IA) | à faire |
| 8. Téléphonie (Twilio, µ-law 8 kHz) | à faire |

## Architecture

```
Navigateur                              Serveur (FastAPI)
──────────                              ─────────────────
micro 48 kHz float32
  │ AudioWorklet : 16 kHz int16
  ▼
WebSocket (binaire, paquets 32 ms) ──▶ FrameAssembler   trames exactes de 512 échantillons
                                          │
                                          ├─▶ TurnRecorder    garde l'audio du tour
                                          ▼
                                       Silero VAD         probabilité de parole par trame
                                          ▼
                                       TurnDetector       début / fin de tour (machine à états)
                                          │ fin de tour
                                          ▼
                                       TranscriptionService   thread dédié, hors boucle asyncio
                                          ▼
                                       WhisperTranscriber + filtre d'hallucinations
                                          │
page : jauge VAD, tours, texte ◀── WebSocket (JSON)
```

| Fichier | Rôle |
|---|---|
| `voc_ai/audio_format.py` | Format pivot unique : PCM 16 bits, mono, 16 kHz, trames de 32 ms |
| `voc_ai/frame_assembler.py` | Redécoupe les paquets réseau en trames de taille fixe |
| `voc_ai/turn_detector.py` | Machine à états pure : probabilités → `SpeechStarted` / `SpeechEnded` |
| `voc_ai/turn_recorder.py` | Garde l'audio d'un tour, avec marge avant (préroll) et après |
| `voc_ai/session.py` | Une conversation, indépendante du transport (WebSocket aujourd'hui, Twilio demain) |
| `voc_ai/transcriber.py` | Interface `SpeechTranscriber` + implémentation Whisper |
| `voc_ai/hallucination_filter.py` | Écarte les segments inventés par Whisper |
| `voc_ai/transcription_service.py` | Exécute le STT bloquant hors de la boucle asyncio, mesure son temps |
| `voc_ai/turn_archive.py` | Sauvegarde chaque tour en WAV (diagnostic) |
| `voc_ai/main.py` | Serveur FastAPI, WebSocket, configuration |
| `web/` | Page de test : micro, jauge VAD, tours détectés, transcriptions |
| `scripts/compare_stt.py` | Rejoue des WAV avec plusieurs réglages de STT |

## Lancer

Prérequis : [uv](https://docs.astral.sh/uv/). Python 3.12 est épinglé (`.python-version`) :
3.14 est trop récent pour les bibliothèques audio et ML.

```bash
uv sync
uv run uvicorn voc_ai.main:app --port 8765
# puis http://localhost:8765 → « Démarrer le micro »
```

Le premier lancement télécharge le modèle Whisper (cache local ensuite).

| Variable | Défaut | Effet |
|---|---|---|
| `VOC_WHISPER_MODEL` | `base` | Taille Whisper : `tiny`, `base`, `small`… |
| `VOC_WHISPER_BEAM` | `5` | Largeur du beam search (1 = décodage glouton) |
| `VOC_WHISPER_PROMPT` | aucun | Vocabulaire attendu : noms propres, termes métier |
| `VOC_RECORD_DIR` | aucun | Si défini, chaque tour est sauvegardé en WAV dans ce dossier |

## Tester

```bash
# Tests unitaires (sans modèle ni micro : les modèles sont remplacés par des faux)
uv run python -m unittest discover -s tests -t .

# Enregistrer ses vrais tours, puis comparer des réglages sur exactement le même audio
VOC_RECORD_DIR=recordings uv run uvicorn voc_ai.main:app --port 8765
uv run python -m scripts.compare_stt recordings/ --models base,small --beams 1,5
uv run python -m scripts.compare_stt recordings/ --prompt "Rue des Lilas, Villeurbanne."
```

`recordings/` est ignoré par git : il contient des voix.

## Ce qu'on a appris

### 1. Un format audio pivot, et tout le reste s'y convertit

Chaque brique attend son propre format : le micro donne du float32 à 48 kHz, Silero et Whisper
veulent du 16 kHz, Twilio enverra du µ-law à 8 kHz. On fixe **un** format interne (PCM int16,
mono, 16 kHz) et on convertit aux frontières. Silero impose des fenêtres de 512 échantillons :
**32 ms**, c'est la résolution temporelle de toute la détection.

### 2. La détection de fin de tour est un compromis, et elle coûte de la latence

- Deux seuils (hystérésis, 0,5 pour entrer, 0,35 pour sortir) évitent que l'état clignote
  quand la probabilité oscille.
- Une durée minimale de parole (96 ms) filtre les clics et bruits brefs.
- Le silence de fin (700 ms) est le réglage clé : trop court, l'IA coupe la parole à
  quelqu'un qui réfléchit ; trop long, elle met du temps à répondre. **Ce délai s'ajoute à
  toute la latence et ne se compresse pas.** Avec des trames de 32 ms, 700 ms deviennent 704 ms.
- Le début de parole n'est confirmé qu'après coup : il faut une mémoire tampon pour récupérer
  les premières trames, plus une marge (préroll ~190 ms) pour ne pas couper le premier mot.

### 3. Ne jamais bloquer la boucle asyncio

Whisper met une à plusieurs secondes. Appelé directement dans le handler WebSocket, il gèle
tout : plus de VAD, plus de réponse. Il tourne donc dans un thread dédié (un seul : il utilise
déjà tous les cœurs), en tâche de fond, pendant que l'audio continue d'arriver.

### 4. Whisper a un coût fixe : il traite toujours 30 secondes

Whisper découpe l'audio en fenêtres de 30 s ; un tour de 2 s est complété par du silence.
Un tour court coûte donc presque autant qu'un long. Mesures sur un i3-1215U sans GPU (int8) :

| Modèle | Temps par tour (voix réelle) | Qualité |
|---|---|---|
| `base` | ~1 à 2 s | correcte, se trompe sur les noms propres et dans le bruit |
| `small` | ~3 à 5 s | nettement meilleure (seul à reconnaître un nom propre) |

C'est ce qui justifie les STT en streaming (Speechmatics, Deepgram) en production : ils
transcrivent pendant que la personne parle, au lieu de tout reprendre à la fin.

### 5. Les hallucinations : Whisper invente, et il est sûr de lui

Face à du silence ou du bruit, Whisper produit des phrases de sous-titres YouTube vues pendant
son entraînement (« Je vous invite à faire une autre vidéo… », « Sous-titrage… »), souvent en
boucle. Sa confiance (`avg_logprob`) reste élevée, donc le filtre par défaut ne suffit pas.

Mesures sur Whisper `base` :

| | `no_speech_prob` | `compression_ratio` |
|---|---|---|
| Vraie parole, propre ou bruitée | ≤ 0,37 | ≤ 1,06 |
| Hallucinations | ≥ 0,70 | ≥ 5,7 |

D'où le filtre : un segment est écarté s'il dépasse 0,6 de `no_speech_prob`, 2,4 de taux de
compression (texte qui boucle) ou s'il contient une phrase de sous-titres connue. Les segments
écartés restent affichés, pour voir si le filtre est trop strict.

Hypothèse **réfutée** au passage : désactiver les réessais à température croissante
(`temperature=0`) pour gagner du temps fait halluciner Whisper sur chaque silence. Ces
réessais sont lents mais protecteurs.

### 6. Les mots mal reconnus : d'abord le bruit, puis le modèle

- Sur de l'audio propre, `base` se trompe surtout sur les noms propres. Avec du bruit, on
  obtient des homophones : « mardi » → « Marie », « Martin » → « Marc-Cent ».
- Le **micro** compte plus que tout réglage logiciel.
- Un **prompt de vocabulaire** corrige les noms qu'il contient, mais attire des mots voisins
  (« mardi » → « Marty » avec « Martin » dans le prompt).
- Le **beam search** n'est pas toujours meilleur : beam 5 aidait sur la voix de synthèse,
  beam 1 a mieux fait sur une vraie voix.

### 7. Tester sur de vrais enregistrements, et vérifier ses données de test

- Les voix de synthèse servent à automatiser, mais les conclusions ont changé sur une vraie voix.
  D'où l'enregistrement des tours en WAV et le script de comparaison : c'est la base d'un banc
  de test « sur appels réels ».
- Un jeu de test peut être faux : des phrases générées par la synthèse vocale de Windows
  contenaient « symbole copyright » parce que PowerShell lisait le script en Windows-1252
  (« numéro » → « numÃ©ro » → « Ã© » prononcé). Whisper transcrivait fidèlement… le mauvais audio.
- Une mesure peut être fausse : `faster-whisper` renvoie un générateur **paresseux** ;
  chronométrer avant de l'itérer mesure 0,03 s au lieu de plusieurs secondes.
- Une mesure peut être biaisée : `small` semblait prendre 11 à 14 s, mais c'était sur du bruit,
  où il génère un long texte halluciné. Sur de la vraie voix : 3 à 5 s.

## Suite

1. Choisir le réglage STT par défaut sur une vingtaine d'enregistrements réels.
2. LLM local (Ollama) en streaming, puis synthèse vocale (Piper).
3. Mesurer la latence de chaque étage et le total « fin de parole → premier son ».
4. Barge-in : couper la réponse quand l'utilisateur reparle.
5. Twilio : un vrai numéro, conversion µ-law 8 kHz ↔ format pivot.
6. Comparer avec des fournisseurs cloud derrière les mêmes interfaces.
