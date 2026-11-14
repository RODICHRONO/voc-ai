import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from pysilero_vad import SileroVoiceActivityDetector

from voc_ai.session import AudioSession
from voc_ai.transcriber import WhisperTranscriber
from voc_ai.transcription_service import TranscriptionService
from voc_ai.turn_archive import TurnArchive
from voc_ai.turn_detector import TurnConfig, TurnDetector
from voc_ai.turn_recorder import CompletedTurn

# Root en WARNING : à INFO, chaque requête HTTP du téléchargement du modèle est loguée.
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("voc_ai")
logger.setLevel(logging.INFO)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
WS_UNSUPPORTED_DATA = 1003
# Configurable sans toucher au code : VOC_WHISPER_MODEL=small pour un meilleur français.
# "base" par défaut : mesuré sur CPU sans GPU, "small" met 10 s et plus par tour, "base" 1 à 3 s.
WHISPER_MODEL = os.environ.get("VOC_WHISPER_MODEL", "base")
# Vocabulaire attendu (noms propres, termes métier), ex. "Rendez-vous. Rue des Lilas, Villeurbanne."
WHISPER_PROMPT = os.environ.get("VOC_WHISPER_PROMPT") or None
# Mesuré sur voix réelle : avec small, beam 1 a mieux reconnu un nom propre que beam 5.
# À choisir avec scripts/compare_stt.py sur ses propres enregistrements.
WHISPER_BEAM = int(os.environ.get("VOC_WHISPER_BEAM", "5"))
# Si défini, chaque tour est sauvegardé en WAV dans ce dossier (diagnostic, voir turn_archive.py).
RECORD_DIR = os.environ.get("VOC_RECORD_DIR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle Whisper se charge UNE fois au démarrage (plusieurs secondes, plus un
    # téléchargement au tout premier lancement), pas à la première phrase de l'utilisateur.
    logger.info("chargement du modèle Whisper '%s' (beam %d)", WHISPER_MODEL, WHISPER_BEAM)
    app.state.transcription = TranscriptionService(
        WhisperTranscriber(WHISPER_MODEL, beam_size=WHISPER_BEAM, initial_prompt=WHISPER_PROMPT)
    )
    logger.info("modèle prêt")
    yield
    app.state.transcription.close()


app = FastAPI(title="voc-ai", lifespan=lifespan)


async def report_transcript(
    websocket: WebSocket, transcription: TranscriptionService, turn: CompletedTurn
) -> None:
    """Transcrit un tour en tâche de fond et envoie le texte au client quand il est prêt."""
    try:
        await websocket.send_json(await transcription.transcribe(turn))
    except Exception:
        # Dans une tâche de fond, une exception non attrapée disparaîtrait en silence.
        logger.exception("transcription échouée pour le tour démarré à %d ms", turn.start_ms)
        await websocket.send_json({"type": "transcript_error", "start_ms": turn.start_ms})


@app.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
    """Reçoit l'audio du navigateur (binaire) et renvoie des événements (JSON).

    Un détecteur Silero par connexion : le modèle est à mémoire (état récurrent), il
    ne faut surtout pas le partager entre deux conversations.

    La transcription tourne en tâche de fond : la boucle de réception continue de lire
    l'audio (et donc le VAD de fonctionner) pendant que Whisper calcule.
    """
    await websocket.accept()
    transcription: TranscriptionService = websocket.app.state.transcription
    session = AudioSession(SileroVoiceActivityDetector(), TurnDetector(TurnConfig()))
    background_tasks: set[asyncio.Task] = set()
    archive = TurnArchive(Path(RECORD_DIR)) if RECORD_DIR else None
    logger.info("connexion audio ouverte")

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                logger.info("connexion audio fermée")
                return

            audio = message.get("bytes")
            if audio is None:
                # Le protocole d'entrée est binaire uniquement : un message texte est une erreur client.
                await websocket.close(code=WS_UNSUPPORTED_DATA)
                return

            result = session.feed(audio)
            for event in result.messages:
                await websocket.send_json(event)

            for turn in result.completed_turns:
                if archive is not None:
                    # Écriture synchrone : quelques dizaines de Ko par tour, négligeable.
                    logger.info("tour enregistré : %s", archive.save(turn))
                task = asyncio.create_task(report_transcript(websocket, transcription, turn))
                background_tasks.add(task)  # référence forte : sinon la tâche peut être ramassée en cours de route
                task.add_done_callback(background_tasks.discard)
    finally:
        # Plus personne à qui envoyer les résultats une fois la connexion fermée.
        for task in background_tasks:
            task.cancel()


# Monté en dernier : sinon "/" capterait aussi /ws/audio.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
