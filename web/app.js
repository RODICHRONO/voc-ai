// Côté navigateur : capture le micro, l'envoie au serveur, affiche ce que le serveur détecte.
// Le navigateur ne fait AUCUNE détection : toute l'intelligence est côté serveur, comme
// ce sera le cas avec Twilio (qui n'est qu'un autre "micro" distant).

const toggleButton = document.querySelector("#toggle");
const statusLabel = document.querySelector("#status");
const probabilityMeter = document.querySelector("#probability");
const log = document.querySelector("#log");
const errorLabel = document.querySelector("#error");

let capture = null; // { stream, audioContext, socket } tant que le micro est actif (les deux derniers peuvent être null pendant le démarrage)

function showError(message) {
  errorLabel.textContent = message;
  errorLabel.hidden = false;
}

function setStatus(state, text) {
  statusLabel.dataset.state = state;
  statusLabel.textContent = text;
}

// Un tour est affiché dès sa fin, puis complété quand la transcription arrive (plusieurs
// secondes plus tard). On retrouve la ligne par start_ms, qui identifie le tour.
const turnItems = new Map();

function logTurn({ start_ms, end_ms, detected_ms }) {
  const item = document.createElement("li");
  const text = document.createElement("div");
  const meta = document.createElement("div");
  text.className = "turn-text";
  text.textContent = "Transcription en cours…";
  meta.className = "turn-meta";
  const durationSeconds = ((end_ms - start_ms) / 1000).toFixed(2);
  // detected - end = le délai incompressible imposé par end_silence_ms côté serveur.
  const delayMs = detected_ms - end_ms;
  meta.textContent = `Parole de ${(start_ms / 1000).toFixed(2)} s, durée ${durationSeconds} s (fin confirmée après ${delayMs} ms de silence)`;

  item.append(text, meta);
  log.append(item);
  turnItems.set(start_ms, { text, meta });
}

function showTranscript({ start_ms, text, rejected, audio_ms, processing_ms }) {
  const item = turnItems.get(start_ms);
  if (!item) return;

  item.text.textContent = text || "(rien d'intelligible)";
  if (rejected.length > 0) {
    // Ce que le filtre anti-hallucinations a retiré : à surveiller pour régler ses seuils.
    const rejectedLine = document.createElement("div");
    rejectedLine.className = "turn-rejected";
    rejectedLine.textContent = `Écarté : ${rejected.join(" · ")}`;
    item.meta.after(rejectedLine);
  }
  // Facteur temps réel : en dessous de 1, le STT va plus vite que la parole.
  const realTimeFactor = (processing_ms / audio_ms).toFixed(2);
  item.meta.textContent += ` · transcrit en ${(processing_ms / 1000).toFixed(1)} s pour ${(audio_ms / 1000).toFixed(1)} s d'audio (×${realTimeFactor})`;
}

function showTranscriptError({ start_ms }) {
  const item = turnItems.get(start_ms);
  if (item) item.text.textContent = "La transcription a échoué (voir les logs du serveur).";
}

function handleServerMessage(message) {
  switch (message.type) {
    case "vad":
      probabilityMeter.value = message.probability;
      break;
    case "speech_started":
      setStatus("speech", "Parole détectée");
      break;
    case "speech_ended":
      setStatus("idle", "Silence");
      logTurn(message);
      break;
    case "transcript":
      showTranscript(message);
      break;
    case "transcript_error":
      showTranscriptError(message);
      break;
  }
}

async function start() {
  errorLabel.hidden = true;

  // echoCancellation sera indispensable plus tard : sans lui, la voix de l'IA qui sort
  // des haut-parleurs rentre dans le micro et déclenche le VAD (faux barge-in).
  // autoGainControl coupé : il amplifie le bruit de fond dans les silences, ce qui
  // fausse la détection.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
  });

  // Enregistré dès maintenant : si une étape suivante échoue, stop() sait quoi libérer
  // (sinon le micro resterait allumé).
  capture = { stream, audioContext: null, socket: null };

  const audioContext = new AudioContext();
  capture.audioContext = audioContext;
  await audioContext.audioWorklet.addModule("mic-worklet.js");

  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/audio`);
  capture.socket = socket;
  socket.binaryType = "arraybuffer";
  socket.addEventListener("message", (event) => handleServerMessage(JSON.parse(event.data)));
  socket.addEventListener("close", stop);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", () => reject(new Error("Connexion au serveur impossible")), { once: true });
  });

  const source = audioContext.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(audioContext, "mic-processor");
  worklet.port.onmessage = (event) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(event.data);
  };

  // Le worklet est relié à la sortie via un gain nul : le navigateur le fait tourner
  // de façon fiable, mais on n'entend pas notre propre voix (larsen).
  const mute = audioContext.createGain();
  mute.gain.value = 0;
  source.connect(worklet).connect(mute).connect(audioContext.destination);

  toggleButton.textContent = "Arrêter le micro";
  setStatus("idle", "Silence");
}

function stop() {
  if (!capture) return;
  const { socket, audioContext, stream } = capture;
  capture = null; // avant les fermetures : l'événement "close" rappelle stop()

  stream.getTracks().forEach((track) => track.stop());
  audioContext?.close();
  socket?.close();

  probabilityMeter.value = 0;
  toggleButton.textContent = "Démarrer le micro";
  setStatus("idle", "Micro arrêté");
}

toggleButton.addEventListener("click", async () => {
  if (capture) {
    stop();
    return;
  }
  try {
    await start();
  } catch (error) {
    stop();
    showError(error.name === "NotAllowedError" ? "Accès au micro refusé." : error.message);
  }
});
