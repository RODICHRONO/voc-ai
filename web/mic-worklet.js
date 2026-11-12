// AudioWorkletProcessor : tourne dans un thread audio dédié, pas dans le thread de la page.
// C'est la seule façon d'avoir un flux audio régulier sans être perturbé par l'UI.
//
// Rôle : prendre l'audio du micro (souvent 48 kHz, float32) et produire du PCM 16 kHz int16,
// le format attendu par le serveur (voir voc_ai/audio_format.py).

const TARGET_SAMPLE_RATE = 16000;
// 512 échantillons = 32 ms : une trame Silero. On envoie des paquets de cette taille
// pour que le serveur n'ait presque rien à réassembler.
const PACKET_SAMPLES = 512;

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // `sampleRate` est une variable globale du contexte audio (ex. 48000).
    this.ratio = sampleRate / TARGET_SAMPLE_RATE;
    // Décimation par moyenne : on additionne `ratio` échantillons d'entrée pour en
    // produire un. La moyenne agit comme un filtre passe-bas rudimentaire, ce qui
    // limite le repliement de spectre (aliasing) sans écrire un vrai filtre.
    this.phase = 0;
    this.sum = 0;
    this.count = 0;
    this.packet = new Int16Array(PACKET_SAMPLES);
    this.packetLength = 0;
  }

  // Appelée par le navigateur toutes les 128 échantillons.
  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) return true; // micro pas encore branché : on reste en vie

    for (let i = 0; i < channel.length; i++) {
      this.sum += channel[i];
      this.count++;
      this.phase += 1;

      if (this.phase >= this.ratio) {
        this.phase -= this.ratio; // on garde la fraction : gère 44.1 kHz (ratio non entier)
        this.pushSample(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
      }
    }
    return true;
  }

  pushSample(floatSample) {
    // float [-1, 1] -> int16 [-32768, 32767], avec écrêtage pour éviter le débordement.
    const clamped = Math.max(-1, Math.min(1, floatSample));
    this.packet[this.packetLength++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

    if (this.packetLength === PACKET_SAMPLES) {
      // Transfert du buffer (sans copie) vers le thread principal, puis on repart à neuf.
      this.port.postMessage(this.packet.buffer, [this.packet.buffer]);
      this.packet = new Int16Array(PACKET_SAMPLES);
      this.packetLength = 0;
    }
  }
}

registerProcessor("mic-processor", MicProcessor);
