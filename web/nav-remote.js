// nav-remote.js — offload client. Same UX as nav.js, but instead of running the
// depth model on the phone (slow), it streams JPEG frames to the laptop over a
// secure WebSocket and gets back the 7 sector distances. Audio is synthesized
// on the phone (synth-worklet.js); Claude taps are unchanged (claude-discuss.js).

import { createDiscussion } from './claude-discuss.js';

const CAP_W = 322, N = 7, ROI_TOP = 0.10, ROI_BOTTOM = 0.80;
const PARAMS = { mode: 'continuous', near_m: 0.5, far_m: 3.0, side_hz: 300, center_hz: 800, falloff: 2.0, sweep_period: 0.3, master: 0.8 };

const $ = (id) => document.getElementById(id);
const video = $('cam'), statusEl = $('status'), startBtn = $('start');
const grid = $('grid').getContext('2d');
const cap = document.createElement('canvas');

let actx = null, node = null, ws = null, running = false, busy = false, stream = null;
let dists = new Array(N).fill(PARAMS.far_m);
const setStatus = (t) => { statusEl.textContent = t; };

async function openCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 960 } }, audio: false,
  });
  video.srcObject = stream; await video.play();
}

async function startAudio() {
  actx = new (window.AudioContext || window.webkitAudioContext)();
  await actx.audioWorklet.addModule('./synth-worklet.js');
  node = new AudioWorkletNode(actx, 'echo-synth', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2] });
  node.connect(actx.destination);
  node.port.postMessage({ params: PARAMS });
  if (actx.state === 'suspended') await actx.resume();
}

function connectWS() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { setStatus('connected to laptop · running'); resolve(); };
    ws.onerror = () => { setStatus('WebSocket error — is the server running & cert trusted?'); reject(new Error('ws')); };
    ws.onclose = () => { setStatus('disconnected from laptop'); running = false; };
    ws.onmessage = (e) => {
      try { const m = JSON.parse(e.data); if (m.dist) { dists = m.dist; node && node.port.postMessage({ dist: dists }); drawGrid(); } } catch {}
      busy = false;
      if (running) requestAnimationFrame(loop);
    };
  });
}

function loop() {
  if (!running || busy || !ws || ws.readyState !== 1) return;
  const vw = video.videoWidth || 4, vh = video.videoHeight || 3;
  cap.width = CAP_W; cap.height = Math.round(CAP_W * vh / vw);
  cap.getContext('2d').drawImage(video, 0, 0, cap.width, cap.height);
  busy = true;
  cap.toBlob((b) => {
    if (!b) { busy = false; return; }
    b.arrayBuffer().then((buf) => { try { ws.send(buf); } catch { busy = false; } });
  }, 'image/jpeg', 0.5);
}

function drawGrid() {
  const W = grid.canvas.width, H = grid.canvas.height;
  grid.clearRect(0, 0, W, H);
  const sw = W / N, near = PARAMS.near_m, far = PARAMS.far_m;
  let openI = 0, openD = -1;
  for (let i = 0; i < N; i++) {
    const d = dists[i], prox = Math.max(0, Math.min(1, (far - d) / (far - near)));
    const barH = prox * H;
    grid.fillStyle = `rgb(${Math.round(255 * prox)},${Math.round(255 * (1 - prox))},60)`;
    grid.fillRect(i * sw + 2, H - barH, sw - 4, barH);
    grid.fillStyle = '#fff'; grid.font = '11px system-ui';
    grid.fillText(d < far ? d.toFixed(1) : 'open', i * sw + 4, 12);
    if (d > openD) { openD = d; openI = i; }
  }
  grid.strokeStyle = '#39d98a'; grid.lineWidth = 3;
  grid.beginPath(); grid.moveTo(W / 2, H); grid.lineTo((openI + 0.5) * sw, 16); grid.stroke();
}

async function start() {
  if (running) return;
  startBtn.disabled = true; setStatus('starting camera…');
  try {
    await openCamera();
    await startAudio();
    createDiscussion({ video, endpoint: '/api/claude', getApiKey: () => null, model: 'claude-sonnet-4-6' });
    await connectWS();
    running = true; startBtn.textContent = 'running';
    loop();
  } catch (e) { setStatus('error: ' + e.message); startBtn.disabled = false; }
}

startBtn.addEventListener('click', start);
for (const m of ['continuous', 'pulse', 'sweep']) {
  $('m_' + m).addEventListener('click', () => {
    PARAMS.mode = m; node && node.port.postMessage({ params: { mode: m } });
    document.querySelectorAll('.mode').forEach((b) => b.classList.toggle('on', b.id === 'm_' + m));
  });
}
$('far_dn').addEventListener('click', () => setFar(-0.5));
$('far_up').addEventListener('click', () => setFar(+0.5));
function setFar(d) { PARAMS.far_m = Math.max(2.0, Math.min(15.0, PARAMS.far_m + d)); node && node.port.postMessage({ params: { far_m: PARAMS.far_m } }); }

// echolocation gain: manual Mute button + auto-duck while Claude is speaking.
let muted = false, ducked = false;
function applyGain() {
  if (!node) return;
  node.port.postMessage({ params: { master: muted ? 0 : (ducked ? 0.12 : PARAMS.master) } });
}
const muteBtn = $('mute');
if (muteBtn) muteBtn.addEventListener('click', () => {
  muted = !muted;
  muteBtn.textContent = muted ? 'Unmute' : 'Mute';
  muteBtn.classList.toggle('on', muted);
  applyGain();
});
setInterval(() => {
  const speaking = !!(window.speechSynthesis && speechSynthesis.speaking);
  if (speaking !== ducked) { ducked = speaking; applyGain(); }
}, 150);
