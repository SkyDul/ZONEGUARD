/* ══════════════════════════════════
   ZONEGUARD – app.js
   Full client-side logic (NO DOM library)
   ══════════════════════════════════ */

'use strict';

// ─── Config ───────────────────────────────────────────────────
const API_BASE         = 'http://localhost:5000';
const SAVE_EVENT_EVERY = 3;   // save snapshot every N intrusion frames
const MAX_LOG_ITEMS    = 50;

// Track color palette (hue steps for distinct per-ID colors)
const TRACK_PALETTE = [
  '#3B9EFF','#22D3A0','#F5A623','#C084FC','#FB7185',
  '#34D399','#FBBF24','#60A5FA','#A78BFA','#F472B6',
];
const trackColors = {};   // track_id → color string
function getTrackColor(id) {
  if (!trackColors[id]) trackColors[id] = TRACK_PALETTE[id % TRACK_PALETTE.length];
  return trackColors[id];
}

// ─── State ────────────────────────────────────────────────────
const state = {
  camActive:       false,
  detecting:       false,
  drawingZone:     false,
  zoneComplete:    false,
  zonePoints:      [],   // [{x,y} normalized 0-1]
  rawZonePoints:   [],   // [{x,y} pixel on canvas]
  audioEnabled:    true,
  lastAlertTime:   0,
  intrusion:       false,
  intrFreezeFrames: 0,
  frames:          0,
  intrCount:       0,
  personsDetected: 0,
  detectTimer:     null,
  eventsSaved:     0,
  lastDetectMs:    0,
  confThreshold:   0.40,
  detectInterval:  300,
  logItems:        [],
};

// ─── DOM Refs ─────────────────────────────────────────────────
const q = id => document.getElementById(id);

const videoEl      = q('webcam-video');
const renderCanvas = q('render-canvas');
const zoneCanvas   = q('zone-canvas');
const rCtx         = renderCanvas.getContext('2d');
const zCtx         = zoneCanvas.getContext('2d');
const placeholder  = q('camera-placeholder');
const scanLine     = q('scan-line');
const alertBanner  = q('alert-banner');
const alertText    = q('alert-text');
const alertCount   = q('alert-count');
const audioEl      = q('audio-alert');
const statusDot    = q('status-dot');
const statusLabel  = q('status-label');

// Buttons
const btnStartCam    = q('btn-start-cam');
const btnStopCam     = q('btn-stop-cam');
const btnDrawZone    = q('btn-draw-zone');
const btnCloseZone   = q('btn-close-zone');
const btnResetZone   = q('btn-reset-zone');
const btnStartDetect = q('btn-start-detect');
const btnStopDetect  = q('btn-stop-detect');
const btnAudioToggle = q('btn-audio-toggle');
const btnClearLog    = q('btn-clear-log');

// Stats
const statFrames    = q('stat-frames');
const statPersons   = q('stat-persons');
const statIntrusion = q('stat-intrusions');
const statFps       = q('stat-fps');

// Sliders
const confSlider    = q('conf-slider');
const confValue     = q('conf-value');
const intervalSlider = q('interval-slider');
const intervalValue  = q('interval-value');

// Zone info
const zonePointCount = q('zone-point-count');
const zoneStatusText = q('zone-status-text');

// Status bar
const vsbDetectLabel = q('vsb-detect-label');
const vsbDot         = q('vsb-detect-status').querySelector('.vsb-dot');
const vsbZoneLabel   = q('vsb-zone-label');
const vsbConfLabel   = q('vsb-conf-label');
const vsbTime        = q('vsb-time');

// Hardware badge
const hwBadgeLabel = q('hw-badge-label');
const hwBadgeDot   = q('hw-badge-dot');

// Log
const logList   = q('log-list');
const logEmpty  = q('log-empty');

// ─── Utility ──────────────────────────────────────────────────
function setStatus(text, type = '') {
  statusLabel.textContent = text;
  statusDot.className = 'status-dot' + (type ? ' ' + type : '');
}

function setVsbStatus(text, type = '') {
  vsbDetectLabel.textContent = text;
  vsbDot.className = 'vsb-dot' + (type ? ' ' + type : '');
}

function setDetectVsb(label, cls) {
  vsbDetectLabel.textContent = label;
  vsbDot.className = 'vsb-dot ' + cls;
}

function updateClock() {
  const now = new Date();
  vsbTime.textContent = now.toLocaleTimeString('id-ID', { hour12: false });
}

setInterval(updateClock, 1000);
updateClock();

// ─── Audio ────────────────────────────────────────────────────
function generateBeep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);
    oscillator.type = 'square';
    oscillator.frequency.setValueAtTime(880, ctx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.1);
    gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + 0.3);
  } catch (e) { /* silent fail */ }
}

function playAlert() {
  if (!state.audioEnabled) return;
  const now = Date.now();
  if (now - state.lastAlertTime < 1500) return;   // debounce
  state.lastAlertTime = now;
  generateBeep();
}

btnAudioToggle.addEventListener('click', () => {
  state.audioEnabled = !state.audioEnabled;
  q('icon-audio-on').style.display  = state.audioEnabled ? '' : 'none';
  q('icon-audio-off').style.display = state.audioEnabled ? 'none' : '';
  btnAudioToggle.title = state.audioEnabled ? 'Matikan audio alert' : 'Aktifkan audio alert';
});

// ─── Slider wiring ────────────────────────────────────────────
confSlider.addEventListener('input', () => {
  state.confThreshold = parseFloat(confSlider.value);
  confValue.textContent = state.confThreshold.toFixed(2);
  vsbConfLabel.textContent = `Conf: ${state.confThreshold.toFixed(2)}`;
});

intervalSlider.addEventListener('input', () => {
  state.detectInterval = parseInt(intervalSlider.value);
  intervalValue.textContent = state.detectInterval + 'ms';
  if (state.detecting) restartDetectLoop();
});

// ─── Canvas sizing ────────────────────────────────────────────
function resizeCanvases(w, h) {
  renderCanvas.width  = w;
  renderCanvas.height = h;
  zoneCanvas.width    = w;
  zoneCanvas.height   = h;
  renderCanvas.style.width  = '';
  renderCanvas.style.height = '';
}

// ─── Webcam ───────────────────────────────────────────────────
async function startCamera() {
  setStatus('Meminta izin webcam…', 'warning');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
      audio: false,
    });
    videoEl.srcObject = stream;
    await new Promise(res => { videoEl.onloadedmetadata = res; });
    await videoEl.play();

    const vw = videoEl.videoWidth  || 640;
    const vh = videoEl.videoHeight || 480;
    resizeCanvases(vw, vh);

    state.camActive = true;
    placeholder.style.display = 'none';
    scanLine.style.display = '';

    btnStartCam.disabled    = true;
    btnStopCam.disabled     = false;
    btnDrawZone.disabled    = false;
    btnStartDetect.disabled = false;

    setStatus('Webcam aktif', 'ready');
    setDetectVsb('Webcam aktif', 'success');

    // Draw idle frames
    requestAnimationFrame(drawIdleFrame);
  } catch (err) {
    setStatus('Gagal akses kamera', '');
    alert('Tidak dapat mengakses webcam: ' + err.message);
  }
}

function stopCamera() {
  const stream = videoEl.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  videoEl.srcObject = null;
  state.camActive = false;

  stopDetection();

  rCtx.clearRect(0, 0, renderCanvas.width, renderCanvas.height);
  zCtx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);

  placeholder.style.display = '';
  scanLine.style.display = 'none';

  btnStartCam.disabled    = false;
  btnStopCam.disabled     = true;
  btnDrawZone.disabled    = true;
  btnStartDetect.disabled = true;
  btnStopDetect.disabled  = true;

  setStatus('Webcam tidak aktif', '');
  setDetectVsb('Idle', '');
}

function drawIdleFrame() {
  if (!state.camActive || state.detecting) return;
  rCtx.drawImage(videoEl, 0, 0, renderCanvas.width, renderCanvas.height);
  drawZoneOverlay();
  requestAnimationFrame(drawIdleFrame);
}

btnStartCam.addEventListener('click', startCamera);
btnStopCam.addEventListener('click', stopCamera);

// ─── Zone Drawing ─────────────────────────────────────────────
function startDrawingZone() {
  if (!state.camActive) return;
  state.drawingZone  = true;
  state.zoneComplete = false;
  state.zonePoints   = [];
  state.rawZonePoints = [];

  zoneCanvas.classList.add('drawing');
  document.getElementById('canvas-wrapper').classList.add('drawing-mode');

  btnCloseZone.disabled  = false;
  btnResetZone.disabled  = false;
  btnDrawZone.disabled   = true;

  updateZoneInfo();
  setStatus('Mode gambar zona – klik titik-titik', 'warning');
}

function closeZonePolygon() {
  if (state.zonePoints.length < 3) {
    alert('Minimal 3 titik diperlukan untuk membentuk polygon.');
    return;
  }
  state.drawingZone  = false;
  state.zoneComplete = true;

  zoneCanvas.classList.remove('drawing');
  document.getElementById('canvas-wrapper').classList.remove('drawing-mode');

  btnCloseZone.disabled = true;

  // Send zone to backend
  sendZoneToBackend();
  drawZoneOverlay();
  updateZoneInfo();
  setStatus('Zona ditetapkan – siap deteksi', 'ready');
}

function resetZone() {
  state.drawingZone   = false;
  state.zoneComplete  = false;
  state.zonePoints    = [];
  state.rawZonePoints = [];

  zCtx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
  zoneCanvas.classList.remove('drawing');
  document.getElementById('canvas-wrapper').classList.remove('drawing-mode');

  btnDrawZone.disabled  = false;
  btnCloseZone.disabled = true;

  fetch(API_BASE + '/zone/clear', { method: 'POST' }).catch(() => {});
  updateZoneInfo();
  setStatus('Zona dihapus', 'ready');
}

function updateZoneInfo() {
  const n = state.zonePoints.length;
  zonePointCount.textContent = n + ' titik';
  vsbZoneLabel.textContent   = 'Zona: ' + n + ' titik';

  if (state.zoneComplete) {
    zoneStatusText.textContent = '✅ Polygon tertutup';
  } else if (state.drawingZone) {
    zoneStatusText.textContent = '✏️ Sedang menggambar…';
  } else if (n === 0) {
    zoneStatusText.textContent = 'Belum ada zona';
  } else {
    zoneStatusText.textContent = 'Zona terbuka';
  }
}

async function sendZoneToBackend() {
  try {
    await fetch(API_BASE + '/zone', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ polygon: state.zonePoints }),
    });
  } catch (e) {
    console.warn('Zone send failed:', e);
  }
}

// Zone canvas click
zoneCanvas.addEventListener('click', (e) => {
  if (!state.drawingZone) return;

  const rect = zoneCanvas.getBoundingClientRect();
  const scaleX = zoneCanvas.width  / rect.width;
  const scaleY = zoneCanvas.height / rect.height;
  const px = (e.clientX - rect.left)  * scaleX;
  const py = (e.clientY - rect.top)   * scaleY;

  state.rawZonePoints.push({ x: px, y: py });
  state.zonePoints.push({
    x: px / zoneCanvas.width,
    y: py / zoneCanvas.height,
  });

  drawZoneOverlay();
  updateZoneInfo();
});

function drawZoneOverlay() {
  zCtx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
  const pts = state.rawZonePoints;
  if (pts.length === 0) return;

  const closed = state.zoneComplete;

  // Fill
  zCtx.beginPath();
  zCtx.moveTo(pts[0].x, pts[0].y);
  pts.slice(1).forEach(p => zCtx.lineTo(p.x, p.y));
  if (closed) zCtx.closePath();
  zCtx.fillStyle = 'rgba(245, 166, 35, 0.12)';
  zCtx.fill();

  // Stroke
  zCtx.beginPath();
  zCtx.moveTo(pts[0].x, pts[0].y);
  pts.slice(1).forEach(p => zCtx.lineTo(p.x, p.y));
  if (closed) zCtx.closePath();
  zCtx.strokeStyle = '#F5A623';
  zCtx.lineWidth   = 2;
  zCtx.setLineDash(closed ? [] : [6, 4]);
  zCtx.stroke();
  zCtx.setLineDash([]);

  // Points
  pts.forEach((p, i) => {
    zCtx.beginPath();
    zCtx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    zCtx.fillStyle   = '#FFD060';
    zCtx.strokeStyle = '#1A0E00';
    zCtx.lineWidth   = 1.5;
    zCtx.fill();
    zCtx.stroke();

    // Index label
    zCtx.fillStyle = '#1A0E00';
    zCtx.font      = 'bold 9px Inter, sans-serif';
    zCtx.textAlign = 'center';
    zCtx.textBaseline = 'middle';
    zCtx.fillText(i + 1, p.x, p.y);
  });
}

btnDrawZone.addEventListener('click',  startDrawingZone);
btnCloseZone.addEventListener('click', closeZonePolygon);
btnResetZone.addEventListener('click', resetZone);

// ─── Automatic CCTV Recording ─────────────────────────────

let cctvRecorder = null;
let cctvChunks = [];
let cctvStopTimer = null;
let cctvRecording = false;

const CCTV_POST_INTRUSION_DELAY = 5000;

function startCCTVRecording() {
    if (cctvRecording) {
        clearTimeout(cctvStopTimer);
        return;
    }

    if (!renderCanvas.captureStream) {
        console.warn("Browser tidak mendukung canvas recording.");
        return;
    }

    const stream = renderCanvas.captureStream(20);

    let mimeType = 'video/webm;codecs=vp9';

    if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'video/webm;codecs=vp8';
    }

    if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'video/webm';
    }

    cctvChunks = [];

    cctvRecorder = new MediaRecorder(stream, {
        mimeType: mimeType
    });

    cctvRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
            cctvChunks.push(event.data);
        }
    };

    cctvRecorder.onstop = () => {
        const blob = new Blob(cctvChunks, {
            type: mimeType
        });

        const url = URL.createObjectURL(blob);

        const now = new Date();

        const timestamp =
            now.getFullYear() +
            String(now.getMonth() + 1).padStart(2, '0') +
            String(now.getDate()).padStart(2, '0') + '_' +
            String(now.getHours()).padStart(2, '0') +
            String(now.getMinutes()).padStart(2, '0') +
            String(now.getSeconds()).padStart(2, '0');

        const a = document.createElement('a');

        a.href = url;
        a.download = `ZONEGUARD_INTRUSION_${timestamp}.webm`;

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        setTimeout(() => {
            URL.revokeObjectURL(url);
        }, 1000);

        console.log("CCTV recording tersimpan:", a.download);

        cctvChunks = [];
        cctvRecorder = null;
        cctvRecording = false;
    };

    cctvRecorder.start();
    cctvRecording = true;

    console.log("🚨 CCTV recording dimulai");
}

function stopCCTVRecording() {
    if (!cctvRecorder || cctvRecorder.state === 'inactive') {
        return;
    }

    cctvRecorder.stop();

    console.log("CCTV recording dihentikan");
}

function handleCCTVIntrusion(isIntrusion) {
    if (isIntrusion) {
        clearTimeout(cctvStopTimer);

        if (!cctvRecording) {
            startCCTVRecording();
        }

    } else if (cctvRecording) {

        clearTimeout(cctvStopTimer);

        cctvStopTimer = setTimeout(() => {
            stopCCTVRecording();
        }, CCTV_POST_INTRUSION_DELAY);
    }
}

// ─── Detection loop ───────────────────────────────────────────
function startDetection() {
  if (!state.camActive || state.detecting) return;
  if (!state.zoneComplete) {
    alert('Gambar dan tutup zona terlebih dahulu sebelum memulai deteksi.');
    return;
  }

  state.detecting = true;
  btnStartDetect.disabled = true;
  btnStopDetect.disabled  = false;
  scanLine.style.display  = '';

  setStatus('Deteksi aktif', 'active');
  setDetectVsb('Deteksi berjalan…', 'active');

  runDetectLoop();
}

function stopDetection() {
  if (!state.detecting) return;
  state.detecting = false;
  clearTimeout(state.detectTimer);

  btnStartDetect.disabled = !state.camActive || !state.zoneComplete;
  btnStopDetect.disabled  = true;

  // Reset tracker state on backend
  fetch(API_BASE + '/tracker/reset', { method: 'POST' }).catch(() => {});
  // Clear local track colors
  Object.keys(trackColors).forEach(k => delete trackColors[k]);

  clearIntrusion();
  setStatus(state.camActive ? 'Webcam aktif' : 'Webcam tidak aktif', state.camActive ? 'ready' : '');
  setDetectVsb('Dihentikan', '');
}

function restartDetectLoop() {
  clearTimeout(state.detectTimer);
  if (state.detecting) runDetectLoop();
}

async function runDetectLoop() {
  if (!state.detecting) return;

  const t0 = performance.now();

  try {
    // Capture frame from video
    const offscreen = document.createElement('canvas');
    offscreen.width  = renderCanvas.width;
    offscreen.height = renderCanvas.height;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(videoEl, 0, 0, offscreen.width, offscreen.height);
    const frameData = offscreen.toDataURL('image/jpeg', 0.75);

    state.frames++;
    statFrames.textContent = state.frames;

    const saveEvent = (state.intrFreezeFrames === 0 && state.intrusion) ||
                      (state.frames % SAVE_EVENT_EVERY === 0 && state.intrusion);

    const resp = await fetch(API_BASE + '/detect', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        frame:          frameData,
        conf_threshold: state.confThreshold,
        save_event:     saveEvent,
      }),
    });

    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    const latency = Math.round(performance.now() - t0);
    state.lastDetectMs = latency;
    statFps.textContent = latency;

    renderDetections(data, frameData);

    // Update stats
    const persons = data.detections ? data.detections.filter(d => d.class_name === 'intruder').length : 0;
    state.personsDetected = Math.max(state.personsDetected, persons);
    statPersons.textContent = state.personsDetected;

    if (data.intrusion_detected) {

      if (!state.intrusion) {
        state.intrCount++;
        statIntrusion.textContent = state.intrCount;
      }

      triggerIntrusion(data.detections);

      handleCCTVIntrusion(true);

      if (saveEvent) addLogItem(frameData, data.detections);

    } else {

      handleCCTVIntrusion(false);

      state.intrFreezeFrames = Math.max(0, state.intrFreezeFrames - 1);

      if (state.intrFreezeFrames === 0) {
        clearIntrusion();
      }
    }

  } catch (err) {
    console.warn('Detect error:', err.message);
    // Still draw idle frame on error
    rCtx.drawImage(videoEl, 0, 0, renderCanvas.width, renderCanvas.height);
    drawZoneOverlay();
  }

  state.detectTimer = setTimeout(runDetectLoop, state.detectInterval);
}

function renderDetections(data, _frameData) {
  rCtx.drawImage(videoEl, 0, 0, renderCanvas.width, renderCanvas.height);
  drawZoneOverlay();

  if (!data.detections || data.detections.length === 0) return;

  const W = renderCanvas.width;
  const H = renderCanvas.height;

  // ── Pass 1: Draw trails (behind boxes) ──────────────────────
  data.detections.forEach(det => {
    const tid   = det.track_id || 0;
    const trail = det.trail || [];
    if (trail.length < 2) return;

    const isAlert = det.class_name === 'intruder' && det.in_zone;
    const tcolor  = isAlert ? '#FF4060' : getTrackColor(tid);

    rCtx.save();
    rCtx.lineJoin = 'round';
    rCtx.lineCap  = 'round';

    // ── 1a: Draw gradient trail line ─────────────────────────
    for (let i = 1; i < trail.length; i++) {
      const progress = i / trail.length;
      const alpha    = progress * 0.9;
      const thick    = progress * 3.5 + 0.5;

      const [px, py] = [trail[i-1][0] * W, trail[i-1][1] * H];
      const [nx, ny] = [trail[i][0]   * W, trail[i][1]   * H];

      rCtx.beginPath();
      rCtx.moveTo(px, py);
      rCtx.lineTo(nx, ny);
      rCtx.strokeStyle    = tcolor;
      rCtx.globalAlpha    = alpha;
      rCtx.lineWidth      = thick;
      rCtx.shadowBlur     = progress * 8;
      rCtx.shadowColor    = tcolor;
      rCtx.stroke();
    }

    // ── 1b: Draw small dot at each trail point ────────────────
    rCtx.shadowBlur = 0;
    trail.forEach((pt, i) => {
      if (i === trail.length - 1) return;  // skip head, drawn separately
      const progress  = (i + 1) / trail.length;
      const dotRadius = progress * 4 + 1;
      const alpha     = progress * 0.75;

      rCtx.globalAlpha = alpha;
      rCtx.beginPath();
      rCtx.arc(pt[0] * W, pt[1] * H, dotRadius, 0, Math.PI * 2);
      rCtx.fillStyle = tcolor;
      rCtx.fill();
    });

    // ── 1c: Large glowing HEAD dot (current position) ─────────
    const head = trail[trail.length - 1];
    const hx   = head[0] * W;
    const hy   = head[1] * H;

    rCtx.globalAlpha = 1;

    // Outer glow ring
    rCtx.beginPath();
    rCtx.arc(hx, hy, 10, 0, Math.PI * 2);
    rCtx.fillStyle   = tcolor + '40';  // 25% opacity
    rCtx.shadowBlur  = 16;
    rCtx.shadowColor = tcolor;
    rCtx.fill();

    // Inner solid dot
    rCtx.beginPath();
    rCtx.arc(hx, hy, 6, 0, Math.PI * 2);
    rCtx.fillStyle   = tcolor;
    rCtx.shadowBlur  = 12;
    rCtx.shadowColor = tcolor;
    rCtx.fill();

    // White center pinpoint
    rCtx.beginPath();
    rCtx.arc(hx, hy, 2, 0, Math.PI * 2);
    rCtx.fillStyle = '#ffffff';
    rCtx.shadowBlur = 0;
    rCtx.fill();

    rCtx.restore();
  });


  // ── Pass 2: Draw bounding boxes + labels ────────────────────
  data.detections.forEach(det => {
    const x1 = det.x1n * W;
    const y1 = det.y1n * H;
    const x2 = det.x2n * W;
    const y2 = det.y2n * H;
    const w  = x2 - x1;
    const h  = y2 - y1;
    const tid = det.track_id || 0;

    const isIntruder = det.class_name === 'intruder';
    const inZone     = det.in_zone;
    const isAlert    = isIntruder && inZone;
    const boxColor   = isAlert ? '#FF4060' : (isIntruder ? getTrackColor(tid) : '#F5A623');

    // Bounding box
    rCtx.save();
    rCtx.shadowBlur  = isAlert ? 24 : 10;
    rCtx.shadowColor = boxColor;
    rCtx.strokeStyle = boxColor;
    rCtx.lineWidth   = isAlert ? 3 : 2;
    rCtx.strokeRect(x1, y1, w, h);
    rCtx.restore();

    // Fill
    rCtx.fillStyle = isAlert
      ? 'rgba(255, 64, 96, 0.08)'
      : 'rgba(59, 158, 255, 0.05)';
    rCtx.fillRect(x1, y1, w, h);

    // Corner marks
    const cs = 14;
    rCtx.strokeStyle = boxColor;
    rCtx.lineWidth   = 2.5;
    [[x1,y1,1,1],[x2,y1,-1,1],[x1,y2,1,-1],[x2,y2,-1,-1]].forEach(([cx,cy,dx,dy]) => {
      rCtx.beginPath();
      rCtx.moveTo(cx + dx * cs, cy);
      rCtx.lineTo(cx, cy);
      rCtx.lineTo(cx, cy + dy * cs);
      rCtx.stroke();
    });

    // ── Track ID badge (#N) top-right corner ────────────────
    const badge  = `#${tid}`;
    rCtx.font    = 'bold 10px "JetBrains Mono", monospace';
    const bw     = rCtx.measureText(badge).width + 10;
    const bh     = 18;
    const bx     = x2 - bw;
    const by     = y1;

    rCtx.fillStyle = boxColor;
    rCtx.beginPath();
    rCtx.roundRect(bx, Math.max(0, by - bh), bw, bh, [0, 4, 0, 4]);
    rCtx.fill();
    rCtx.fillStyle = '#000';
    rCtx.fillText(badge, bx + 5, Math.max(bh - 5, by - 5));

    // ── Class + confidence label (top-left) ─────────────────
    const label = isAlert
      ? `⚠ INTRUSI ${(det.confidence * 100).toFixed(0)}%`
      : `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;

    rCtx.font  = 'bold 11px "JetBrains Mono", monospace';
    const lw   = rCtx.measureText(label).width + 14;
    const lh   = 20;
    const lx   = x1;
    const ly   = y1 - lh - 2;

    rCtx.fillStyle = boxColor;
    rCtx.beginPath();
    rCtx.roundRect(lx, Math.max(0, ly), lw, lh, 3);
    rCtx.fill();
    rCtx.fillStyle = isAlert ? '#fff' : '#000';
    rCtx.fillText(label, lx + 7, Math.max(lh - 6, ly + lh - 6));

    // ── Foot point ──────────────────────────────────────────
    const fx = (x1 + x2) / 2;
    const fy = y2;
    rCtx.beginPath();
    rCtx.arc(fx, fy, 4, 0, Math.PI * 2);
    rCtx.fillStyle   = boxColor;
    rCtx.shadowBlur  = 6;
    rCtx.shadowColor = boxColor;
    rCtx.fill();
    rCtx.strokeStyle = '#fff';
    rCtx.lineWidth   = 1;
    rCtx.shadowBlur  = 0;
    rCtx.stroke();
  });
}

// ─── Intrusion state ──────────────────────────────────────────
function triggerIntrusion(detections) {
  state.intrusion = true;
  state.intrFreezeFrames = 5;

  const n = detections ? detections.filter(d => d.in_zone && d.class_name === 'intruder').length : 0;

  alertBanner.classList.add('active', 'pulsing');
  alertCount.textContent = n > 0 ? `${n} orang` : '';
  document.body.classList.add('intrusion-active');

  setStatus('⚠ INTRUSI TERDETEKSI!', 'danger');
  setDetectVsb('⚠ INTRUSI!', 'danger');

  playAlert();
}

function clearIntrusion() {
  state.intrusion = false;
  alertBanner.classList.remove('active', 'pulsing');
  document.body.classList.remove('intrusion-active');

  if (state.detecting) {
    setStatus('Deteksi aktif', 'active');
    setDetectVsb('Deteksi berjalan…', 'active');
  }
}

btnStartDetect.addEventListener('click', startDetection);
btnStopDetect.addEventListener('click',  stopDetection);

// ─── Event Log ────────────────────────────────────────────────
function addLogItem(frameDataUrl, detections) {
  const now       = new Date();
  const timeStr   = now.toLocaleTimeString('id-ID', { hour12: false });
  const dateStr   = now.toLocaleDateString('id-ID');
  const inZoneDets = detections ? detections.filter(d => d.in_zone && d.class_name === 'intruder') : [];
  const trackIds   = inZoneDets.map(d => d.track_id).filter(Boolean);

  const item = {
    time: timeStr, date: dateStr, thumb: frameDataUrl,
    count: inZoneDets.length,
    trackIds,
  };
  state.logItems.unshift(item);
  if (state.logItems.length > MAX_LOG_ITEMS) state.logItems.pop();

  renderLog();
}

function renderLog() {
  // Remove empty state
  logEmpty.style.display = state.logItems.length === 0 ? '' : 'none';

  // Remove existing items (keep logEmpty)
  const existing = logList.querySelectorAll('.log-item');
  existing.forEach(el => el.remove());

  state.logItems.forEach(item => {
    const div = document.createElement('div');
    div.className = 'log-item';

    const img = document.createElement('img');
    img.className = 'log-thumb';
    img.src = item.thumb;
    img.alt = 'Snapshot intrusi ' + item.time;
    img.loading = 'lazy';

    const meta = document.createElement('div');
    meta.className = 'log-meta';

    const timeEl = document.createElement('div');
    timeEl.className = 'log-time';
    timeEl.textContent = '⚠ ' + item.time + ' · ' + item.date;

    const detail = document.createElement('div');
    detail.className = 'log-detail';
    const ids = item.trackIds && item.trackIds.length ? ' (ID: ' + item.trackIds.map(i => '#' + i).join(', ') + ')' : '';
    detail.textContent = item.count + ' intruder di zona' + ids;

    meta.appendChild(timeEl);
    meta.appendChild(detail);
    div.appendChild(img);
    div.appendChild(meta);
    logList.insertBefore(div, logList.firstChild);
  });
}

btnClearLog.addEventListener('click', () => {
  state.logItems = [];
  const existing = logList.querySelectorAll('.log-item');
  existing.forEach(el => el.remove());
  logEmpty.style.display = '';
  fetch(API_BASE + '/events/clear', { method: 'POST' }).catch(() => {});
});

// ─── Hardware badge update ────────────────────────────────────
function updateHardwareBadge(data) {
  if (!hwBadgeLabel) return;

  if (!data.model_loaded) {
    hwBadgeLabel.textContent = 'Model tidak termuat';
    hwBadgeDot.style.background = '#ef4444';
    hwBadgeDot.style.boxShadow  = '0 0 6px #ef4444';
    return;
  }

  const label = data.hardware_label || `${data.backend} [${data.device}]`;
  hwBadgeLabel.textContent = label;

  // Color: blue for OpenVINO, purple for PyTorch
  const color = data.backend === 'openvino' ? '#3B9EFF' : '#C084FC';
  hwBadgeDot.style.background = color;
  hwBadgeDot.style.boxShadow  = `0 0 8px ${color}`;
}

// ─── Initial server status check ─────────────────────────────
async function checkServerStatus() {
  try {
    const resp = await fetch(API_BASE + '/status', { signal: AbortSignal.timeout(3000) });
    const data = await resp.json();
    if (data.model_loaded) {
      setStatus('Server siap · Model dimuat', 'ready');
    } else {
      setStatus('Server aktif · Model gagal dimuat', 'warning');
    }
    updateHardwareBadge(data);
  } catch (e) {
    setStatus('Server tidak terhubung – jalankan app.py', '');
    if (hwBadgeLabel) hwBadgeLabel.textContent = 'Server offline';
    console.warn('Server not reachable:', e);
  }
}

checkServerStatus();
setInterval(checkServerStatus, 15000);   // re-check every 15s

// ─── Prevent context menu on canvases ────────────────────────
[renderCanvas, zoneCanvas].forEach(c => {
  c.addEventListener('contextmenu', e => e.preventDefault());
});

// ─── Keyboard shortcuts ───────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key) {
    case 'd': case 'D':
      if (!btnDrawZone.disabled)    btnDrawZone.click();
      else if (!btnCloseZone.disabled) btnCloseZone.click();
      break;
    case 'r': case 'R':
      if (!btnResetZone.disabled)   btnResetZone.click();    break;
    case 's': case 'S':
      if (!btnStartDetect.disabled) btnStartDetect.click();
      else if (!btnStopDetect.disabled) btnStopDetect.click();
      break;
    case 'm': case 'M':
      btnAudioToggle.click(); break;
    case 'Escape':
      if (state.drawingZone) resetZone(); break;
  }
});
