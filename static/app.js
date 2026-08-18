const PALETTE = [
  "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff",
  "#00ffff", "#ff8000", "#8000ff", "#00ff80", "#ff0080",
];

const $ = (id) => document.getElementById(id);

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
    stopMotion();
  });
});

function status(el, msg, isError = false) {
  el.textContent = msg;
  el.className = "status" + (isError ? " error" : "");
}

async function loadImageLists() {
  try {
    const res = await fetch("/api/images");
    const data = await res.json();
    ["static-image-select", "objectness-image-select"].forEach((id) => {
      const sel = $(id);
      sel.innerHTML = "";
      data.images.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      });
    });
  } catch (e) {
    console.error(e);
  }
}

function pickImage(selectId, fileId, formData) {
  const file = $(fileId).files[0];
  if (file) {
    formData.append("file", file);
  } else {
    formData.append("image", $(selectId).value);
  }
}

function dataURL(b64) {
  return "data:image/png;base64," + b64;
}

/* ---------------- Static ---------------- */

async function runStatic() {
  const st = $("static-status");
  status(st, "Processing...");
  const fd = new FormData();
  pickImage("static-image-select", "static-file", fd);
  try {
    const res = await fetch("/api/static", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "error");
    $("static-original").src = dataURL(data.images.original);
    $("static-spectral").src = dataURL(data.images.spectral);
    $("static-fine").src = dataURL(data.images.fine);
    $("static-threshold").src = dataURL(data.images.threshold);
    $("static-roi").src = dataURL(data.images.roi);
    $("static-roi-caption").textContent = `ROIs (${data.count})`;
    status(st, `Done — ${data.count} ROI(s) from ${data.name}`);
  } catch (e) {
    status(st, e.message, true);
  }
}

$("static-run").addEventListener("click", runStatic);

/* ---------------- Objectness ---------------- */

const objMax = $("objectness-max");
objMax.addEventListener("input", () => {
  $("objectness-max-label").textContent = objMax.value;
});

async function runObjectness() {
  const st = $("objectness-status");
  status(st, "Processing...");
  const fd = new FormData();
  pickImage("objectness-image-select", "objectness-file", fd);
  fd.append("max_detections", objMax.value);
  try {
    const res = await fetch("/api/objectness", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "error");
    $("objectness-img").src = dataURL(data.image);
    $("objectness-caption").textContent = `Proposals (${data.count})`;
    const list = $("objectness-list");
    list.innerHTML = "";
    data.proposals.forEach((p, i) => {
      const li = document.createElement("li");
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = PALETTE[i % PALETTE.length];
      li.appendChild(sw);
      li.appendChild(document.createTextNode(
        `#${i + 1}: (${p.x1}, ${p.y1}) → (${p.x2}, ${p.y2})`
      ));
      list.appendChild(li);
    });
    status(st, `Done — ${data.count} proposal(s) from ${data.name}`);
  } catch (e) {
    status(st, e.message, true);
  }
}

$("objectness-run").addEventListener("click", runObjectness);

/* ---------------- Motion ---------------- */

let webcamStream = null;
let webcamTimer = null;
let sessionId = "";

function showMotion(mode) {
  $("motion-stream-box").hidden = mode !== "stream";
  $("motion-webcam-box").hidden = mode !== "webcam";
  if (mode === "stream") {
    stopWebcam();
  } else if (mode === "webcam") {
    $("motion-stream").src = "";
  }
}

function stopMotion() {
  stopWebcam();
  $("motion-stream").src = "";
}

function stopWebcam() {
  if (webcamTimer) { clearInterval(webcamTimer); webcamTimer = null; }
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  $("motion-webcam-video").srcObject = null;
}

$("motion-synthetic").addEventListener("click", () => {
  status($("motion-status"), "Streaming synthetic motion...");
  showMotion("stream");
  $("motion-stream").src = "/api/motion/synthetic?" + Date.now();
});

$("motion-upload").addEventListener("click", async () => {
  const st = $("motion-status");
  const file = $("motion-file").files[0];
  if (!file) { status(st, "Choose a video file first", true); return; }
  status(st, "Uploading...");
  const fd = new FormData();
  fd.append("video", file);
  try {
    const res = await fetch("/api/motion/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "error");
    showMotion("stream");
    $("motion-stream").src = "/api/motion/stream/" + data.id;
    status(st, `Streaming processed video: ${data.name}`);
  } catch (e) {
    status(st, e.message, true);
  }
});

$("motion-stop").addEventListener("click", () => {
  stopMotion();
  status($("motion-status"), "Stopped");
});

$("motion-webcam").addEventListener("click", async () => {
  const st = $("motion-status");
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (e) {
    status(st, "Webcam access denied or unavailable", true);
    return;
  }
  const video = $("motion-webcam-video");
  video.srcObject = webcamStream;
  await video.play();
  showMotion("webcam");
  status(st, "Webcam live — computing motion saliency");
  sessionId = "";

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  webcamTimer = setInterval(() => {
    if (video.videoWidth === 0) return;
    const scale = 500 / video.videoWidth;
    canvas.width = 500;
    canvas.height = Math.round(video.videoHeight * scale);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const fd = new FormData();
      fd.append("frame", blob, "frame.jpg");
      if (sessionId) fd.append("session_id", sessionId);
      fetch("/api/motion/frame", { method: "POST", body: fd })
        .then((res) => {
          const sid = res.headers.get("X-Session-Id");
          if (sid) sessionId = sid;
          return res.blob();
        })
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          $("motion-webcam-map").src = url;
        })
        .catch(() => {});
    }, "image/jpeg", 0.8);
  }, 66);
});

loadImageLists();
