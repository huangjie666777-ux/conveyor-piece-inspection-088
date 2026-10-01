"use strict";

const $ = (sel) => document.querySelector(sel);

function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.className = isError ? "err" : "";
  el.style.display = "block";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.style.display = "none"; }, 4000);
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
}

async function refreshStatus() {
  const s = await api("/api/status");
  const built = s.built_at
    ? `已构建（${new Date(s.built_at).toLocaleString()}，库 ${s.memory_bank_items} 项）`
    : "尚未构建";
  const thr = s.threshold === null ? "–" : s.threshold.toFixed(4) +
    (s.threshold_is_zero ? "（阈值为零，见检测说明）" : "");
  $("#status").innerHTML = `
    参考图：<strong>${s.reference_count}</strong> 张 ·
    校准图：<strong>${s.calibration_count}</strong> 张 ·
    记忆库：<strong>${built}</strong> ·
    阈值：<strong>${thr}</strong>`;
  return s;
}

async function refreshGroup(group) {
  const listId = group === "reference" ? "#reference-list" : "#calibration-list";
  const data = await api(`/api/samples/${group}`);
  const row = $(listId);
  row.innerHTML = "";
  for (const sample of data.samples) {
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML = `
      <img src="/api/samples/${group}/${encodeURIComponent(sample.id)}/image" alt="">
      <div class="name">${sample.filename}</div>`;
    const btn = document.createElement("button");
    btn.className = "danger";
    btn.textContent = "删除";
    btn.onclick = async () => {
      try {
        await api(`/api/samples/${group}/${encodeURIComponent(sample.id)}`, {
          method: "DELETE",
        });
        await Promise.all([refreshStatus(), refreshGroup(group)]);
      } catch (err) { toast(err.message, true); }
    };
    div.appendChild(btn);
    row.appendChild(div);
  }
}

async function uploadFiles(group, files) {
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      await api(`/api/samples/${group}`, { method: "POST", body: form });
    } catch (err) {
      toast(`${file.name}：${err.message}`, true);
    }
  }
  await Promise.all([refreshStatus(), refreshGroup(group)]);
}

$("#reference-input").addEventListener("change", (e) => {
  uploadFiles("reference", e.target.files); e.target.value = "";
});
$("#calibration-input").addEventListener("change", (e) => {
  uploadFiles("calibration", e.target.files); e.target.value = "";
});

$("#build-btn").addEventListener("click", async () => {
  const msg = $("#build-msg");
  msg.textContent = "构建中（CPU 推理，请稍候）…";
  msg.className = "msg";
  try {
    await api("/api/build", { method: "POST" });
    msg.textContent = "构建成功，旧检测请求不会混用新库。";
    msg.className = "msg ok";
    await refreshStatus();
  } catch (err) {
    msg.textContent = `构建失败，已保留旧库：${err.message}`;
    msg.className = "msg err";
  }
});

let currentOriginal = null;
let currentHeat = null;

function renderOverlay() {
  if (!currentOriginal || !currentHeat) return;
  const alpha = Number($("#alpha").value) / 100;
  $("#alpha-val").textContent = `${Math.round(alpha * 100)}%`;
  const canvas = document.createElement("canvas");
  canvas.width = currentOriginal.naturalWidth;
  canvas.height = currentOriginal.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(currentOriginal, 0, 0);
  ctx.globalAlpha = alpha;
  ctx.drawImage(currentHeat, 0, 0);
  $("#overlay").src = canvas.toDataURL("image/png");
}

async function inspectFile(file) {
  const form = new FormData();
  form.append("file", file);
  let result;
  try {
    result = await api("/api/inspect", { method: "POST", body: form });
  } catch (err) { toast(err.message, true); return; }
  $("#inspect-result").classList.remove("hidden");
  $("#metric-score").textContent = result.score.toFixed(4);
  $("#metric-threshold").textContent = result.threshold.toFixed(4);
  const verdict = $("#metric-verdict");
  verdict.textContent = result.is_anomaly ? "异常（超过阈值）" : "正常";
  verdict.className = result.is_anomaly ? "anomaly" : "normal";
  $("#metric-vmax").textContent = result.threshold_is_zero
    ? "阈值为 0，改用本图距离最大值（固定色标不可用时的明确回退）"
    : `固定为 2 × 阈值 = ${result.color_scale_vmax.toFixed(4)}（不逐图拉伸）`;
  $("#zero-note").hidden = !result.threshold_is_zero;
  currentOriginal = new Image();
  currentHeat = new Image();
  currentOriginal.onload = renderOverlay;
  currentHeat.onload = renderOverlay;
  currentOriginal.src = "data:image/png;base64," + result.original;
  currentHeat.src = "data:image/png;base64," + result.heatmap;
}

$("#inspect-input").addEventListener("change", (e) => {
  if (e.target.files[0]) inspectFile(e.target.files[0]);
  e.target.value = "";
});
$("#alpha").addEventListener("input", renderOverlay);

document.querySelectorAll(".ex-item button").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const file = btn.dataset.file;
    const group = btn.dataset.group;
    try {
      const resp = await fetch(`/examples/${file}`);
      const blob = await resp.blob();
      const upload = new File([blob], file, { type: blob.type });
      if (group === "inspect") {
        inspectFile(upload);
      } else {
        await uploadFiles(group, [upload]);
      }
    } catch (err) { toast(err.message, true); }
  });
});

(async function init() {
  try {
    await refreshStatus();
    await Promise.all([refreshGroup("reference"), refreshGroup("calibration")]);
  } catch (err) { toast(err.message, true); }
})();

// ============================================================ conveyor video
(function () {
  const v = {
    videoFile: null,
    bgUrl: null,
    videoUrl: null,
    frameW: 0,
    frameH: 0,
    dispScale: 1,
    roi: null,      // [x, y, w, h] in frame coordinates
    lineX: null,
    drag: null,
    job: null,
    poll: null,
    tracks: null,   // decoded frame -> {track_id: [x,y,w,h,observed,crossed]}
    crossingFrames: {},
  };

  const canvas = $("#v-setup");
  const ctx = canvas.getContext("2d");

  function setupReady() {
    $("#v-submit").disabled = !(v.videoUrl && v.bgUrl && v.roi && v.lineX !== null);
  }

  function drawSetup() {
    if (!v.frameW) return;
    const bg = $("#v-bg");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (v.bgImg && v.bgImg.complete) {
      ctx.globalAlpha = 0.8;
      ctx.drawImage(v.bgImg, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    }
    if (v.roi) {
      const [x, y, w, h] = v.roi.map((n) => n * v.dispScale);
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "rgba(245,158,11,.08)";
      ctx.fillRect(x, y, w, h);
    }
    if (v.lineX !== null) {
      const x = v.lineX * v.dispScale;
      ctx.strokeStyle = "#16a34a";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  $("#v-video").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (v.videoUrl) URL.revokeObjectURL(v.videoUrl);
    v.videoFile = file;
    v.videoUrl = URL.createObjectURL(file);
    const probe = document.createElement("video");
    probe.onloadedmetadata = () => {
      v.frameW = probe.videoWidth;
      v.frameH = probe.videoHeight;
      v.dispScale = Math.min(720 / v.frameW, 300 / v.frameH, 1);
      canvas.width = Math.round(v.frameW * v.dispScale);
      canvas.height = Math.round(v.frameH * v.dispScale);
      v.roi = null; v.lineX = null;
      drawSetup();
      setupReady();
    };
    probe.src = v.videoUrl;
  });

  $("#v-bg").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (v.bgUrl) URL.revokeObjectURL(v.bgUrl);
    v.bgUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (v.frameW && (img.naturalWidth !== v.frameW || img.naturalHeight !== v.frameH)) {
        toast("背景图尺寸必须与视频一致", true);
        v.bgUrl = null;
        return;
      }
      v.bgImg = img;
      drawSetup();
      setupReady();
    };
    img.src = v.bgUrl;
  });

  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    v.drag = {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  });
  canvas.addEventListener("mouseup", (e) => {
    if (!v.drag) return;
    const rect = canvas.getBoundingClientRect();
    const x2 = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y2 = ((e.clientY - rect.top) / rect.height) * canvas.height;
    const x1 = Math.min(v.drag.x, x2), y1 = Math.min(v.drag.y, y2);
    const w = Math.abs(x2 - v.drag.x), h = Math.abs(y2 - v.drag.y);
    if (w < 8 || h < 8) {
      // treat as click: set counting line
      v.lineX = Math.round(x2 / v.dispScale);
    } else {
      v.roi = [
        Math.round(x1 / v.dispScale), Math.round(y1 / v.dispScale),
        Math.round(w / v.dispScale), Math.round(h / v.dispScale),
      ];
    }
    v.drag = null;
    drawSetup();
    setupReady();
  });

  $("#v-submit").addEventListener("click", async () => {
    const form = new FormData();
    form.append("video", v.videoFile);
    const blob = await (await fetch(v.bgUrl)).blob();
    form.append("background", new File([blob], "background.png", { type: blob.type }));
    const [rx, ry, rw, rh] = v.roi;
    form.append("roi_x", rx); form.append("roi_y", ry);
    form.append("roi_w", rw); form.append("roi_h", rh);
    form.append("line_x", v.lineX);
    form.append("direction", $("#v-direction").value);
    $("#v-submit").disabled = true;
    $("#v-progress").classList.remove("hidden");
    try {
      v.job = await api("/api/jobs", { method: "POST", body: form });
      $("#v-cancel").hidden = false;
      pollJob();
    } catch (err) {
      toast("任务提交失败：" + err.message, true);
      $("#v-submit").disabled = false;
      $("#v-progress").classList.add("hidden");
    }
  });

  $("#v-cancel").addEventListener("click", async () => {
    if (!v.job) return;
    $("#v-cancel").disabled = true;
    try { await api("/api/jobs/" + v.job.id + "/cancel", { method: "POST" }); }
    catch (err) { toast(err.message, true); }
  });

  async function pollJob() {
    clearInterval(v.poll);
    v.poll = setInterval(async () => {
      try {
        const job = await api("/api/jobs/" + v.job.id);
        v.job = job;
        renderProgress(job);
        if (["completed", "cancelled", "failed", "interrupted"].includes(job.status)) {
          clearInterval(v.poll);
          $("#v-cancel").hidden = true;
          $("#v-submit").disabled = false;
          if (job.status === "completed") showResults(job);
        }
      } catch (err) { clearInterval(v.poll); toast(err.message, true); }
    }, 700);
  }

  function renderProgress(job) {
    const p = job.progress;
    const pct = p.total ? Math.round((p.processed / p.total) * 100) : 0;
    const label = {
      running: "处理中", completed: "已完成", cancelled: "已取消（未伪报完成）",
      failed: "失败", interrupted: "已中断（服务重启）",
    }[job.status];
    $("#v-progress").innerHTML =
      "<strong>" + label + "</strong> " + p.processed + "/" + p.total +
      " 帧（" + pct + "%）" + (job.error ? "<br><span class=msg.err>" + job.error + "</span>" : "");
  }

  function decodeTracks(job) {
    const frames = {};
    v.crossingFrames = {};
    for (const tr of job.tracks) {
      const data = tr.data;
      let f = -1, x = 0, y = 0, w = 0, h = 0;
      for (let i = 0; i < data.length; i += 6) {
        f += data[i] + 1;
        x += data[i + 1]; y += data[i + 2]; w += data[i + 3]; h += data[i + 4];
        (frames[f] = frames[f] || []).push({
          id: tr.track_id, box: [x, y, w, h], observed: data[i + 5] === 1,
          crossed: tr.crossed,
        });
      }
      if (tr.crossing_frame !== null) v.crossingFrames[tr.track_id] = tr.crossing_frame;
    }
    return frames;
  }

  function showResults(job) {
    $("#v-result").classList.remove("hidden");
    const player = $("#v-player");
    player.src = "/api/jobs/" + job.id + "/video";
    const oc = $("#v-overlay");
    oc.width = job.width; oc.height = job.height;
    v.tracks = decodeTracks(job);
    renderPieces(job);
    if (!player._bound) {
      player._bound = true;
      player.addEventListener("timeupdate", () => drawOverlay(job));
      player.addEventListener("seeked", () => drawOverlay(job));
    }
  }

  function drawOverlay(job) {
    const player = $("#v-player");
    const oc = $("#v-overlay");
    const c = oc.getContext("2d");
    c.clearRect(0, 0, oc.width, oc.height);
    const fi = Math.round(player.currentTime * job.fps);
    const items = v.tracks[fi] || [];
    for (const it of items) {
      const [x, y, w, h] = it.box;
      c.strokeStyle = it.observed ? (it.crossed ? "#16a34a" : "#2563eb") : "#94a3b8";
      c.lineWidth = Math.max(2, oc.width / 360);
      if (!it.observed) c.setLineDash([5, 4]); else c.setLineDash([]);
      c.strokeRect(x, y, w, h);
      c.setLineDash([]);
      c.fillStyle = c.strokeStyle;
      c.font = Math.round(oc.width / 60) + "px sans-serif";
      c.fillText("#" + it.id + (it.observed ? "" : " 预测"), x, Math.max(14, y - 4));
    }
    c.strokeStyle = "#16a34a";
    c.lineWidth = 2; c.setLineDash([8, 5]);
    c.beginPath();
    c.moveTo(job.config.line_x, 0); c.lineTo(job.config.line_x, oc.height);
    c.stroke(); c.setLineDash([]);
    const r = job.config.roi;
    c.strokeStyle = "rgba(245,158,11,.9)";
    c.strokeRect(r[0], r[1], r[2], r[3]);
  }

  function renderPieces(job) {
    const tbody = $("#v-pieces tbody");
    tbody.innerHTML = "";
    for (const piece of job.pieces) {
      const tr = document.createElement("tr");
      tr.className = "verdict-" + piece.verdict;
      const cells = [
        piece.track_id, piece.time_seconds.toFixed(2),
        piece.score === null ? "待复核" : piece.score.toFixed(3),
        piece.threshold.toFixed(3),
        { ok: "合格", defect: "缺陷", review: "待复核" }[piece.verdict],
      ];
      for (const val of cells) {
        const td = document.createElement("td");
        td.textContent = val;
        tr.appendChild(td);
      }
      tr.onclick = () => {
        $("#v-player").currentTime = piece.time_seconds;
        drawOverlay(job);
        showEvidence(job, piece);
      };
      tbody.appendChild(tr);
    }
  }

  async function showEvidence(job, piece) {
    const box = $("#v-evidence");
    box.classList.remove("hidden");
    $("#v-ev-title").textContent = "工件 #" + piece.track_id + " 证据";
    const base = "/api/jobs/" + job.id + "/evidence/" + piece.track_id + "/";
    const crop = $("#v-ev-crop"), heat = $("#v-ev-heat");
    if (piece.has_crop) {
      crop.src = base + "crop?t=" + piece.frame;
      crop.dataset.url = base + "crop";
    } else { crop.removeAttribute("src"); }
    if (piece.has_heatmap) {
      heat.src = base + "heatmap?t=" + piece.frame;
    } else { heat.removeAttribute("src"); }
    $("#v-ev-note").textContent = piece.review_reason ||
      (piece.verdict === "defect" ? "分数超过快照阈值，判定缺陷。" : "分数未超过快照阈值，判定合格。");
  }
})();
