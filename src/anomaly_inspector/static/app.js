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
