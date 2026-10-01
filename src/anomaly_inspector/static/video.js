"use strict";

(function () {
  const $ = (sel) => document.querySelector(sel);

  function toast(message, isError = false) {
    const el = $("#toast");
    el.textContent = message;
    el.className = isError ? "err" : "";
    el.style.display = "block";
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.style.display = "none"; }, 5000);
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

  let currentJob = null;
  let pollTimer = null;
  const trackColors = {};
  const palette = [
    "#22c55e", "#a855f7", "#f59e0b", "#06b6d4",
    "#ec4899", "#84cc16", "#eab308", "#14b8a6",
  ];
  function colorFor(id) {
    if (!trackColors[id]) trackColors[id] = palette[id % palette.length];
    return trackColors[id];
  }

  async function validateFiles() {
    const video = $("#job-video").files[0];
    const bg = $("#job-bg").files[0];
    if (!video || !bg) { toast("请同时选择视频和空背景图", true); return; }
    const msg = $("#job-validate-msg");
    msg.textContent = "正在校验视频尺寸/帧率/可解码性…";
    msg.className = "msg";
    const form = new FormData();
    form.append("video", video);
    form.append("background", bg);
    try {
      const info = await api("/api/video/validate", { method: "POST", body: form });
      const s = info.suggested;
      $("#job-roi").value = s.roi.join(",");
      $("#job-line").value = s.line_x;
      $("#job-direction").value = s.direction;
      $("#job-diff").value = s.diff_threshold;
      $("#job-min-area").value = s.min_area;
      $("#job-max-area").value = s.max_piece_area;
      $("#job-min-side").value = s.min_side;
      $("#job-max-side").value = s.max_side;
      $("#job-params").hidden = false;
      $("#job-start").disabled = false;
      msg.textContent =
        `视频可处理：${info.width}x${info.height}，${info.fps.toFixed(1)} fps，` +
        `${info.frames} 帧，${info.duration_sec.toFixed(1)} 秒（参数已预填，可调整）`;
      msg.className = "msg ok";
    } catch (err) {
      msg.textContent = err.message;
      msg.className = "msg err";
    }
  }

  function formParams() {
    return {
      roi: $("#job-roi").value.trim(),
      line_x: Number($("#job-line").value),
      direction: $("#job-direction").value,
      diff_threshold: Number($("#job-diff").value),
      min_area: Number($("#job-min-area").value),
      max_piece_area: Number($("#job-max-area").value),
      min_side: Number($("#job-min-side").value),
      max_side: Number($("#job-max-side").value),
    };
  }

  async function startJob() {
    const video = $("#job-video").files[0];
    const bg = $("#job-bg").files[0];
    if (!video || !bg) { toast("请先选择文件并校验", true); return; }
    const form = new FormData();
    form.append("video", video);
    form.append("background", bg);
    for (const [k, v] of Object.entries(formParams())) form.append(k, v);
    $("#job-start").disabled = true;
    $("#job-cancel").disabled = false;
    try {
      currentJob = await api("/api/jobs", { method: "POST", body: form });
      toast(`任务 ${currentJob.job_id} 已开始`);
      pollProgress();
    } catch (err) {
      toast(err.message, true);
      $("#job-start").disabled = false;
      $("#job-cancel").disabled = true;
    }
  }

  async function cancelJob() {
    if (!currentJob) return;
    $("#job-cancel").disabled = true;
    try {
      const job = await api(`/api/jobs/${currentJob.job_id}/cancel`, {
        method: "POST",
      });
      toast(`任务已${job.status === "cancelled" ? "取消" : "标记"}，不会伪报完成`);
      await renderJob(job);
      refreshHistory();
    } catch (err) { toast(err.message, true); }
  }

  async function pollProgress() {
    if (!currentJob) return;
    clearTimeout(pollTimer);
    try {
      const [job, prog] = await Promise.all([
        api(`/api/jobs/${currentJob.job_id}`),
        api(`/api/jobs/${currentJob.job_id}/progress`),
      ]);
      currentJob = job;
      const msg = $("#job-progress-msg");
      msg.textContent =
        `状态：${statusZh(job.status)} · 帧 ${prog.processed_frames}/` +
        `${prog.total_frames}（${prog.percent}%）· 已过线 ${prog.counted} 件`;
      msg.className = job.status === "error" ? "msg err" : "msg";
      if (job.status === "running") {
        pollTimer = setTimeout(pollProgress, 500);
      } else {
        $("#job-start").disabled = false;
        $("#job-cancel").disabled = true;
        await renderJob(job);
        refreshHistory();
      }
    } catch (err) {
      toast(err.message, true);
      pollTimer = setTimeout(pollProgress, 1500);
    }
  }

  function statusZh(s) {
    return {
      running: "处理中", completed: "已完成", cancelled: "已取消",
      error: "失败", interrupted: "中断（服务重启）",
    }[s] || s;
  }

  function verdictZh(v) {
    return { ok: "合格", anomaly: "异常", review_required: "待复核" }[v] || v;
  }

  async function renderJob(job) {
    currentJob = job;
    if (job.status === "completed") {
      $("#job-replay").hidden = false;
      const video = $("#job-video-el");
      video.src = `/api/jobs/${job.job_id}/video`;
      video.width = job.width;
      setupCanvas(job);
      renderResults(job);
    } else {
      $("#job-replay").hidden = true;
    }
  }

  function setupCanvas(job) {
    const video = $("#job-video-el");
    const canvas = $("#job-canvas");
    canvas.width = job.width;
    canvas.height = job.height;
    const draw = () => {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const fps = job.fps;
      const fi = Math.min(
        job.total_frames - 1,
        Math.round(video.currentTime * fps)
      );
      // Detection ROI.
      const [rx1, ry1, rx2, ry2] = job.roi;
      ctx.strokeStyle = "rgba(255,255,255,.8)";
      ctx.setLineDash([8, 6]);
      ctx.lineWidth = 2;
      ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
      ctx.setLineDash([]);
      // Counting line.
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(job.line_x, ry1);
      ctx.lineTo(job.line_x, ry2);
      ctx.stroke();
      const dir = job.direction === "ltr" ? "▶" : "◀";
      ctx.fillStyle = "#22c55e";
      ctx.font = "22px sans-serif";
      ctx.fillText(dir, job.line_x + (job.direction === "ltr" ? 8 : -24), ry1 + 24);
      // Track paths: full history up to current frame.
      for (const track of job.tracks) {
        const pts = track.points.filter((p) => p[0] <= fi);
        if (!pts.length) continue;
        ctx.strokeStyle = colorFor(track.track_id);
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach((p, i) => i ? ctx.lineTo(p[1], p[2]) : ctx.moveTo(p[1], p[2]));
        ctx.stroke();
        const last = pts[pts.length - 1];
        ctx.beginPath();
        ctx.arc(last[1], last[2], 5, 0, Math.PI * 2);
        ctx.fillStyle = colorFor(track.track_id);
        ctx.fill();
        ctx.fillStyle = "#111";
        ctx.font = "bold 13px sans-serif";
        ctx.fillText(String(track.track_id), last[1] + 7, last[2] - 7);
      }
      if (job.merged_frames.includes(fi)) {
        ctx.fillStyle = "rgba(220,38,38,.85)";
        ctx.fillRect(12, 12, 300, 28);
        ctx.fillStyle = "#fff";
        ctx.font = "16px sans-serif";
        ctx.fillText("当前帧存在粘连候选，未合并轨迹", 22, 32);
      }
      requestAnimationFrame(draw);
    };
    video.ontimeupdate = null;
    draw();
  }

  function renderResults(job) {
    const box = $("#job-results");
    box.innerHTML = "";
    if (!job.results.length) {
      box.innerHTML = '<p class="hint">没有工件按指定方向过线。</p>';
      return;
    }
    for (const r of job.results) {
      const btn = document.createElement("button");
      btn.className = "piece-chip " + r.verdict;
      btn.style.borderColor = colorFor(r.track_id);
      btn.textContent =
        `#${r.track_id} · ${r.time_sec.toFixed(2)}s · ${verdictZh(r.verdict)}` +
        (r.score !== null ? ` · ${r.score.toFixed(3)}` : "");
      btn.onclick = () => {
        const video = $("#job-video-el");
        video.currentTime = r.time_sec;
        video.play().catch(() => {});
        showEvidence(job, r);
      };
      box.appendChild(btn);
    }
  }

  function showEvidence(job, r) {
    $("#job-evidence").hidden = false;
    $("#ev-title").textContent = `工件 #${r.track_id} 证据`;
    $("#ev-id").textContent = r.track_id;
    $("#ev-time").textContent = `${r.time_sec.toFixed(3)} 秒（帧 ${r.frame}）`;
    $("#ev-score").textContent = r.score === null ? "–" : r.score.toFixed(4);
    $("#ev-thr").textContent = Number(r.threshold).toFixed(4);
    const verdict = $("#ev-verdict");
    verdict.textContent = verdictZh(r.verdict);
    verdict.className = r.verdict;
    $("#ev-reason").textContent = r.reason || "";
    if (r.has_evidence) {
      $("#ev-crop").src =
        `/api/jobs/${job.job_id}/evidence/${r.track_id}/crop.png`;
      $("#ev-heat").src =
        `/api/jobs/${job.job_id}/evidence/${r.track_id}/heat.png`;
    } else {
      $("#ev-crop").src = "";
      $("#ev-heat").src = "";
    }
  }

  async function refreshHistory() {
    const box = $("#job-history");
    try {
      const data = await api("/api/jobs");
      box.innerHTML = "";
      for (const job of data.jobs) {
        const btn = document.createElement("button");
        const anomaly = job.results.filter((r) => r.verdict === "anomaly").length;
        const review = job.results.filter((r) => r.verdict === "review_required").length;
        btn.textContent =
          `${job.job_id} · ${statusZh(job.status)} · ${job.results.length} 件` +
          (anomaly ? ` · 异常 ${anomaly}` : "") +
          (review ? ` · 待复核 ${review}` : "");
        btn.onclick = () => renderJob(job);
        if (job.status !== "completed") btn.className = "inactive";
        box.appendChild(btn);
      }
    } catch (err) { /* history is best-effort */ }
  }

  $("#job-validate").addEventListener("click", validateFiles);
  $("#job-start").addEventListener("click", startJob);
  $("#job-cancel").addEventListener("click", cancelJob);
  refreshHistory();
})();
