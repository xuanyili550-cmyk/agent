"use strict";
const $ = (s) => document.querySelector(s);
const player = $("#player");
let data = null, mode = "passage", playToken = 0;
const audioCache = new Map();   // text -> objectURL

const SAMPLES = {
  passage: "Hello 你好，welcome to my English class。\nToday we will learn some new words。\n这是一个中英双语的朗读示例。",
  cards: "apple\nbanana\ncomputer\nbeautiful\nlanguage\nvideo",
};

// ---------- 基础 ----------
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast show" + (err ? " err" : "");
  clearTimeout(t._t); t._t = setTimeout(() => (t.className = "toast"), 2600);
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error?.message || `请求失败(${r.status})`); }
  return r.json();
}
async function fetchAudio(text) {
  if (audioCache.has(text)) return audioCache.get(text);
  const r = await fetch("/api/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error?.message || "合成失败"); }
  const url = URL.createObjectURL(await r.blob());
  audioCache.set(text, url); return url;
}

// ---------- 分析 & 渲染 ----------
async function analyze() {
  const text = $("#text").value.trim();
  if (!text) return toast("请先输入文本", true);
  const btn = $("#analyzeBtn"); btn.disabled = true; btn.textContent = "分析中…";
  try {
    data = await postJSON("/api/analyze", { text });
    render();
    $("#stage").classList.remove("hidden");
    $("#playAllBtn").disabled = $("#exportBtn").disabled = false;
    const s = data.stats;
    $("#stats").textContent = `${s.sentences} 句 · ${s.en_words} 个英文词` + (s.approx_ipa ? ` · ${s.approx_ipa} 词音标为近似(装 eng-to-ipa 更准)` : "");
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; btn.textContent = "分析"; }
}

function wordEl(t) {
  const w = document.createElement("span"); w.className = "word";
  const ipa = document.createElement("span"); ipa.className = "ipa" + (t.ipa_source === "approx" ? " approx" : "");
  ipa.textContent = t.ipa || ""; if (t.ipa_source === "approx") ipa.title = "近似音标";
  const word = document.createElement("span"); word.className = "w"; word.textContent = t.text;
  w.append(ipa, word);
  w.onclick = (ev) => { ev.stopPropagation(); playText(t.text); };
  return w;
}

function render() {
  const box = $("#render"); box.innerHTML = "";
  if (mode === "cards") {
    box.className = "cards";
    const seen = new Set();
    for (const se of data.sentences) for (const t of se.tokens) {
      if (t.kind !== "en" || seen.has(t.text.toLowerCase())) continue;
      seen.add(t.text.toLowerCase());
      const c = document.createElement("div"); c.className = "card";
      c.innerHTML = `<div class="w">${t.text}</div><div class="ipa">${t.ipa || ""}</div>`;
      c.onclick = () => playText(t.text); box.appendChild(c);
    }
    if (!box.children.length) toast("单词卡模式：请输入英文单词", true);
    return;
  }
  box.className = "render";
  data.sentences.forEach((se) => {
    const s = document.createElement("div"); s.className = "sentence"; s.dataset.index = se.index;
    const sweep = document.createElement("div"); sweep.className = "sweep"; s.appendChild(sweep);
    se.tokens.forEach((t) => {
      if (t.kind === "en") s.appendChild(wordEl(t));
      else { const sp = document.createElement("span"); sp.className = "tok " + (t.kind === "cjk" ? "cjk" : ""); sp.textContent = t.text; s.appendChild(sp); }
    });
    s.onclick = () => playSentence(s, se.text);
    box.appendChild(s);
  });
}

// ---------- 播放 ----------
function clearActive() {
  document.querySelectorAll(".sentence.active").forEach((e) => { e.classList.remove("active"); e.querySelector(".sweep").style.width = "0"; });
}
function playText(text) {   // 单词/单句即点即读
  playToken++; return _play(text, null);
}
function playSentence(el, text) {
  playToken++; const my = playToken;
  clearActive(); el.classList.add("active");
  return _play(text, el).then(() => { if (playToken === my) { el.classList.remove("active"); el.querySelector(".sweep").style.width = "0"; } });
}
function _play(text, el) {
  const my = playToken;
  return fetchAudio(text).then((url) => new Promise((resolve) => {
    player.src = url;
    const sweep = el ? el.querySelector(".sweep") : null;
    const onTime = () => { if (sweep && player.duration) sweep.style.width = (player.currentTime / player.duration * 100) + "%"; };
    const done = () => { player.removeEventListener("timeupdate", onTime); player.removeEventListener("ended", done); resolve(); };
    player.addEventListener("timeupdate", onTime); player.addEventListener("ended", done);
    if (my !== playToken) return resolve();
    $("#stopBtn").disabled = false;
    player.play().catch(() => resolve());
  })).catch((e) => toast(e.message, true));
}
async function playAll() {
  if (!data) return; playToken++; const my = playToken; $("#stopBtn").disabled = false;
  for (const se of data.sentences) {
    if (my !== playToken) break;
    const el = document.querySelector(`.sentence[data-index="${se.index}"]`);
    if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); await playSentence(el, se.text); }
    else await playText(se.text);
  }
  if (my === playToken) $("#stopBtn").disabled = true;
}
function stop() { playToken++; player.pause(); player.currentTime = 0; clearActive(); $("#stopBtn").disabled = true; }

// ---------- 导出 MP4 ----------
async function exportVideo() {
  const btn = $("#exportBtn"); btn.disabled = true;
  $("#exportPanel").classList.remove("hidden");
  $("#preview").classList.add("hidden"); $("#downloadLink").classList.add("hidden");
  $("#exportStatus").textContent = "排队中…"; $("#progressBar").style.width = "5%";
  try {
    const { job_id } = await postJSON("/api/export", { text: $("#text").value.trim() });
    await poll(job_id);
  } catch (e) { toast(e.message, true); $("#exportStatus").textContent = "失败"; }
  finally { btn.disabled = false; }
}
function poll(jobId) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const j = await (await fetch(`/api/jobs/${jobId}`)).json();
        $("#progressBar").style.width = Math.max(5, j.progress * 100) + "%";
        $("#exportStatus").textContent = { queued: "排队中…", running: `合成中… ${Math.round(j.progress * 100)}%`, done: "完成", error: "失败" }[j.status] || j.status;
        if (j.status === "done") {
          const v = $("#preview"); v.src = j.download_url; v.classList.remove("hidden");
          const a = $("#downloadLink"); a.href = j.download_url; a.classList.remove("hidden");
          toast("视频已生成"); return resolve();
        }
        if (j.status === "error") { toast(j.error || "导出失败", true); return reject(new Error(j.error)); }
        setTimeout(tick, 800);
      } catch (e) { reject(e); }
    };
    tick();
  });
}

// ---------- UI 绑定 ----------
$("#analyzeBtn").onclick = analyze;
$("#playAllBtn").onclick = playAll;
$("#stopBtn").onclick = stop;
$("#exportBtn").onclick = exportVideo;
document.querySelectorAll(".mode").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".mode").forEach((x) => x.classList.remove("active"));
  b.classList.add("active"); mode = b.dataset.mode; if (data) render();
});
document.querySelectorAll(".chip").forEach((b) => b.onclick = () => { $("#text").value = SAMPLES[b.dataset.sample]; });
$("#themeBtn").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", cur); localStorage.setItem("theme", cur);
  $("#themeBtn").textContent = cur === "dark" ? "☀️" : "🌙";
};
(function initTheme() {
  const t = localStorage.getItem("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", t); $("#themeBtn").textContent = t === "dark" ? "☀️" : "🌙";
})();
