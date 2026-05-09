const form = document.querySelector("#generateForm");
const promptInput = document.querySelector("#promptInput");
const durationInput = document.querySelector("#durationInput");
const durationOutput = document.querySelector("#durationOutput");
const guidanceInput = document.querySelector("#guidanceInput");
const temperatureInput = document.querySelector("#temperatureInput");
const seedInput = document.querySelector("#seedInput");
const generateButton = document.querySelector("#generateButton");
const statusStrip = document.querySelector("#statusStrip");
const statusText = document.querySelector("#statusText");
const modelLabel = document.querySelector("#modelLabel");
const audioPlayer = document.querySelector("#audioPlayer");
const generationMeta = document.querySelector("#generationMeta");
const deviceValue = document.querySelector("#deviceValue");
const modeValue = document.querySelector("#modeValue");
const historyList = document.querySelector("#historyList");
const refreshButton = document.querySelector("#refreshButton");
const waveform = document.querySelector("#waveform");

function setStatus(kind, text) {
  statusStrip.classList.remove("online", "error");
  if (kind) {
    statusStrip.classList.add(kind);
  }
  statusText.textContent = text;
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function apiUrl(path) {
  const value = String(path);
  return new URL(value.startsWith("/") ? value.slice(1) : value, window.location.href).toString();
}

function mediaUrl(path) {
  const value = String(path);
  return new URL(value.startsWith("/") ? value.slice(1) : value, window.location.href).toString();
}

async function getJson(url, options) {
  const response = await fetch(apiUrl(url), options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP status text when the body is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

async function refreshHealth() {
  try {
    const health = await getJson("api/health");
    setStatus("online", "Online");
    modelLabel.textContent = health.model_loaded
      ? `${health.model_id} loaded`
      : `${health.model_id} ready`;
    deviceValue.textContent = health.device;
    modeValue.textContent = health.demo_mode ? "Demo stub" : "Fine-tuned MusicGen";
    durationInput.max = String(health.max_duration_seconds);
  } catch (error) {
    setStatus("error", `API unavailable: ${error.message}`);
  }
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!items.length) {
    historyList.innerHTML = '<div class="empty-state">No generated audio yet</div>';
    return;
  }

  for (const item of items) {
    const row = document.createElement("article");
    row.className = "history-item";

    const details = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = item.prompt;
    const meta = document.createElement("span");
    meta.textContent = `${formatDate(item.created_at)} · ${item.duration_seconds}s · ${item.device} · ${item.generation_seconds}s`;
    details.append(title, meta);

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = mediaUrl(item.audio_url);

    row.append(details, audio);
    historyList.append(row);
  }
}

async function refreshHistory() {
  const items = await getJson("api/history");
  renderHistory(items);
}

durationInput.addEventListener("input", () => {
  durationOutput.value = `${durationInput.value} s`;
});

refreshButton.addEventListener("click", () => {
  refreshHistory().catch((error) => setStatus("error", error.message));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const seed = seedInput.value === "" ? null : Number(seedInput.value);
  const payload = {
    prompt: promptInput.value.trim(),
    duration_seconds: Number(durationInput.value),
    guidance_scale: Number(guidanceInput.value),
    temperature: Number(temperatureInput.value),
    seed,
  };

  generateButton.disabled = true;
  generateButton.querySelector("span:last-child").textContent = "Generating audio";
  waveform.classList.add("is-generating");
  generationMeta.textContent = "REST request in progress";
  setStatus("online", "POST /api/generate running");

  try {
    const result = await getJson("api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    audioPlayer.src = mediaUrl(result.audio_url);
    audioPlayer.load();
    generationMeta.textContent = `${result.duration_seconds}s generated in ${result.generation_seconds}s`;
    deviceValue.textContent = result.device;
    modeValue.textContent = result.demo_mode ? "Demo stub" : "Fine-tuned MusicGen";
    setStatus("online", "Generation complete");
    await refreshHealth();
    await refreshHistory();
  } catch (error) {
    setStatus("error", error.message);
    generationMeta.textContent = "Generation failed";
  } finally {
    generateButton.disabled = false;
    generateButton.querySelector("span:last-child").textContent = "Generate";
    waveform.classList.remove("is-generating");
  }
});

await refreshHealth();
await refreshHistory();
