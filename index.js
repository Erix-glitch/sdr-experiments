import { RTL2832U_Provider } from "@jtarrio/webrtlsdr/rtlsdr.js";

var elements = {};
var provider;
var device;

const SAMPLE_RATE = 1024000;
const filterCache = new Map();

async function main() {
  preparePage();
  elements.startButton.addEventListener("click", onStartButtonClick);
  elements.frequencyInput.addEventListener("change", onFrequencyInputChange);
  elements.autoGainBox.addEventListener("change", onAutoGainBoxChange);
  elements.gainInput.addEventListener("change", onGainInputChange);
}

// We ask the user to click "start" so the WebUSB API will let us connect to the device.
async function onStartButtonClick() {
  let frequency = Number(elements.frequencyInput.value);
  if (isNaN(frequency)) {
    log(`Invalid frequency: ${elements.frequencyInput.value}`);
    return;
  }
  let gain = elements.autoGainBox.checked
    ? null
    : Number(elements.gainInput.value);
  let bandwidth = Number(elements.bandwidthInput.value);
  let numSamples = Number(elements.numSamplesInput.value);
  if (isNaN(gain)) {
    log(`Invalid gain: ${elements.gainInput.value}`);
    return;
  }
  if (isNaN(bandwidth) || bandwidth <= 0) {
    log(`Invalid bandwidth: ${elements.bandwidthInput.value}`);
    return;
  }
  if (bandwidth > SAMPLE_RATE) {
    log(
      `Bandwidth must be ≤ sample rate (${SAMPLE_RATE} Hz): ${bandwidth} Hz requested`
    );
    return;
  }
  if (!Number.isInteger(numSamples) || numSamples <= 0) {
    log(
      `Number of samples must be a positive integer: ${elements.numSamplesInput.value}`
    );
    return;
  }

  elements.startButton.disabled = true;
  try {
    // We keep the provider around so the user only needs to choose the USB device once.
    if (!provider) provider = new RTL2832U_Provider();
    device = await provider.get();
    // Set the device parameters
    await device.setSampleRate(SAMPLE_RATE);
    await device.setCenterFrequency(frequency);
    await device.setGain(gain);
    // Reset the buffer and then start reading samples
    await device.resetBuffer();
    for (let i = 0; i < numSamples; i++) {
      let samples = await device.readSamples(65536);
      let dB = measurePower(samples, bandwidth);

      const threshold = parseFloat(elements.thresholdInput.value);
      const overThreshold = dB >= threshold;

      log(`${samples.frequency} Hz — ${dB} dB`, overThreshold);
    }
    // Close the device when done. You can reopen it with provider.get()
    await device.close();
  } catch (e) {
    log(`Error: ${e}`);
  } finally {
    elements.startButton.disabled = false;
  }
}

function measurePower(samples, bandwidthHz) {
  // Apply a low-pass FIR filter so we only integrate the requested bandwidth.
  let u8Samples = new Uint8Array(samples.data);
  let sampleCount = Math.floor(u8Samples.length / 2);
  if (sampleCount === 0) {
    return -Infinity;
  }
  if (bandwidthHz >= SAMPLE_RATE) {
    return measureFullBandPower(u8Samples, sampleCount);
  }
  let filter = getBandwidthFilter(bandwidthHz);
  let tapCount = filter.length;
  if (tapCount === 0) {
    return measureFullBandPower(u8Samples, sampleCount);
  }
  let bufferI = new Float64Array(tapCount);
  let bufferQ = new Float64Array(tapCount);
  let bufferIndex = 0;
  let powerSum = 0;
  for (let n = 0; n < sampleCount; n++) {
    let I = (2 * u8Samples[2 * n]) / 255 - 1;
    let Q = (2 * u8Samples[2 * n + 1]) / 255 - 1;
    bufferI[bufferIndex] = I;
    bufferQ[bufferIndex] = Q;
    let filteredI = 0;
    let filteredQ = 0;
    let idx = bufferIndex;
    for (let k = 0; k < tapCount; k++) {
      filteredI += filter[k] * bufferI[idx];
      filteredQ += filter[k] * bufferQ[idx];
      idx = idx === 0 ? tapCount - 1 : idx - 1;
    }
    bufferIndex = bufferIndex === tapCount - 1 ? 0 : bufferIndex + 1;
    powerSum += filteredI * filteredI + filteredQ * filteredQ;
  }
  let meanPower = Math.max(powerSum / sampleCount, Number.EPSILON);
  let dB = 10 * Math.log10(meanPower);
  return Math.round(dB * 100) / 100;
}

function measureFullBandPower(u8Samples, sampleCount) {
  let power = 0;
  for (let i = 0; i < sampleCount; i++) {
    let I = (2 * u8Samples[2 * i]) / 255 - 1;
    let Q = (2 * u8Samples[2 * i + 1]) / 255 - 1;
    power += I * I + Q * Q;
  }
  let meanPower = Math.max(power / sampleCount, Number.EPSILON);
  let dB = 10 * Math.log10(meanPower);
  return Math.round(dB * 100) / 100;
}

function getBandwidthFilter(bandwidthHz) {
  let bw = Number(bandwidthHz);
  if (filterCache.has(bw)) {
    return filterCache.get(bw);
  }
  let filter = createLowPassKernel(bw);
  filterCache.set(bw, filter);
  return filter;
}

function createLowPassKernel(bandwidthHz) {
  // Windowed-sinc low-pass, normalized for unity gain at DC.
  if (!isFinite(bandwidthHz) || bandwidthHz <= 0) {
    return new Float64Array();
  }
  let cutoff = Math.min(0.5, bandwidthHz / (2 * SAMPLE_RATE));
  if (cutoff <= 0) {
    return new Float64Array([1]);
  }
  let tapCount = 31;
  let taps = new Float64Array(tapCount);
  let center = (tapCount - 1) / 2;
  let sum = 0;
  for (let n = 0; n < tapCount; n++) {
    let k = n - center;
    let coeff;
    if (k === 0) {
      coeff = 2 * cutoff;
    } else {
      coeff = Math.sin(2 * Math.PI * cutoff * k) / (Math.PI * k);
    }
    let window = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (tapCount - 1));
    let value = coeff * window;
    taps[n] = value;
    sum += value;
  }
  if (sum === 0) {
    return new Float64Array([1]);
  }
  for (let n = 0; n < tapCount; n++) {
    taps[n] /= sum;
  }
  return taps;
}

function log(msg, highlight = false) {
  const container = elements.logArea;
  if (!container) {
    return;
  }
  const entry = document.createElement("div");
  entry.className =
    "rounded border border-slate-200 bg-white/80 p-2 text-slate-800 shadow-sm dark:border-[#333333] dark:bg-[#232323] dark:text-[#e0e0e0]";
  if (highlight) {
    entry.classList.add(
      "bg-green-100",
      "border-green-300",
      "dark:bg-green-900/30",
      "dark:border-green-600"
    );
  }
  entry.textContent = `${new Date().toISOString()} — ${msg}`;
  container.prepend(entry);
  const maxEntries = 50;
  while (container.childElementCount > maxEntries) {
    container.removeChild(container.lastElementChild);
  }
}

function onFrequencyInputChange() {
  setNumberInput(
    elements.frequencyInput,
    () => radio.getFrequency(),
    (v) => radio.setFrequency(v)
  );
}

function onAutoGainBoxChange() {
  let checked = elements.autoGainBox.checked;
  elements.gainInput.disabled = checked;
  if (checked) {
    radio.setGain(null);
  } else {
    onGainChange();
  }
}

function onGainInputChange() {
  setNumberInput(
    elements.gainInput,
    () => radio.getGain(),
    (v) => radio.setGain(v)
  );
}

function setNumberInput(element, getter, setter) {
  let v = Number(element.value);
  if (isNaN(v)) {
    v = getter();
  } else {
    setter(v);
  }
  element.value = String(v);
}

function preparePage() {
  for (let id of [
    "startButton",
    "stopButton",
    "frequencyInput",
    "autoGainBox",
    "gainInput",
    "bandwidthInput",
    "numSamplesInput",
    "logArea",
    "thresholdInput"
  ]) {
    elements[id] = document.getElementById(id);
  }
  if (elements.bandwidthInput && !elements.bandwidthInput.value) {
    elements.bandwidthInput.value = "200000";
  }
}

window.addEventListener("load", main);
