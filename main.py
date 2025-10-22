import asyncio
import threading

import numpy as np
import websockets
from rtlsdr import RtlSdr
from scipy.signal import butter, decimate, find_peaks, sosfiltfilt

# --- SDR CONFIGURATION ---
SAMPLE_RATE = 2.4e6
CENTER_FREQ = 13.56e6
SDR_GAIN = 35

# --- DETECTION PARAMETERS ---
SNR_THRESHOLD_DB = 32.0  # minimum dB above median noise floor to accept a peak
DECIM_FACTOR = 50
MIN_BURST_GAP = 0.001
BANDWIDTH_HZ = 1_000.0

clients = set()
latest_peak_strength = None
latest_peak_lock = threading.Lock()
stop_event = threading.Event()


async def handler(websocket, path=None):  # pylint: disable=unused-argument
    """Handle a new WebSocket client connection."""
    clients.add(websocket)
    addr = getattr(websocket, "remote_address", None)
    print(f"Client connected: {addr}")

    with latest_peak_lock:
        latest = latest_peak_strength

    if latest is not None:
        await websocket.send(f"{latest:.3f}")

    try:
        async for _ in websocket:
            pass
    except websockets.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        print(f"Client disconnected: {addr}")


async def broadcast(message):
    """Send a strength update to all connected clients."""
    if not clients:
        return

    stale = []
    for ws in list(clients):
        try:
            await ws.send(message)
        except websockets.ConnectionClosed:
            stale.append(ws)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"WebSocket send error: {exc}")
            stale.append(ws)

    for ws in stale:
        clients.discard(ws)


async def broadcast_loop(message_queue):
    """Consume strength updates from the queue and broadcast to clients."""
    while True:
        message = await message_queue.get()
        try:
            if message is None:
                break
            await broadcast(message)
        finally:
            message_queue.task_done()


def sdr_worker(message_queue, loop):
    """Continuously read from the SDR, detect peaks, and enqueue strength updates."""
    global latest_peak_strength  # pylint: disable=global-statement

    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = SDR_GAIN

    counter = 0

    print("Listening... (press Ctrl+C to stop)")

    try:
        while not stop_event.is_set():
            samples = sdr.read_samples(256 * 1024)  # ~100 ms at 2.4 MS/s

            # keep only +/-1 kHz around center (DC) before envelope
            nyq = sdr.sample_rate / 2.0
            sos = butter(6, BANDWIDTH_HZ / nyq, btype="low", output="sos")
            samples_nb = sosfiltfilt(sos, samples)  # complex IQ, zero-phase narrowband

            # decimate the narrowband complex IQ, then take envelope
            x_dec = decimate(samples_nb, DECIM_FACTOR, ftype="fir", zero_phase=True)
            fs_dec = sdr.sample_rate / DECIM_FACTOR

            env_dec = np.abs(x_dec)
            env_db = 20 * np.log10(env_dec + 1e-12)
            noise_floor_db = np.median(env_db)

            # detect peaks that rise above the noise floor by the requested SNR
            min_height_db = noise_floor_db + SNR_THRESHOLD_DB
            peaks, _ = find_peaks(
                env_db,
                height=min_height_db,
                distance=int(MIN_BURST_GAP * fs_dec),
            )

            for peak_idx in peaks:
                strength = env_dec[peak_idx]
                strength_db = env_db[peak_idx]
                snr_db = strength_db - noise_floor_db
                counter += 1
                with latest_peak_lock:
                    latest_peak_strength = strength
                message = f"{strength:.3f}"
                print(
                    f"Peak {counter}: strength {strength:.3f} ({strength_db:.1f} dB, {snr_db:.1f} dB above noise)"
                )
                loop.call_soon_threadsafe(message_queue.put_nowait, message)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"SDR worker error: {exc}")
    finally:
        sdr.close()
        stop_event.set()
        print("SDR closed.")


async def run():
    loop = asyncio.get_running_loop()
    message_queue = asyncio.Queue()

    worker = threading.Thread(target=sdr_worker, args=(message_queue, loop), daemon=True)
    worker.start()

    broadcaster = asyncio.create_task(broadcast_loop(message_queue))

    try:
        async with websockets.serve(handler, "0.0.0.0", 5001):
            print("WebSocket server listening on ws://0.0.0.0:5001")
            await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        await asyncio.to_thread(worker.join)
        await message_queue.put(None)
        await broadcaster


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()


if __name__ == "__main__":
    main()
