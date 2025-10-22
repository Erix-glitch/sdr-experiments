from rtlsdr import RtlSdr
import numpy as np
from scipy.signal import find_peaks, decimate
# --- SDR CONFIGURATION ---
sdr = RtlSdr()
sdr.sample_rate = 2.4e6
sdr.center_freq = 13.56e6
sdr.gain = 35

# --- DETECTION PARAMETERS ---
threshold = 1
decim_factor = 50
min_burst_gap = 0.01

counter = 0

print("Listening... (press Ctrl+C to stop)")

try:
    while True:
        # read samples
        samples = sdr.read_samples(256*1024)  # ~100 ms at 2.4 MS/s

        # compute envelope
        env = np.abs(samples)

        # decimate
        env_dec = decimate(env, decim_factor)
        fs_dec = sdr.sample_rate / decim_factor

        # detect peaks
        env_dec /= np.max(env_dec) + 1e-9
        peaks, _ = find_peaks(env_dec, height=threshold, distance=int(min_burst_gap * fs_dec))

        for _ in range(len(peaks)):
            counter += 1
            print(counter)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    sdr.close()
