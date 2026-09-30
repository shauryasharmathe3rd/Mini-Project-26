"""
preprocess.py
=============================================================================
Digital Signal Processing (DSP), Feature Extraction, and Synthetic
Pathology Generation for Sepsis Hemodynamic Signatures.
=============================================================================
"""

import os
import logging
import numpy as np
from scipy.signal import butter, filtfilt, resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Preprocess) %(message)s"
)


class PhysiologicalPreprocessor:
    def __init__(self, target_fs: int = 125):
        self.target_fs = target_fs

    def butter_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
        """
        Zero-phase forward-backward 4th-order Butterworth bandpass filter.
        """
        nyq = 0.5 * fs
        low = max(0.001, lowcut / nyq)
        high = min(0.999, highcut / nyq)
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, data).astype(np.float32)
        return filtered

    def process_ecg(self, ecg_raw: np.ndarray, fs: int) -> np.ndarray:
        """Removes baseline wander (0.5 Hz) and high-frequency noise (40 Hz)."""
        logging.info(f"Filtering ECG: 0.5 - 40.0 Hz bandpass at fs={fs} Hz.")
        filtered = self.butter_bandpass_filter(ecg_raw, 0.5, 40.0, fs)
        if fs != self.target_fs:
            logging.info(f"Resampling ECG from {fs} Hz to {self.target_fs} Hz...")
            num_target = int(len(filtered) * self.target_fs / fs)
            filtered = resample(filtered, num_target).astype(np.float32)
        std = np.std(filtered) + 1e-6
        return ((filtered - np.mean(filtered)) / std).astype(np.float32)

    def process_ppg(self, ppg_raw: np.ndarray, fs: int) -> np.ndarray:
        """Removes respiratory baseline shift (0.5 Hz) and harmonics (>8.0 Hz)."""
        logging.info(f"Filtering PPG: 0.5 - 8.0 Hz bandpass at fs={fs} Hz.")
        filtered = self.butter_bandpass_filter(ppg_raw, 0.5, 8.0, fs)
        if fs != self.target_fs:
            logging.info(f"Resampling PPG from {fs} Hz to {self.target_fs} Hz...")
            num_target = int(len(filtered) * self.target_fs / fs)
            filtered = resample(filtered, num_target).astype(np.float32)
        std = np.std(filtered) + 1e-6
        return ((filtered - np.mean(filtered)) / std).astype(np.float32)

    def simulate_sepsis_morphology(self, ecg_clean: np.ndarray, ppg_clean: np.ndarray, fs: int = 125):
        """
        Simulates physiological hallmarks of septic shock:
        1. Tachycardia & shortened R-R interval (approx. 1.55x HR increase).
        2. Peripheral vasoconstriction & hypoperfusion (damped PPG amplitude by ~58%).
        3. Loss of PPG dicrotic notch clarity & added high-frequency physiological noise.
        """
        logging.info("Synthesizing septic hemodynamic waveform pathology...")

        warp_factor = 1.55  # Increases HR from e.g. 72 to ~112 bpm
        target_len = len(ecg_clean)
        resampled_len = int(target_len * warp_factor)

        ecg_tachy = resample(ecg_clean, resampled_len)
        ppg_tachy = resample(ppg_clean, resampled_len)

        if len(ecg_tachy) >= target_len:
            ecg_sepsis = ecg_tachy[:target_len]
            ppg_sepsis = ppg_tachy[:target_len]
        else:
            ecg_sepsis = np.tile(ecg_tachy, int(np.ceil(target_len / len(ecg_tachy))))[:target_len]
            ppg_sepsis = np.tile(ppg_tachy, int(np.ceil(target_len / len(ppg_tachy))))[:target_len]

        ppg_sepsis = ppg_sepsis * 0.42

        noise_ecg = np.random.normal(0, 0.08, size=target_len).astype(np.float32)
        noise_ppg = np.random.normal(0, 0.04, size=target_len).astype(np.float32)

        ecg_sepsis = (ecg_sepsis + noise_ecg).astype(np.float32)
        ppg_sepsis = (ppg_sepsis + noise_ppg).astype(np.float32)

        logging.info("Sepsis waveform synthesis complete: Tachycardia + Hypoperfusion added.")
        return ecg_sepsis, ppg_sepsis

    def create_memmap_dataset(self, filename: str, num_windows: int, window_len: int = 1250, channels: int = 2):
        """
        Creates a disk-backed numpy.memmap for out-of-core window storage.
        """
        logging.info(f"Allocating memory map at {filename} (Shape: [{num_windows}, {channels}, {window_len}]).")
        shape = (num_windows, channels, window_len)
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=shape)
        return fp
