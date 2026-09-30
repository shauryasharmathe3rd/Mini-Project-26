"""
data_loader.py
=============================================================================
Memory-Efficient Waveform Data Loaders for CSV, WFDB, and MAT Formats.

Key Engineering Features:
- Chunked iteration for CSV to avoid loading gigabytes into RAM.
- WFDB partial reads using `sampto` parameter.
- MAT selective variable extraction with immediate np.float32 downcasting.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
import wfdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (DataLoader) %(message)s"
)


class WaveformDataLoader:
    def __init__(self, target_fs: int = 125):
        self.target_fs = target_fs

    def load_csv_chunked(self, file_path: str, chunksize: int = 10000, max_samples: int = 2500):
        """
        Memory-efficient CSV reader reading in chunks and downcasting to float32.
        """
        logging.info(f"Initiating chunked CSV load: {os.path.basename(file_path)} (chunksize={chunksize})")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        collected_ecg, collected_ppg = [], []
        total_loaded = 0

        # Read specific signal columns lazily
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            cols = {c.lower(): c for c in chunk.columns}
            ecg_col = cols.get('ecg') or cols.get('lead_ii') or chunk.columns[0]
            ppg_col = cols.get('ppg') or cols.get('pleth') or (chunk.columns[1] if len(chunk.columns) > 1 else chunk.columns[0])

            ecg_data = chunk[ecg_col].values.astype(np.float32)
            ppg_data = chunk[ppg_col].values.astype(np.float32)

            collected_ecg.append(ecg_data)
            collected_ppg.append(ppg_data)
            total_loaded += len(chunk)

            logging.info(f"Streamed {total_loaded} samples from CSV chunk...")
            if total_loaded >= max_samples:
                break

        ecg = np.concatenate(collected_ecg)[:max_samples]
        ppg = np.concatenate(collected_ppg)[:max_samples]
        logging.info(f"CSV loading complete. Yielded {len(ecg)} samples downcasted to float32.")
        return ecg, ppg

    def load_wfdb_partial(self, record_path: str, sampto: int = 2500):
        """
        WFDB partial read: only loads `sampto` samples into memory from header & binary.
        """
        base_record = os.path.splitext(record_path)[0]
        logging.info(f"Initiating partial WFDB read: {os.path.basename(base_record)} (sampto={sampto})")

        record = wfdb.rdrecord(base_record, sampto=sampto)
        sig_names = [s.lower() for s in record.sig_name]

        ecg_idx = next((i for i, s in enumerate(sig_names) if 'ecg' in s or 'ii' in s), 0)
        ppg_idx = next((i for i, s in enumerate(sig_names) if 'ppg' in s or 'pleth' in s), 1 if record.n_sig > 1 else 0)

        ecg = record.p_signal[:, ecg_idx].astype(np.float32)
        ppg = record.p_signal[:, ppg_idx].astype(np.float32)

        logging.info(f"WFDB record '{record.record_name}' loaded successfully. Sampling rate: {record.fs} Hz.")
        return ecg, ppg, record.fs

    def load_mat_selective(self, mat_file_path: str, max_samples: int = 2500):
        """
        MAT file loader: selectively loads specific keys and enforces float32.
        """
        logging.info(f"Initiating selective MAT read: {os.path.basename(mat_file_path)}")
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(f"MAT file not found: {mat_file_path}")

        mat_dict = sio.loadmat(mat_file_path, variable_names=['data', 'param', 'val'])

        if 'data' in mat_dict:
            raw_data = mat_dict['data']
        elif 'val' in mat_dict:
            raw_data = mat_dict['val']
        else:
            keys = [k for k in mat_dict.keys() if not k.startswith('__')]
            raw_data = mat_dict[keys[0]]

        raw_data = np.asarray(raw_data, dtype=np.float32)
        if raw_data.ndim == 1:
            ecg = raw_data[:max_samples]
            ppg = raw_data[:max_samples]
        elif raw_data.shape[0] > raw_data.shape[1]:
            ecg = raw_data[:max_samples, 0]
            ppg = raw_data[:max_samples, 1 if raw_data.shape[1] > 1 else 0]
        else:
            ecg = raw_data[0, :max_samples]
            ppg = raw_data[1 if raw_data.shape[0] > 1 else 0, :max_samples]

        logging.info(f"MAT structure converted to float32 arrays (Shape: {ecg.shape}).")
        return ecg, ppg
