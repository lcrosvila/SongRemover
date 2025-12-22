import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter
import subprocess
import shutil
from pathlib import Path
import argparse

class SourceSeparator:
    def __init__(self, output_dir="temp_separation"):
        self.output_dir = Path(output_dir)
        self.model = "htdemucs"
        if shutil.which("demucs") is None:
            raise RuntimeError("Demucs not found. pip install demucs")

    def separate(self, audio_path):
        audio_path = Path(audio_path)
        track_folder = audio_path.stem
        base_path = self.output_dir / self.model / track_folder
        inst_path = base_path / "no_vocals.wav"
        voc_path = base_path / "vocals.wav"

        if inst_path.exists() and voc_path.exists():
            return str(inst_path), str(voc_path)

        # print(f"Separating stems for: {audio_path.name}...")
        cmd = [
            "demucs", "--two-stems=vocals", "-n", self.model,
            "-o", str(self.output_dir), str(audio_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return str(inst_path), str(voc_path)

class TempoMatchedRemover:
    def __init__(self, reference_path, mixed_path, sr=44100):
        self.sr = sr
        self.separator = SourceSeparator()
        
        print("--- [1/2] Loading & Separating REFERENCE ---")
        self.ref_audio = self._load(reference_path)
        r_inst, r_voc = self.separator.separate(reference_path)
        self.ref_inst = self._load(r_inst)
        self.ref_voc = self._load(r_voc)

        print("--- [2/2] Loading & Separating MIX ---")
        self.mix_audio = self._load(mixed_path)
        m_inst, m_voc = self.separator.separate(mixed_path)
        self.mix_inst = self._load(m_inst)
        self.mix_voc = self._load(m_voc)

    def _load(self, path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        return y / (np.max(np.abs(y)) + 1e-8)

    def analyze_alignment(self, mix_stem, ref_stem, hop_length):
        seg_len = min(len(mix_stem), self.sr * 45)
        ref_tempo = librosa.feature.rhythm.tempo(y=ref_stem[:seg_len], sr=self.sr)[0]
        mix_tempo = librosa.feature.rhythm.tempo(y=mix_stem[:seg_len], sr=self.sr)[0]
        
        if mix_tempo > ref_tempo * 1.8: mix_tempo /= 2
        if ref_tempo > mix_tempo * 1.8: ref_tempo /= 2
        tempo_ratio = mix_tempo / ref_tempo

        ref_aligned = ref_stem
        if abs(tempo_ratio - 1.0) > 0.005:
            ref_aligned = librosa.effects.time_stretch(ref_stem, rate=tempo_ratio)

        ref_cens = librosa.feature.chroma_cens(y=ref_aligned, sr=self.sr, hop_length=hop_length)
        mix_cens = librosa.feature.chroma_cens(y=mix_stem, sr=self.sr, hop_length=hop_length)
        
        total_corr = np.zeros(mix_cens.shape[1] + ref_cens.shape[1] - 1)
        for i in range(12):
            total_corr += signal.correlate(mix_cens[i], ref_cens[i], mode='full')
            
        peak_idx = np.argmax(total_corr)
        max_score = total_corr[peak_idx]
        confidence = max_score / (len(mix_stem) / hop_length)
        
        lag_frames = peak_idx - (ref_cens.shape[1] - 1)
        lag_seconds = lag_frames * hop_length / self.sr
        
        return tempo_ratio, lag_seconds, confidence

    def run_experiment(self, params, output_path=None):
        n_fft = params['n_fft']
        hop_length = params['hop_length']
        alpha = params['alpha']
        time_dilation = params['time_dilation']
        freq_dilation = params.get('freq_dilation', 1) # New Parameter
        floor = params['floor']
        lag_offset = params['lag_offset']

        # 1. Alignment 
        tr_voc, lag_voc, conf_voc = self.analyze_alignment(self.mix_voc, self.ref_voc, hop_length)
        tr_inst, lag_inst, conf_inst = self.analyze_alignment(self.mix_inst, self.ref_inst, hop_length)
        
        if conf_voc > conf_inst * 1.1:
            tempo_ratio, lag_seconds, confidence = tr_voc, lag_voc, conf_voc
            source = "vocal"
        else:
            tempo_ratio, lag_seconds, confidence = tr_inst, lag_inst, conf_inst
            source = "inst"
            
        # 2. Transform Ref
        proc_ref = self.ref_audio
        if abs(tempo_ratio - 1.0) > 0.005:
            proc_ref = librosa.effects.time_stretch(proc_ref, rate=tempo_ratio)
            
        total_lag = lag_seconds + lag_offset
        lag_samples = int(abs(total_lag) * self.sr)
        
        if total_lag < 0:
            if lag_samples < len(proc_ref): proc_ref = proc_ref[lag_samples:]
            else: proc_ref = np.zeros_like(self.mix_audio)
        else:
            proc_ref = np.pad(proc_ref, (lag_samples, 0))
            
        target_len = len(self.mix_audio)
        if len(proc_ref) > target_len:
            proc_ref = proc_ref[:target_len]
        else:
            proc_ref = np.pad(proc_ref, (0, target_len - len(proc_ref)))
            
        # 3. Spectral Subtraction (AGGRESSIVE MODE)
        n = min(len(self.mix_audio), len(proc_ref))
        
        S_mix = librosa.stft(self.mix_audio[:n], n_fft=n_fft, hop_length=hop_length)
        S_ref = librosa.stft(proc_ref[:n], n_fft=n_fft, hop_length=hop_length)
        
        mag_mix = np.abs(S_mix)
        mag_ref = np.abs(S_ref)
        phase_mix = np.angle(S_mix)
        
        # --- DILATION (Time AND Frequency) ---
        # size=(Frequency_Bins, Time_Frames)
        # Increasing index 0 smears Pitch. Increasing index 1 smears Time.
        if time_dilation > 1 or freq_dilation > 1:
            mag_ref = maximum_filter(mag_ref, size=(freq_dilation, time_dilation))
            
        # Energy Scaling
        ref_E = signal.medfilt(np.mean(mag_ref**2, axis=1), 5)
        mix_E = signal.medfilt(np.mean(mag_mix**2, axis=1), 5)
        scaling = np.ones((mag_mix.shape[0], 1))
        mask = ref_E > 1e-6
        scaling[mask, 0] = np.sqrt(mix_E[mask] / ref_E[mask])
        
        # Subtraction
        mag_clean = np.maximum(mag_mix - (alpha * mag_ref * scaling), floor * mag_mix)
        
        audio_clean = librosa.istft(mag_clean * np.exp(1j * phase_mix), hop_length=hop_length)
        
        if output_path:
            sf.write(output_path, audio_clean, self.sr)
            
        return {
            "confidence": confidence,
            "lag_seconds": total_lag,
            "tempo_ratio": tempo_ratio,
            "align_source": source
        }