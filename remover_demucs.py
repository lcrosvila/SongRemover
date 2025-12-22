import os
# --- FIX FOR MKL ERROR ---
os.environ["MKL_THREADING_LAYER"] = "GNU"
# -------------------------

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import argparse
import subprocess
import shutil
from pathlib import Path
from scipy.ndimage import maximum_filter

class SourceSeparator:
    """Wrapper for Demucs SOTA Source Separation"""
    
    def __init__(self, output_dir="temp_separation"):
        self.output_dir = Path(output_dir)
        self.model = "htdemucs"  
        
        if shutil.which("demucs") is None:
            raise RuntimeError("Demucs is not installed. Please install it via: pip install demucs")

    def separate(self, audio_path):
        audio_path = Path(audio_path)
        print(f"\n[Demucs] Separating stems for: {audio_path.name}...")
        
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", self.model,
            "-o", str(self.output_dir),
            str(audio_path)
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"Demucs Error: {process.stderr}")
            raise RuntimeError("Demucs separation failed.")

        track_folder = audio_path.stem
        base_path = self.output_dir / self.model / track_folder
        
        inst_path = base_path / "no_vocals.wav"
        voc_path = base_path / "vocals.wav"
        
        return str(inst_path), str(voc_path)

    def cleanup(self):
        print(f"\n[Debug] Temp files preserved at: {self.output_dir.absolute()}")

class TempoMatchedRemover:
    
    def __init__(self, reference_path, sr=44100):
        self.sr = sr
        self.separator = SourceSeparator()
        
        print(f"Loading reference: {reference_path}")
        self.ref_audio, _ = librosa.load(reference_path, sr=sr, mono=True)
        self.ref_audio = self._normalize(self.ref_audio)
        
        # Separate Reference
        self.ref_inst_path, self.ref_voc_path = self.separator.separate(reference_path)
        
        self.ref_inst, _ = librosa.load(self.ref_inst_path, sr=sr, mono=True)
        self.ref_voc, _ = librosa.load(self.ref_voc_path, sr=sr, mono=True)
        self.ref_inst = self._normalize(self.ref_inst)
        self.ref_voc = self._normalize(self.ref_voc)

    def _normalize(self, audio):
        return audio / (np.max(np.abs(audio)) + 1e-8)

    def estimate_pitch_robust(self, mix_seg, ref_seg):
        """Robust pitch detection using Chroma Cross-Correlation"""
        # Compute Chroma CQT (more accurate for pitch than CENS)
        # We use a longer segment for pitch to average out noise
        n_bins = min(len(mix_seg), len(ref_seg), self.sr * 30)
        
        chroma_ref = librosa.feature.chroma_cqt(y=ref_seg[:n_bins], sr=self.sr)
        chroma_mix = librosa.feature.chroma_cqt(y=mix_seg[:n_bins], sr=self.sr)
        
        # Sum over time to get the "Key Profile" (12x1 vector)
        ref_profile = np.sum(chroma_ref, axis=1)
        mix_profile = np.sum(chroma_mix, axis=1)
        
        # Normalize
        ref_profile /= (np.linalg.norm(ref_profile) + 1e-8)
        mix_profile /= (np.linalg.norm(mix_profile) + 1e-8)
        
        # Cyclical Cross-Correlation (slide ref against mix)
        correlations = []
        for shift in range(12):
            # Roll reference by 'shift' semitones
            rolled_ref = np.roll(ref_profile, shift)
            corr = np.dot(rolled_ref, mix_profile)
            correlations.append(corr)
            
        best_shift = np.argmax(correlations)
        max_corr = correlations[best_shift]
        
        # Handle wrap-around (e.g., 11 semitones is likely -1)
        if best_shift > 6:
            best_shift -= 12
            
        print(f"  Pitch Analysis: Detected {best_shift:+d} semitones (Confidence: {max_corr:.3f})")
        
        # If confidence is low, assume 0 shift (safer)
        if max_corr < 0.6:
            print("  -> Confidence too low. Assuming 0 shift.")
            return 0
            
        return best_shift
        
    def analyze_stems(self, mix_stem, ref_stem, stem_name="Instrumental", disable_pitch=False):
        print(f"\n--- Analyzing {stem_name} Stems ---")
        
        # 1. Estimate Tempo Ratio
        seg_len = min(len(mix_stem), self.sr * 45)
        ref_tempo = librosa.beat.tempo(y=ref_stem[:seg_len], sr=self.sr)[0]
        mix_tempo = librosa.beat.tempo(y=mix_stem[:seg_len], sr=self.sr)[0]
        
        if mix_tempo > ref_tempo * 1.8: mix_tempo /= 2
        if ref_tempo > mix_tempo * 1.8: ref_tempo /= 2
        
        tempo_ratio = mix_tempo / ref_tempo
        print(f"  Tempo Ratio: {tempo_ratio:.3f}")
        
        # 2. Stretch Reference
        if abs(tempo_ratio - 1.0) > 0.005:
            ref_stretched = librosa.effects.time_stretch(ref_stem, rate=tempo_ratio)
        else:
            ref_stretched = ref_stem
            
        # 3. Estimate Pitch (New Robust Method)
        if disable_pitch:
            pitch_shift = 0
            print("  Pitch detection disabled by user.")
        else:
            pitch_shift = self.estimate_pitch_robust(mix_stem, ref_stretched)
        
        # Apply pitch shift to ref for alignment check
        if pitch_shift != 0:
            ref_aligned = librosa.effects.pitch_shift(ref_stretched, sr=self.sr, n_steps=pitch_shift)
        else:
            ref_aligned = ref_stretched

        # 4. Bi-Directional Alignment (Full Cross-Correlation)
        print("  Calculating Alignment...")
        hop_length = 256
        
        ref_cens = librosa.feature.chroma_cens(y=ref_aligned, sr=self.sr, hop_length=hop_length)
        mix_cens = librosa.feature.chroma_cens(y=mix_stem, sr=self.sr, hop_length=hop_length)
        
        # Correlation
        total_corr = np.zeros(mix_cens.shape[1] + ref_cens.shape[1] - 1)
        for i in range(12):
            total_corr += signal.correlate(mix_cens[i], ref_cens[i], mode='full')
            
        peak_idx = np.argmax(total_corr)
        max_score = total_corr[peak_idx]
        confidence = max_score / (len(mix_stem) / hop_length) 
        
        lag_frames = peak_idx - (ref_cens.shape[1] - 1)
        lag_seconds = lag_frames * hop_length / self.sr
        
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Offset: {lag_seconds:.2f}s")
        
        return {
            "tempo_ratio": tempo_ratio,
            "pitch_shift": pitch_shift,
            "lag_seconds": lag_seconds,
            "confidence": confidence
        }

    def spectral_subtraction(self, mixed, reference, alpha=2.0):
        n = min(len(mixed), len(reference))
        mixed = mixed[:n]
        reference = reference[:n]
        
        # n_fft, hop = 2048, 256
        # let's make it more precise
        n_fft, hop = 4096, 256
        S_mix = librosa.stft(mixed, n_fft=n_fft, hop_length=hop)
        S_ref = librosa.stft(reference, n_fft=n_fft, hop_length=hop)
        
        mag_mix = np.abs(S_mix)
        mag_ref = np.abs(S_ref)
        phase_mix = np.angle(S_mix)
        
        # Smoothed energy scaling
        ref_E = signal.medfilt(np.mean(mag_ref**2, axis=1), 5)
        mix_E = signal.medfilt(np.mean(mag_mix**2, axis=1), 5)
        
        scaling = np.ones((mag_mix.shape[0], 1))
        mask = ref_E > 1e-6
        scaling[mask, 0] = np.sqrt(mix_E[mask] / ref_E[mask])
        
        mag_clean = np.maximum(mag_mix - (alpha * mag_ref * scaling), 0.05 * mag_mix)
        
        return librosa.istft(mag_clean * np.exp(1j * phase_mix), hop_length=hop)

    def process(self, mixed_path, output_path, alpha=2.0, disable_pitch=False):
        print("\n" + "="*60)
        print("DUAL-PATH SOTA AUDIO REMOVAL (FIXED PITCH)")
        print("="*60)
        
        print(f"Loading mixed: {mixed_path}")
        mixed_audio, _ = librosa.load(mixed_path, sr=self.sr, mono=True)
        mixed_audio = self._normalize(mixed_audio)
        
        mix_inst_path, mix_voc_path = self.separator.separate(mixed_path)
        mix_inst, _ = librosa.load(mix_inst_path, sr=self.sr, mono=True)
        mix_voc, _ = librosa.load(mix_voc_path, sr=self.sr, mono=True)
        mix_inst = self._normalize(mix_inst)
        mix_voc = self._normalize(mix_voc)
        
        # Analyze
        res_inst = self.analyze_stems(mix_inst, self.ref_inst, "Instrumental", disable_pitch)
        res_voc = self.analyze_stems(mix_voc, self.ref_voc, "Vocals", disable_pitch)
        
        # Pick Winner
        if res_voc["confidence"] > res_inst["confidence"] * 1.2:
            print("\n>> Using VOCAL alignment")
            params = res_voc
        else:
            print("\n>> Using INSTRUMENTAL alignment")
            params = res_inst
            
        # Transform Original Reference
        print("\nApplying alignment to full reference...")
        proc_ref = self.ref_audio
        
        if abs(params["tempo_ratio"] - 1.0) > 0.005:
            print(f"Stretching by {params['tempo_ratio']:.3f}x")
            proc_ref = librosa.effects.time_stretch(proc_ref, rate=params['tempo_ratio'])
            
        if params["pitch_shift"] != 0:
            print(f"Shifting pitch by {params['pitch_shift']} semitones")
            proc_ref = librosa.effects.pitch_shift(proc_ref, sr=self.sr, n_steps=params['pitch_shift'])
            
        # Alignment
        lag_s = params["lag_seconds"]
        # lag_s = params["lag_seconds"] - 5.0 # let's see if this will work better
        lag_samples = int(abs(lag_s) * self.sr)
        
        if lag_s < 0:
            print(f"Cutting first {lag_s:.2f}s of reference...")
            if lag_samples < len(proc_ref):
                proc_ref = proc_ref[lag_samples:]
            else:
                print("Error: Offset larger than file length!")
        else:
            print(f"Padding reference start with {lag_s:.2f}s...")
            proc_ref = np.pad(proc_ref, (lag_samples, 0))
            
        # Trim/Pad End
        if len(proc_ref) > len(mixed_audio):
            proc_ref = proc_ref[:len(mixed_audio)]
        else:
            proc_ref = np.pad(proc_ref, (0, len(mixed_audio) - len(proc_ref)))
            
        # Subtraction
        cleaned = self.spectral_subtraction(mixed_audio, proc_ref, alpha=alpha)
        
        sf.write(output_path, self._normalize(cleaned) * 0.95, self.sr)
        print(f"\n✓ Saved cleaned audio: {output_path}")
        
        # Debug
        debug_path = output_path.replace(".mp3", "_aligned_ref_debug.wav")
        sf.write(debug_path, proc_ref, self.sr)
        print(f"✓ Saved debug reference: {debug_path}")
        
        self.separator.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', '-r', required=True)
    parser.add_argument('--mixed', '-m', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--alpha', '-a', type=float, default=2.0)
    parser.add_argument('--no-pitch', action='store_true', help="Disable automatic pitch shifting")
    args = parser.parse_args()
    
    remover = TempoMatchedRemover(args.reference)
    remover.process(args.mixed, args.output, args.alpha, args.no_pitch)