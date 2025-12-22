import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import argparse

class TempoMatchedRemover:
    """Remove audio with automatic tempo/pitch matching"""
    
    def __init__(self, reference_path, sr=22050):
        print(f"Loading reference: {reference_path}")
        self.ref_audio, self.sr = librosa.load(reference_path, sr=sr, mono=True)
        self.ref_audio = self.ref_audio / (np.max(np.abs(self.ref_audio)) + 1e-8)
        print(f"Reference: {len(self.ref_audio)/self.sr:.2f}s @ {self.sr}Hz")
        
    def estimate_tempo_ratio(self, mixed_audio):
        """Estimate tempo ratio between reference and mixed"""
        print("\nEstimating tempo...")
        
        # Use a segment for tempo estimation
        segment_length = min(len(mixed_audio), self.sr * 10)
        
        ref_tempo = librosa.beat.tempo(
            y=self.ref_audio[:segment_length], 
            sr=self.sr
        )[0]
        
        mixed_tempo = librosa.beat.tempo(
            y=mixed_audio[:segment_length], 
            sr=self.sr
        )[0]
        
        tempo_ratio = mixed_tempo / ref_tempo
        
        print(f"Reference tempo: {ref_tempo:.1f} BPM")
        print(f"Mixed tempo: {mixed_tempo:.1f} BPM")
        print(f"Tempo ratio: {tempo_ratio:.3f} ({(tempo_ratio-1)*100:+.1f}%)")
        
        return tempo_ratio
    
    def time_stretch_reference(self, rate):
        """Time-stretch reference to match mixed audio tempo"""
        print(f"\nTime-stretching reference by {rate:.3f}x...")
        stretched = librosa.effects.time_stretch(self.ref_audio, rate=rate)
        return stretched
    
    def estimate_pitch_shift(self, mixed_audio):
        """Estimate pitch shift in semitones"""
        # Use chromagrams
        ref_chroma = librosa.feature.chroma_cqt(
            y=self.ref_audio[:min(len(self.ref_audio), len(mixed_audio))], 
            sr=self.sr
        )
        mixed_chroma = librosa.feature.chroma_cqt(y=mixed_audio, sr=self.sr)
        
        # Find dominant pitch
        ref_pitch = np.argmax(np.sum(ref_chroma, axis=1))
        mixed_pitch = np.argmax(np.sum(mixed_chroma, axis=1))
        
        shift = (mixed_pitch - ref_pitch) % 12
        if shift > 6:
            shift -= 12
        
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        print(f"\nPitch analysis:")
        print(f"Reference: {pitch_names[ref_pitch]}")
        print(f"Mixed: {pitch_names[mixed_pitch]}")
        
        if shift != 0:
            print(f"Pitch shift detected: {shift:+d} semitones")
        
        return shift
    
    def pitch_shift_reference(self, semitones):
        """Pitch shift reference audio"""
        if semitones == 0:
            return self.ref_audio
        
        print(f"Pitch shifting reference by {semitones:+d} semitones...")
        shifted = librosa.effects.pitch_shift(
            self.ref_audio, 
            sr=self.sr, 
            n_steps=semitones
        )
        return shifted
    
    def find_alignment_dtw(self, reference, mixed, max_search=60):
        """Find alignment using Dynamic Time Warping"""
        print("\nFinding alignment with DTW...")
        
        # Use chroma features for DTW (more robust)
        ref_chroma = librosa.feature.chroma_cqt(y=reference, sr=self.sr, hop_length=2048)
        mixed_chroma = librosa.feature.chroma_cqt(y=mixed, sr=self.sr, hop_length=2048)
        
        # Search for best starting position
        best_score = -np.inf
        best_position = 0
        
        hop_length = 2048
        search_frames = int(max_search * self.sr / hop_length)
        
        for start_frame in range(0, min(search_frames, ref_chroma.shape[1] - mixed_chroma.shape[1]), 5):
            ref_segment = ref_chroma[:, start_frame:start_frame + mixed_chroma.shape[1]]
            
            if ref_segment.shape[1] < mixed_chroma.shape[1]:
                continue
            
            # Simple correlation
            score = np.sum(ref_segment * mixed_chroma[:, :ref_segment.shape[1]])
            
            if score > best_score:
                best_score = score
                best_position = start_frame * hop_length / self.sr
        
        # Normalize score
        norm_score = best_score / (np.linalg.norm(ref_chroma) * np.linalg.norm(mixed_chroma) + 1e-8)
        
        print(f"Best alignment: {best_position:.2f}s (score: {norm_score:.4f})")
        
        return int(best_position * self.sr), norm_score
    
    def spectral_subtraction(self, mixed, reference, alpha=2.0):
        """Spectral subtraction with adapted reference"""
        print("\nApplying spectral subtraction...")
        
        # Ensure same length
        min_len = min(len(mixed), len(reference))
        mixed = mixed[:min_len]
        reference = reference[:min_len]
        
        # STFT
        D_mixed = librosa.stft(mixed, n_fft=2048, hop_length=512)
        D_ref = librosa.stft(reference, n_fft=2048, hop_length=512)
        
        # Magnitude and phase
        mag_mixed = np.abs(D_mixed)
        phase_mixed = np.angle(D_mixed)
        mag_ref = np.abs(D_ref)
        
        # Estimate scaling factor per frequency band
        scaling = np.zeros(mag_ref.shape[0])
        for freq in range(mag_ref.shape[0]):
            ref_energy = np.mean(mag_ref[freq, :]**2)
            mixed_energy = np.mean(mag_mixed[freq, :]**2)
            if ref_energy > 1e-10:
                scaling[freq] = np.sqrt(mixed_energy / ref_energy)
        
        # Smooth scaling
        scaling = signal.medfilt(scaling, kernel_size=5)
        
        # Apply scaled subtraction
        mag_cleaned = np.maximum(
            mag_mixed - alpha * scaling[:, np.newaxis] * mag_ref,
            0.1 * mag_mixed
        )
        
        # Reconstruct
        D_cleaned = mag_cleaned * np.exp(1j * phase_mixed)
        cleaned = librosa.istft(D_cleaned, hop_length=512)
        
        return cleaned
    
    def process_with_adaptation(self, mixed_path, output_path, 
                                manual_tempo_ratio=None, 
                                manual_pitch_shift=None,
                                alpha=2.0):
        """Process with automatic tempo/pitch adaptation"""
        
        print("\n" + "="*60)
        print("TEMPO-ADAPTIVE REMOVAL")
        print("="*60)
        
        # Load mixed audio
        print(f"\nLoading mixed audio: {mixed_path}")
        mixed_audio, _ = librosa.load(mixed_path, sr=self.sr, mono=True)
        mixed_audio = mixed_audio / (np.max(np.abs(mixed_audio)) + 1e-8)
        print(f"Mixed: {len(mixed_audio)/self.sr:.2f}s")
        
        # Estimate or use manual tempo ratio
        if manual_tempo_ratio is None:
            tempo_ratio = self.estimate_tempo_ratio(mixed_audio)
        else:
            tempo_ratio = manual_tempo_ratio
            print(f"\nUsing manual tempo ratio: {tempo_ratio:.3f}")
        
        # Time-stretch reference
        if abs(tempo_ratio - 1.0) > 0.02:  # More than 2% difference
            adapted_ref = self.time_stretch_reference(tempo_ratio)
        else:
            print("\nTempo difference negligible, skipping time-stretch")
            adapted_ref = self.ref_audio
        
        # Estimate or use manual pitch shift
        if manual_pitch_shift is None:
            pitch_shift = self.estimate_pitch_shift(mixed_audio)
        else:
            pitch_shift = manual_pitch_shift
            print(f"\nUsing manual pitch shift: {pitch_shift:+d} semitones")
        
        # Pitch-shift reference
        if pitch_shift != 0:
            adapted_ref = self.pitch_shift_reference(pitch_shift)
        
        print(f"\nAdapted reference length: {len(adapted_ref)/self.sr:.2f}s")
        
        # Find alignment
        align_pos, align_score = self.find_alignment_dtw(adapted_ref, mixed_audio)
        
        if align_score < 0.1:
            print("\n⚠️  WARNING: Poor alignment score!")
            print("   The reference may still not match the mixed audio.")
            print("   Results may be suboptimal.")
        
        # Extract aligned segment
        ref_start = align_pos
        ref_end = min(ref_start + len(mixed_audio), len(adapted_ref))
        aligned_ref = adapted_ref[ref_start:ref_end]
        
        # Pad if needed
        if len(aligned_ref) < len(mixed_audio):
            aligned_ref = np.pad(aligned_ref, (0, len(mixed_audio) - len(aligned_ref)))
        else:
            aligned_ref = aligned_ref[:len(mixed_audio)]
        
        # Apply spectral subtraction
        cleaned = self.spectral_subtraction(mixed_audio, aligned_ref, alpha=alpha)
        
        # Normalize
        cleaned = cleaned / (np.max(np.abs(cleaned)) + 1e-8) * 0.95
        
        # Save
        sf.write(output_path, cleaned, self.sr)
        print(f"\n✓ Saved cleaned audio: {output_path}")
        
        # Save adapted reference for verification
        ref_output = output_path.replace('.mp3', '_adapted_reference.wav').replace('.wav', '_adapted_reference.wav')
        sf.write(ref_output, aligned_ref, self.sr)
        print(f"✓ Saved adapted reference: {ref_output}")
        
        return cleaned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Tempo-Adaptive Audio Removal System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This tool automatically adapts the reference audio to match tempo/pitch changes
in the mixed audio before attempting removal.

Examples:
  # Automatic mode (detects tempo/pitch)
  python script.py -r song.mp3 -m mixed.mp3 -o cleaned.mp3
  
  # Manual tempo ratio (if detection fails)
  python script.py -r song.mp3 -m mixed.mp3 -o cleaned.mp3 --tempo-ratio 1.35
  
  # Manual pitch shift
  python script.py -r song.mp3 -m mixed.mp3 -o cleaned.mp3 --pitch-shift 2
  
  # Adjust subtraction strength
  python script.py -r song.mp3 -m mixed.mp3 -o cleaned.mp3 --alpha 3.0
        '''
    )
    
    parser.add_argument('--reference', '-r', required=True,
                       help='Reference audio file')
    parser.add_argument('--mixed', '-m', required=True,
                       help='Mixed audio file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output cleaned audio file')
    
    parser.add_argument('--sample-rate', '-sr', type=int, default=22050,
                       help='Sample rate (default: 22050)')
    
    parser.add_argument('--tempo-ratio', '-tr', type=float, default=None,
                       help='Manual tempo ratio (mixed/reference), e.g., 1.35 for 35%% faster')
    parser.add_argument('--pitch-shift', '-ps', type=int, default=None,
                       help='Manual pitch shift in semitones')
    
    parser.add_argument('--alpha', '-a', type=float, default=2.0,
                       help='Subtraction strength (default: 2.0, higher = more aggressive)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TEMPO-ADAPTIVE AUDIO REMOVAL")
    print("="*60)
    
    remover = TempoMatchedRemover(args.reference, sr=args.sample_rate)
    
    remover.process_with_adaptation(
        mixed_path=args.mixed,
        output_path=args.output,
        manual_tempo_ratio=args.tempo_ratio,
        manual_pitch_shift=args.pitch_shift,
        alpha=args.alpha
    )
    
    print("\n✓ Done!")
    print("\nTIP: Listen to the '_adapted_reference.wav' file to verify")
    print("     it matches the mixed audio. If not, adjust parameters.")