import itertools
import pandas as pd
import os
from remover import TempoMatchedRemover  # Importing the class directly

# --- CONFIGURATION ---
INPUT_REF = "song.mp3"  
INPUT_MIX = "mixed.mp3"       
OUTPUT_DIR = "grid_results"

# Define the Grid
param_grid = {
    "n_fft": [2048, 4096],
    "hop_length": [128, 256, 512],
    
    # Subtraction Strength (Standard is ~2.0. We go up to 5.0)
    "alpha": [2.0, 3.5, 5.0],     
    
    # Time Smear (Standard is 1. We go up to 10 frames ~0.1s)
    "time_dilation": [3, 6, 10],     
    
    # NEW: Pitch Smear (Catch vibrato/tuning drift)
    # 1 = Off, 5 = Aggressive smear
    "freq_dilation": [1, 3, 5],

    # Floor (0.0 = Allow absolute silence / max removal)
    "floor": [0.0, 0.05],   
    
    "lag_offset": [0.0, 0.05] 
}
# ---------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Initializing Remover...")
engine = TempoMatchedRemover(INPUT_REF, INPUT_MIX, sr=44100)

results = []
print(f"Starting Aggressive Grid Search with {len(combinations)} combinations...")

for i, params in enumerate(combinations):
    fname = f"run_{i:03d}_a{params['alpha']}_tdil{params['time_dilation']}_fdil{params['freq_dilation']}.wav"
    out_path = os.path.join(OUTPUT_DIR, fname)
    
    print(f"[{i+1}/{len(combinations)}] Alpha={params['alpha']} | TimeDil={params['time_dilation']} | FreqDil={params['freq_dilation']}")
    
    try:
        metrics = engine.run_experiment(params, output_path=out_path)
        row = {**params, **metrics, "filename": fname}
        results.append(row)
    except Exception as e:
        print(f"  Error: {e}")

df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "results_log.csv")
df.to_csv(csv_path, index=False)

print("\nGrid Search Complete!")