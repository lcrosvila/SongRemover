import argparse
import math
import os
import tempfile

import torch
import torchaudio
from pydub import AudioSegment
from sam_audio import SAMAudio, SAMAudioProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-Audio separation with memory-safe chunking")
    parser.add_argument("--audio", default="mixed.mp3", help="Path to input audio file")
    parser.add_argument("--description", default="music", help="Text description for target separation")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Select device for inference"
    )
    parser.add_argument(
        "--gpu-index", type=int, default=-1, help="CUDA device index (use -1 for auto pick based on free memory)"
    )
    parser.add_argument(
        "--chunk-seconds", type=int, default=20, help="Chunk length in seconds (smaller reduces GPU memory)"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Use float16 autocast on CUDA to reduce memory"
    )
    parser.add_argument(
        "--out-target", default="target.wav", help="Output wav for isolated target"
    )
    parser.add_argument(
        "--out-residual", default="residual.wav", help="Output wav for residual"
    )
    return parser.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_best_cuda_index() -> int:
    count = torch.cuda.device_count()
    if count == 0:
        return -1
    best_i = 0
    best_free = -1
    for i in range(count):
        try:
            torch.cuda.set_device(i)
            free, total = torch.cuda.mem_get_info()
            if free > best_free:
                best_free = free
                best_i = i
        except Exception:
            continue
    return best_i


def separate_chunked(audio_path: str, description: str, device: torch.device, chunk_seconds: int,
                     mixed_precision: bool, out_target: str, out_residual: str, gpu_index: int):
    if device.type == "cuda":
        if gpu_index is None or gpu_index < 0:
            gpu_index = pick_best_cuda_index()
        if gpu_index >= 0:
            torch.cuda.set_device(gpu_index)
            print(f"Using device: cuda:{gpu_index}")
        else:
            print("CUDA requested but not available; falling back to CPU.")
            device = torch.device("cpu")
            print(f"Using device: {device}")
    else:
        print(f"Using device: {device}")

    model = SAMAudio.from_pretrained("facebook/sam-audio-small").to(device).eval()
    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-small")
    sr = processor.audio_sampling_rate
    print("Model loaded successfully!")

    # Load audio and chunk with pydub (uses ffmpeg)
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)
    chunk_ms = max(1000, int(chunk_seconds * 1000))
    num_chunks = math.ceil(total_ms / chunk_ms)
    print(f"Input length: {total_ms/1000:.1f}s | Chunks: {num_chunks} x {chunk_ms/1000:.1f}s")

    target_chunks = []
    residual_chunks = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if (device.type == "cuda" and mixed_precision) else None
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in range(num_chunks):
            start = idx * chunk_ms
            end = min((idx + 1) * chunk_ms, total_ms)
            clip = audio[start:end]

            # Export chunk to a temp wav (processor accepts file path input)
            chunk_path = os.path.join(tmpdir, f"chunk_{idx}.wav")
            clip.export(chunk_path, format="wav")

            inputs = processor(audios=[chunk_path], descriptions=[description])
            if device.type != "cpu":
                inputs = inputs.to(device)

            if autocast_ctx is not None:
                ctx = autocast_ctx
            else:
                # no-op context
                class _NullCtx:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc, tb):
                        return False

                ctx = _NullCtx()

            with torch.inference_mode(), ctx:
                result = model.separate(inputs)

            # Collect chunk outputs
            tgt = result.target[0].detach().cpu().unsqueeze(0)  # [1, samples]
            res = result.residual[0].detach().cpu().unsqueeze(0)
            target_chunks.append(tgt)
            residual_chunks.append(res)

            # Cleanup per-chunk to avoid OOM
            del inputs, result, tgt, res
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Concatenate along time and save
    target_full = torch.cat(target_chunks, dim=1)
    residual_full = torch.cat(residual_chunks, dim=1)

    torchaudio.save(out_target, target_full, sr)
    torchaudio.save(out_residual, residual_full, sr)

    print("Separation complete!")
    print(f"- {out_target}: The isolated sound you described")
    print(f"- {out_residual}: Everything else")


def main():
    args = parse_args()
    device = pick_device(args.device)

    # Helpful CUDA allocator tuning via env (optional):
    # You can set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 before running for better fragmentation behavior.

    separate_chunked(
        audio_path=args.audio,
        description=args.description,
        device=device,
        chunk_seconds=args.chunk_seconds,
        mixed_precision=args.mixed_precision,
        out_target=args.out_target,
        out_residual=args.out_residual,
        gpu_index=args.gpu_index,
    )


if __name__ == "__main__":
    main()