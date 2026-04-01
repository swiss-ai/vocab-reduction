import torch
import time
import gc
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "Yusif-EPFL/apertus-1p5",
    "faruk-zahiragic/Apertus-1p5-96000-image-only",
    "faruk-zahiragic/Apertus-1p5-96000-audio-only"
]
NUM_TOKENS_TO_GENERATE = 50
NUM_WARMUP_RUNS = 5
NUM_RUNS = 100
PROMPT = "The future of artificial intelligence in Switzerland is"

def safe_reset_memory():
    """Safely attempts to reset memory stats, ignoring broken C++ bindings."""
    try:
        torch.cuda.reset_peak_memory_stats()
    except:
        pass

def cleanup():
    """Clear all GPU memory to prevent leakage between benchmarks."""
    gc.collect()
    torch.cuda.empty_cache()
    safe_reset_memory()
    torch.cuda.synchronize()

def benchmark_model(model_id):
    print(f"\n>>> Benchmarking: {model_id}")
    cleanup()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0", 
        trust_remote_code=True
    )
    
    weight_size = model.get_memory_footprint() / (1024**3)
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")

    print(f"  Running {NUM_WARMUP_RUNS} warmup iterations...")
    for _ in range(NUM_WARMUP_RUNS):
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    latencies = []
    peak_mems = []
    
    print(f"  Collecting {NUM_RUNS} samples...")
    for _ in range(NUM_RUNS):
        safe_reset_memory()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model.generate(
                **inputs, 
                max_new_tokens=NUM_TOKENS_TO_GENERATE,
                min_new_tokens=NUM_TOKENS_TO_GENERATE,
                do_sample=False,
                use_cache=True
            )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        tps = NUM_TOKENS_TO_GENERATE / (end_time - start_time)
        latencies.append(tps)
        peak_mems.append(torch.cuda.max_memory_allocated() / (1024**3))

    avg_tps = statistics.mean(latencies)
    std_tps = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
    del model
    cleanup()
    
    return {
        "model": model_id,
        "weight_gb": weight_size,
        "peak_vram_gb": max(peak_mems),
        "avg_tps": avg_tps,
        "std_tps": std_tps
    }

if __name__ == "__main__":
    results = []
    for m_id in MODELS:
        try:
            results.append(benchmark_model(m_id))
        except Exception as e:
            print(f"FAILED to benchmark {m_id}: {e}")

    print("\n" + "="*95)
    print(f"{'MODEL NAME':<40} | {'WEIGHTS':<10} | {'PEAK VRAM':<10} | {'TPS (AVG ± STD)':<20}")
    print("-" * 95)
    for r in results:
        tps_str = f"{r['avg_tps']:>6.2f} ± {r['std_tps']:<5.2f}"
        print(f"{r['model'][:40]:<40} | {r['weight_gb']:>7.2f} GB | {r['peak_vram_gb']:>7.2f} GB | {tps_str:<20}")
    print("="*95)