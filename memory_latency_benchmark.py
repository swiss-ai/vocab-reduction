import torch
import time
import gc
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "Yusif-EPFL/apertus-1p5",
    "Yusif-EPFL/Apertus-1p5-96000-text-only"
]
NUM_TOKENS_TO_GENERATE = 50
NUM_RUNS = 100
PROMPT = "The future of artificial intelligence in Switzerland is"

def cleanup():
    """Clear all GPU memory to prevent leakage between benchmarks."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def benchmark_model(model_id):
    print(f"\n>>> Starting Benchmark: {model_id}")
    cleanup()
    
    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mem_start = torch.cuda.memory_allocated() / (1024**3)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0", 
        trust_remote_code=True
    )
    
    mem_after_load = torch.cuda.memory_allocated() / (1024**3)
    weight_size = mem_after_load - mem_start

    # 2. Warmup
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    # 3. Latency & Peak Memory Loop
    latencies = []
    peak_mems = []
    
    for i in range(NUM_RUNS):
        torch.cuda.reset_peak_memory_stats()
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
        print(f"  Run {i+1}: {tps:.2f} tokens/sec")

    # 4. Cleanup model from memory before returning
    avg_tps = statistics.mean(latencies)
    max_vram = max(peak_mems)
    
    del model
    cleanup()
    
    return {
        "model": model_id,
        "weight_gb": weight_size,
        "peak_vram_gb": max_vram,
        "avg_tps": avg_tps
    }

if __name__ == "__main__":
    results = []
    for m_id in MODELS:
        try:
            res = benchmark_model(m_id)
            results.append(res)
        except Exception as e:
            print(f"FAILED to benchmark {m_id}: {e}")

    # Final Comparison Table
    print("\n" + "="*80)
    print(f"{'MODEL NAME':<40} | {'WEIGHTS':<10} | {'PEAK VRAM':<10} | {'TPS':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['model'][:40]:<40} | {r['weight_gb']:>7.2f} GB | {r['peak_vram_gb']:>7.2f} GB | {r['avg_tps']:>7.2f}")
    print("="*80)