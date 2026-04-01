import torch
import torch.utils.benchmark as benchmark
import gc
from transformers import AutoModelForCausalLM

MODELS = [
    "Yusif-EPFL/apertus-1p5",
    "faruk-zahiragic/Apertus-1p5-96000-image-only",
    "faruk-zahiragic/Apertus-1p5-96000-audio-only"
]

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def measure_layer_latency(model_id):
    print(f"\n>>> Isolating Last Layer: {model_id}")
    cleanup()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda:0", 
        trust_remote_code=True
    )
    
    lm_head = model.lm_head
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    
    dummy_input = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16, device="cuda:0")
    
    for _ in range(10):
        _ = lm_head(dummy_input)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    latencies = []
    with torch.no_grad():
        for _ in range(1000):
            start_event.record()
            _ = lm_head(dummy_input)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

    avg_ms = sum(latencies) / len(latencies)
    theoretical_tps = 1000 / avg_ms 

    weight_gb = model.get_memory_footprint() / (1024**3)
    
    del model
    cleanup()
    
    return {
        "model": model_id,
        "vocab": vocab_size,
        "latency_ms": avg_ms,
        "theoretical_tps": theoretical_tps,
    }

if __name__ == "__main__":
    results = []
    for m in MODELS:
        try:
            results.append(measure_layer_latency(m))
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*95)
    print(f"{'MODEL NAME':<40} | {'VOCAB':<10} | {'HEAD LATENCY':<15} | {'MAX HEAD TPS':<10}")
    print("-" * 95)
    for r in results:
        print(f"{r['model'][:40]:<40} | {r['vocab']:<10} | {r['latency_ms']:>8.4f} ms | {r['theoretical_tps']:>10.2f}")
    print("="*95)
    print("Note: 'HEAD LATENCY' is the time for a single linear projection in the LM Head.")