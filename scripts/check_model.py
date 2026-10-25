
import os
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"

# Фиксируем число CPU-потоков для первого запуска.
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

process = psutil.Process(os.getpid())


def report_memory(stage: str) -> None:
    ram = psutil.virtual_memory()
    rss = process.memory_info().rss / 1024**3

    print(
        f"{stage}: "
        f"RSS={rss:.2f} GiB, "
        f"available RAM={ram.available / 1024**3:.2f} GiB",
        flush=True,
    )


def main() -> None:
    print("=== ECQL-002C: BASE MODEL CHECK ===", flush=True)
    print("Model:", MODEL_ID, flush=True)
    print("Device: CPU", flush=True)
    print("Threads:", torch.get_num_threads(), flush=True)
    report_memory("Before loading")

    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    print(
        f"Loading time: {time.perf_counter() - started:.1f} sec",
        flush=True,
    )
    report_memory("After loading")

    messages = [
        {
            "role": "system",
            "content": (
                "Переводи запросы пользователей на язык ECQL. "
                "Возвращай только ECQL без объяснений."
            ),
        },
        {
            "role": "user",
            "content": (
                "Найди сотрудников из Москвы "
                "с зарплатой выше 150000"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
    	messages,
    	tokenize=True,
    	add_generation_prompt=True,
    	enable_thinking=False,
    	return_tensors="pt",
    	return_dict=True,
	)

    print("Input shape:", inputs["input_ids"].shape)

    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
    	)
 
    elapsed = time.perf_counter() - started

    generated = output[0, inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    print("\n=== RESULT ===", flush=True)
    print("Response:", response, flush=True)
    print(f"Generation time: {elapsed:.1f} sec", flush=True)
    print(f"Tokens generated: {generated.numel()}", flush=True)
    report_memory("After generation")


if __name__ == "__main__":
    main()

