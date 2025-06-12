import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline

llm_model_map = {
    "exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
    "hyperclova": "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    "kanana": "kakaocorp/kanana-1.5-8b-instruct-2505",
    "yanolja" : "yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview",
}
system_prompt_map = {
    "exaone": "You are EXAONE model from LG AI Research, a helpful assistant.",
    "qwen": "You are an AI secretary named Qwen. Always answer all your questions in Korean only.",
    "hyperclova": "You are HyperCLOVA X, a Korean language expert.",
    "kanana": "You are Kanana, a bilingual assistant.",
    "yanolja" : "You are EEVE model, a helpful assistant."
}

def get_hf_llm(model_name: str, system_prompt: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.7
    )

    def llm_func(prompt):
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                output = model.generate(
                    input_ids,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=1024
                )
                return tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Chat template error: {e}")
                return pipe(prompt)[0]["generated_text"]
        else:
            return pipe(prompt)[0]["generated_text"]

    return llm_func