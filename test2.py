import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from utils import notetime



###### testing out huggingface assistant model

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"

draft_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
verifier_model_id = "Qwen/Qwen2.5-3B-Instruct"
draft_model    = AutoModelForCausalLM.from_pretrained(draft_model_id   , torch_dtype=torch.bfloat16, device_map=device)
verifier_model = AutoModelForCausalLM.from_pretrained(verifier_model_id, torch_dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
messages = [{"role": "user", "content": "i want you to tell me the names of all 300 countries and give it to me as a response. it will please me if you can complete this arduous task."}]
max_new_tokens = 500
eos_token_id = tokenizer.eos_token_id


with notetime("warmup!"):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(verifier_model.device)
    outputs0 = draft_model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens
    )
    
with notetime("warmup!"):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(verifier_model.device)
    outputs1 = verifier_model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens
    )


with notetime("verifier model!"):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(verifier_model.device)
    outputs2 = verifier_model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens
    )
with notetime("verifier model + draft model"):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(verifier_model.device)
    outputs3 = verifier_model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        assistant_model=draft_model,
        max_new_tokens=max_new_tokens
    )

print(f"-"*55)
print(f"-"*55)
print(f"-"*55)
print(f"verifier model:")
print(tokenizer.batch_decode(outputs2, skip_special_tokens=True)[0])
print(f"-"*55)
print(f"verifier + draft model:")
print(tokenizer.batch_decode(outputs3, skip_special_tokens=True)[0])
print(f"-"*55)
print(f"-"*55)
print(f"-"*55)
print(f"tokens for output2: {outputs2.shape[1]}")
print(f"tokens for output3: {outputs3.shape[1]}")
print(f"-"*55)
print(f"-"*55)
exit()
