
def remove_last_t_from_kvcache(kvcache, t):
    for i, (k_layer, v_layer) in enumerate(kvcache):
        if k_layer is not None:
            kvcache[i] = (
                k_layer[:, :, :-t, :],  # remove last num_to_remove keys
                v_layer[:, :, :-t, :]   # remove last num_to_remove values
            )
"""
1. Inference loop:
- inputs object with keys "input_ids" and "attention_mask" # TODO: code currently works for mask of shape [1, L] but won't for a 3d [1, L, L] square mask.
- past_key_values_draft : (DynamicCache(config=draft_model.config))
- past_key_values_verifier: (DynamicCache(config=draft_model.config))
- k = 10
- max_new_tokens = 100
"""

# initialize generated_ids for the recursion
generated_ids = inputs["input_ids"]

# infer draft model k times
start_position = 0 if past_key_values_draft[0][0] is None else past_key_values_draft[0].shape[-2]
draft_cache_position = torch.arange(start_position, inputs.input_ids.shape[1], dtype=torch.int64, device=draft_model.device)
draft_logits = torch.empty_like(inputs.input_ids[:, 0:0]) # tensor of shape [N, 0]

for draft_iter in range(k):
	outputs = draft_model(**inputs, cache_position=draft_cache_position, past_key_values=past_key_values_draft, use_cache=True) # TODO: add stopping condition here.
	
	# sampling - greedy
	next_token_ids = outputs.logits[:, -1].argmax(-1)
	draft_logits = torch.concat([draft_logits, outputs.logits[:, -1:, :]], dim=-2)
	
	generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
	attention_mask = inputs["attention_mask"]
	attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
	inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
	draft_cache_position = draft_cache_position[-1:] + 1 # add one more position for the next token to be decoded.



# run verifier in parallel
inputs = {"input_ids": generated_ids, "attention_mask": inputs["attention_mask"]}
start_position = 0 if past_key_values_verifier[0][0] is None else past_key_values_verifier[0].shape[-2]
verifier_cache_position = torch.arange(start_position, inputs.input_ids.shape[1], dtype=torch.int64, device=verifier_model.device)
outputs = verifier_model(**inputs, cache_position=verifier_cache_position, past_key_values=past_key_values_verifier, use_cache=True) # TODO: add stopping condition here.

# determine the number of accepted guesses n
verifier_logits = outputs.logits[:, -(draft_logits.shape[-2]+1):-1, :]
verifier_probs = verifier_logits.softmax(-1)
draft_probs = draft_logits.softmax(-1)
prob_ratios = verifier_probs / draft_probs # TODO: make the sampling safe for zero division errors?
prob_ratios_per_token = torch.gather(prob_ratios, dim=-2, index=generated_ids.unsqueeze(dim=-2))

## sample tokens until rejection criteria is met
### r1 ∼ U(0, 1), . . . , rγ ∼ U(0, 1)
uniform_mask = torch.randn_like(prob_ratios_per_token) > prob_ratios_per_token
uniform_mask = torch.concat([uniform_mask, torch.ones(uniform_mask.shape[0], 1, dtype=torch.bool)]) # verifier token index, if all accept.
### n ← min({i − 1 | 1 ≤ i ≤ γ, ri > pi(x) / qi(x)} ∪ {γ})
n = argmax(uniform_mask, dim=-1)

# destroy the last (k-n) key-values from draft kvcache
## note that this implementation is not batched.
remove_last_t_from_kvcache(past_key_values_draft, k-n)

# destroy the last (k-n) key-values from verifier kvcache
remove_last_t_from_kvcache(past_key_values_verifier, k-n)

# adjust the final token from verifier if needed
if n < k:
	norm = 1 /  torch.sum(torch.maximum(torch.tensor(0), draft_probs - verifier_probs), dim=-1)
	# norm = 1 / [ sum(q(xi) - p(xi) if p(xi) < q(xi) else 0) ]
	extra_token_logits = norm[:, -1-(k-n), :] * (max(0, draft_logits[:, -1-(k-n), :] - verifier_logits[:, -1-(k-n), :]))
else:
	extra_token_logits = verifier_logits[:, -1, :]
extra_token_logits = extra_token_logits.unsqueeze(1)

# sample one extra token from adjusted distribution
# sampling - greedy
extra_token = extra_token_logits.argmax(-1)

# add the generated ids to inputs object to close the recursion, and adjust attention mask. rebuild attention mask entirely?
generated_ids = torch.concat([generated_ids, extra_token], dim=-1)
attention_mask = attention_mask.new_ones((attention_mask.shape[0], generated_ids.shape[-1]))
inputs = {"inputs": generated_ids, "attention_mask": attention_mask}
