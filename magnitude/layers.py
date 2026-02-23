from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DataCollatorWithPadding
import torch
import heapq
import json

# torch.manual_seed(67)
# np.random.seed(67)
# random.seed(67)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(67)

model_name = "microsoft/deberta-v3-small"
model = DebertaV2ForSequenceClassification.from_pretrained(model_name)



# Separate heaps
top_params_l1   = []
top_params_spec = []
top_params_fro  = []

k = 40  # top-k

for name, param in model.named_parameters():
    if param.requires_grad and len(param.shape) != 1:
        data = param.data.float()  # cast to float32 to avoid Half precision errors

        # Compute norms
        l1_norm   = torch.norm(data, p=1).item()
        fro_norm  = torch.norm(data, p='fro').item()
        spec_norm = torch.linalg.norm(data, 2).item()

        # Push into each heap
        heapq.heappush(top_params_l1,   (l1_norm, name, param.shape))
        heapq.heappush(top_params_fro,  (fro_norm, name, param.shape))
        heapq.heappush(top_params_spec, (spec_norm, name, param.shape))

        # Keep heap size at most k
        if len(top_params_l1)   > k: heapq.heappop(top_params_l1)
        if len(top_params_fro)  > k: heapq.heappop(top_params_fro)
        if len(top_params_spec) > k: heapq.heappop(top_params_spec)

# Sort each heap descending
top_params_l1   = heapq.nlargest(k, top_params_l1,   key=lambda x: x[0])
top_params_fro  = heapq.nlargest(k, top_params_fro,  key=lambda x: x[0])
top_params_spec = heapq.nlargest(k, top_params_spec, key=lambda x: x[0])

# Print results
print("\nTop by L1 norm:")
for l1_norm, name, shape in top_params_l1:
    print(f"{name:60} | shape: {shape} | l1_norm: {l1_norm:.6f}")

print("\nTop by Frobenius norm:")
for fro_norm, name, shape in top_params_fro:
    print(f"{name:60} | shape: {shape} | fro_norm: {fro_norm:.6f}")

print("\nTop by Spectral norm:")
for spec_norm, name, shape in top_params_spec:
    print(f"{name:60} | shape: {shape} | spectral_norm: {spec_norm:.6f}")

results = {
    "top_l1": {key: val for key, val in top_params_l1},
    "top_fro": {key: val for key, val in top_params_fro},
    "top_spec": {key: val for key, val in top_params_spec}
}

# Save to JSON file
with open("./layer_ranking.json", "w") as f:
    json.dump(results, f, indent=4)