import json
from train_base import train

k = 0.5

with open(f'./layer_ranking.json', 'r') as f:
    results = json.load(f)
    modules = list(results['top_spec'].keys())
    modules = modules[:round(len(modules) * k)]

train(epochs=1, modules=modules, k=k, magnitude_type='spec')