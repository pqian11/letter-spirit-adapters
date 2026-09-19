import numpy as np
import scipy.stats
import json
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import os


model_identifiers = ['Llama-3.1-70B-Instruct', 'Qwen3.5-27B']
model_identifier2lora_layer_id = {'Llama-3.1-70B-Instruct':26, 'Qwen3.5-27B':32}

model_yes_probs_all = {}

for model_identifier in model_identifiers:
    lora_layer_id = model_identifier2lora_layer_id[model_identifier]
    model_yes_probs_all[model_identifier] = {}
    model_yes_probs_all[model_identifier]['letter'] = json.load(open('results/model_eval/reddit/{}_letter_{}_reddit_yes_probs.json'.format(model_identifier, lora_layer_id)))
    model_yes_probs_all[model_identifier]['base'] = json.load(open('results/model_eval/reddit/{}_base_reddit_yes_probs.json'.format(model_identifier)))
    model_yes_probs_all[model_identifier]['spirit'] = json.load(open('results/model_eval/reddit/{}_spirit_{}_reddit_yes_probs.json'.format(model_identifier, lora_layer_id)))


conds = ['spirit', 'base', 'letter']
pretty_conds =['Spirit', 'Base', 'Letter']

for model_identifier in model_identifiers:
    plt.figure(figsize=(2.8, 2.5))
    ax = plt.gca()

    ys = []
    yerrs = []
    for cond_idx, cond in enumerate(conds):
        y = np.mean(model_yes_probs_all[model_identifier][cond])
        yerr = 1.96*scipy.stats.sem(model_yes_probs_all[model_identifier][cond])
        ys.append(y)
        yerrs.append(yerr)

    plt.errorbar(range(len(ys)), ys, yerr=yerrs, color='orange', capsize=3, marker='o')
    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels(pretty_conds)

    ax.set_xlim(-0.25, 2.25)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('P("Yes")')

    plt.savefig(f'fig/{model_identifier}_reddit_mc_10k_yes_prob.pdf', bbox_inches='tight')
    plt.show()