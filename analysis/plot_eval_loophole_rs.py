import numpy as np
import scipy.stats
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import pandas as pd
plt.rcParams['font.family'] = 'Arial'
import os


def load_adapter_eval_result(dataset_name, layer_ids, scales, model_identifier):
    if dataset_name.startswith('loophole'):
        OUTPUT_DIR = 'results/model_eval/scaled_lora/loophole'
    else:
        OUTPUT_DIR = 'results/model_eval/scaled_lora/rule'

    lora_directions = ['letter', 'spirit']

    rule_lora_rating_dict = {}

    for lora_direction in lora_directions:
        rule_lora_rating_dict[lora_direction] = {}
        for layer_id in layer_ids:
            rule_lora_rating_dict[lora_direction][layer_id] = {}
            for scale in scales:
                rule_lora_rating_dict[lora_direction][layer_id][scale] = {}
                model_judgment_dict = json.load(open(os.path.join(OUTPUT_DIR, f"{model_identifier}/{model_identifier}_{layer_id:02d}_{lora_direction}_scale_{scale}_{dataset_name}_yes_probs.json")))
                rule_lora_rating_dict[lora_direction][layer_id][scale] = model_judgment_dict

    return rule_lora_rating_dict


model_meta_info = {
    "Llama-3.1-70B-Instruct": {
        "selected_layer_id": 26,
        "layer_ids": [26],
    },
    "Qwen3.5-27B": {
        "selected_layer_id": 32,
        "layer_ids": [32],
    },   
}


behavior_type2pretty_label = dict(zip(['compliance', 'underinclusion', 'overinclusion', 'core'], ['Compliance', 'Underinclusion', 'Overinclusion', 'Noncompliance']))

for model_identifier in ['Llama-3.1-70B-Instruct', "Qwen3.5-27B"]:

    selected_layer_id = model_meta_info[model_identifier]['selected_layer_id']
    layer_ids = model_meta_info[model_identifier]['layer_ids']

    rs_meta_info = {
        'base': {'lora_direction': 'letter', 'scale': 0},
        'letter': {'lora_direction': 'letter', 'scale': 100},
        'spirit': {'lora_direction': 'spirit', 'scale': 100},
    }

    scales = [0, 100]

    # eval_dataset_names = ['novel_rule_stories', 'held-out_novel_rule_stories']
    # dataset_name2fname = {'novel_rule_stories': 'novel_stories.csv', 'held-out_novel_rule_stories': 'extra_novel_stories.csv'}

    eval_result_all = {}

    conditions = ['spirit', 'base', 'letter']

    dataset_name = 'loophole_eval_stimuli'

    eval_result_all[f'lora_{dataset_name}'] = {}
    eval_result_all[f'lora_{dataset_name}'][model_identifier] = load_adapter_eval_result(dataset_name, layer_ids, scales, model_identifier)


    action_conditions = ['compliance', 'loophole', 'noncompliance']
    power_relations = ['up', 'equal', 'down']
    stimuli_df = pd.read_csv('stimuli/loophole_scenarios.csv')
    story_names = stimuli_df['story name'].tolist()

    a_colors = ['#66A182', 'orange', "red"]
    action2color = dict(zip(action_conditions, a_colors))

    fig = plt.figure(figsize=(2.8, 2.5))
    ax = plt.gca()

    for action_condition in action_conditions:
        xs = [-1, 0, 1]

        ys = [np.mean(
            [
            np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier][rs_meta_info[cond]['lora_direction']][selected_layer_id][rs_meta_info[cond]['scale']][story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names
            ]
            ) for cond in conditions]
        yerrs = [1.96*scipy.stats.sem(
            [
            np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier][rs_meta_info[cond]['lora_direction']][selected_layer_id][rs_meta_info[cond]['scale']][story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names
            ]
        ) for cond in conditions]
        ax.errorbar(
            xs, ys, yerr=yerrs, marker='o', color=action2color[action_condition], ls='-', capsize=5
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('P("Yes")')
    ax.set_title(model_identifier)

    ax.set_xticks([-1, 0, 1])

    ax.set_xticklabels(['Spirit', 'Base', 'Letter'])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(0, 1.02)

    plt.savefig(f'fig/{model_identifier}_{dataset_name}_yes_prob_no-legend.pdf', bbox_inches='tight')

    color_legend = ax.legend(handles=[Line2D([], [], marker='o', color=action2color[action_condition], label=action_condition.title()) for action_condition in action_conditions[::-1]],
                            loc='center left', bbox_to_anchor=(0, 0.25), ncol=1, frameon=False, fontsize=8)
    ax.add_artist(color_legend)

    plt.savefig(f'fig/{model_identifier}_{dataset_name}_yes_prob.pdf', bbox_inches='tight', bbox_extra_artists=[color_legend])
    plt.show()
