import numpy as np
from scipy.stats import bootstrap
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


behavior_types = ['compliance', 'underinclusion', 'overinclusion', 'core']
behavior_colors = ['#66A182', 'orange', 'violet', 'red']
behavior_type2color = dict(zip(behavior_types, behavior_colors))

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

    eval_dataset_names = ['novel_rule_stories', 'held-out_novel_rule_stories']
    dataset_name2fname = {'novel_rule_stories': 'novel_stories.csv', 'held-out_novel_rule_stories': 'extra_novel_stories.csv'}

    eval_result_all = {}

    for dataset_name in eval_dataset_names:
        eval_result_all[f'lora_{dataset_name}'] = {}
        eval_result_all[f'lora_{dataset_name}'][model_identifier] = load_adapter_eval_result(dataset_name, layer_ids, scales, model_identifier)


    conditions = ['spirit', 'base', 'letter']

    ytick_labels = ['Compliance', 'Underinclusion', 'Overinclusion', 'Core']

    question_types = ['question_infraction', 'question_text', 'question_purpose']
    pretty_subfig_titles = [
        'Question (rule violation)\n(e.g. "Did Jack violated the rule?")',
        'Question (rule\'s text)\n(e.g. "Did Jack fail to keep the dog on a leash?")',
        'Question (rule\'s purpose)\n(e.g. "Was Jack\'s dog under control?")'
    ]

    for dataset_name in eval_dataset_names:

        stimuli_df = pd.read_csv(os.path.join('stimuli/', dataset_name2fname[dataset_name]))
        groups = stimuli_df['group'].tolist()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3.5))

        for col_idx, ax in enumerate(axes):
            question_type = question_types[col_idx]

            for behavior_type in behavior_types:

                xs = [-1, 0, 1]

                ys = [np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier][rs_meta_info[cond]['lora_direction']][selected_layer_id][rs_meta_info[cond]['scale']][group][behavior_type][question_type] for group in groups]) for cond in conditions]
                
                yerrs = [[], []]
                for cond in conditions:
                    vals = [eval_result_all[f'lora_{dataset_name}'][model_identifier][rs_meta_info[cond]['lora_direction']][selected_layer_id][rs_meta_info[cond]['scale']][group][behavior_type][question_type] for group in groups]
                    res = bootstrap((vals,), np.mean, confidence_level=0.95, method='percentile')
                    ci_low, ci_upp = res.confidence_interval
                    mean_val = np.mean(vals)
                    yerrs[0].append(mean_val - ci_low)
                    yerrs[1].append(ci_upp - mean_val)

                ax.plot(
                    xs, ys, marker='o', color=behavior_type2color[behavior_type], ls='-',
                )

                ax.errorbar(
                    xs, ys, marker='o', yerr=yerrs, color=behavior_type2color[behavior_type], ls='-', capsize=5
                )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if col_idx == 0:
                ax.set_ylabel("P('yes')")

            if col_idx > 0:
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])

            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels([r'$\bf{Spirit}$'+'\nadapted', r'$\bf{Base}$'+'\nmodel', r'$\bf{Letter}$'+'\nadapted'])

            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(0, 1.05)

            ax.set_title(pretty_subfig_titles[col_idx], fontsize=10)
            

            if col_idx == 1:
                color_legend = ax.legend(handles=[Line2D([], [], color=behavior_type2color[behavior_type], marker='o', label=behavior_type2pretty_label[behavior_type]) for behavior_type in behavior_types], 
                                        loc='center', bbox_to_anchor=(0.5, -0.34), ncol=4)
                ax.add_artist(color_legend)

                annotated_title = ax.annotate(f'{model_identifier}', (0.5, 1.25), xycoords='axes fraction', fontsize=12, ha='center')
        

        plt.tight_layout()
        plt.savefig(f'fig/{model_identifier}_question_comparison_panel_{dataset_name}_yes_prob.pdf', bbox_inches='tight', bbox_extra_artists=[color_legend, annotated_title])
        plt.show()


    # Plot results for each question type separately
    for dataset_name in eval_dataset_names:
        stimuli_df = pd.read_csv(os.path.join('stimuli/', dataset_name2fname[dataset_name]))
        groups = stimuli_df['group'].tolist()

        for question_idx, question_type in enumerate(question_types):
            plt.figure(figsize=(3.2, 2.36))
            ax = plt.gca()

            for behavior_type in behavior_types:

                xs = [-1, 0, 1]

                ys = [np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier][rs_meta_info[cond]['lora_direction']][selected_layer_id][rs_meta_info[cond]['scale']][group][behavior_type][question_type] for group in groups]) for cond in conditions]

                yerrs = [[], []]
                for cond in conditions:
                    vals = [eval_result_all[f'lora_{dataset_name}'][model_identifier][rs_meta_info[cond]['lora_direction']][selected_layer_id][rs_meta_info[cond]['scale']][group][behavior_type][question_type] for group in groups]
                    res = bootstrap((vals,), np.mean, confidence_level=0.95, method='percentile')
                    ci_low, ci_upp = res.confidence_interval
                    mean_val = np.mean(vals)
                    yerrs[0].append(mean_val - ci_low)
                    yerrs[1].append(ci_upp - mean_val)

                ax.plot(
                    xs, ys, marker='o', color=behavior_type2color[behavior_type], ls='-',
                )

                ax.errorbar(
                    xs, ys, marker='o', yerr=yerrs, color=behavior_type2color[behavior_type], ls='-', capsize=5
                )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xticks([-1, 0, 1])
            
            ax.set_xticklabels(['Spirit', 'Base', 'Letter'])
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-0.05, 1.05)

            ax.set_title(pretty_subfig_titles[question_idx], fontsize=10)
            
            color_legend = ax.legend(handles=[Line2D([], [], color=behavior_type2color[behavior_type], marker='o', label=behavior_type2pretty_label[behavior_type]) for behavior_type in ['compliance', 'underinclusion', 'core', 'overinclusion']], 
                                    loc='center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
            ax.add_artist(color_legend)

            annotated_title = ax.annotate(f'{model_identifier}', (0.5, 1.25), xycoords='axes fraction', fontsize=12, ha='center')
    
            plt.savefig(f'fig/{model_identifier}_{dataset_name}_{question_type}_yes_prob.pdf', bbox_inches='tight', bbox_extra_artists=[color_legend, annotated_title])
            plt.show()