import numpy as np
import scipy.stats
from scipy.stats import bootstrap
import json
import matplotlib.pyplot as plt
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


def load_contrastive_activation_eval_result(dataset_name, layer_ids, model_identifier):
    if dataset_name.startswith('loophole'):
        OUTPUT_DIR = 'results/model_eval/steering/loophole'
    else:
        OUTPUT_DIR = 'results/model_eval/steering/rule'

    steering_rating_dict = {}

    for layer_id in layer_ids:

        model_judgment_dict = json.load(open(os.path.join(OUTPUT_DIR, f"{model_identifier}/{model_identifier}_{layer_id:02d}_{dataset_name}_yes_probs.json")))
        steering_rating_dict[layer_id] = model_judgment_dict
    return steering_rating_dict


# def load_contrastive_activation_eval_result_on_loophole_stimuli(dataset_name, layer_ids, model_identifier):
#     OUTPUT_DIR = '../results/model_eval/steering/loophole'

#     steering_rating_dict = {}

#     for layer_id in layer_ids:
#         model_judgment_dict = json.load(open(os.path.join(OUTPUT_DIR, f"{model_identifier}/{model_identifier}_{layer_id:02d}_{dataset_name}_yes_probs.json")))
#         steering_rating_dict[layer_id] = model_judgment_dict
#     return steering_rating_dict



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

for model_identifier in ['Llama-3.1-70B-Instruct', "Qwen3.5-27B"]:

    selected_layer_id = model_meta_info[model_identifier]['selected_layer_id']
    layer_ids = model_meta_info[model_identifier]['layer_ids']
        
    scales = [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150]
    coefs = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]

    eval_dataset_names = ['novel_rule_stories', 'held-out_novel_rule_stories']
    dataset_name2fname = {'novel_rule_stories': 'novel_stories.csv', 'held-out_novel_rule_stories': 'extra_novel_stories.csv'}

    eval_result_all = {}

    for dataset_name in eval_dataset_names:
        eval_result_all[f'lora_{dataset_name}'] = {}
        eval_result_all[f'lora_{dataset_name}'][model_identifier] = load_adapter_eval_result(dataset_name, layer_ids, scales, model_identifier)
        eval_result_all[f'contrastive_{dataset_name}'] = {}
        eval_result_all[f'contrastive_{dataset_name}'][model_identifier] = load_contrastive_activation_eval_result(dataset_name, layer_ids, model_identifier)


    methods = ['letter', 'spirit', 'contrastive']
    method_shapes = ['star', 'o', 'o']

    ytick_labels = ['Compliance', 'Underinclusion', 'Overinclusion', 'Core']

    ##################################
    # Plot panel on rule stimuli
    ##################################
    for dataset_name in eval_dataset_names:

        stimuli_df = pd.read_csv(os.path.join('stimuli/', dataset_name2fname[dataset_name]))
        groups = stimuli_df['group'].tolist()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.2))

        for col_idx, ax in enumerate(axes):
            method = methods[col_idx]

            if col_idx < 2:
                lora_direction = method
                for behavior_type in behavior_types:

                    xs = scales

                    ys = [np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier][lora_direction][selected_layer_id][scale][group][behavior_type]['question_infraction'] for group in groups]) for scale in scales]

                    yerrs = [[], []]
                    for scale in scales:
                        vals = [eval_result_all[f'lora_{dataset_name}'][model_identifier][lora_direction][selected_layer_id][scale][group][behavior_type]['question_infraction'] for group in groups]
                        res = bootstrap((vals,), np.mean, confidence_level=0.95, method='percentile')
                        ci_low, ci_upp = res.confidence_interval
                        mean_val = np.mean(vals)
                        yerrs[0].append(mean_val - ci_low)
                        yerrs[1].append(ci_upp - mean_val)


                    ax.plot(
                        xs, ys, marker='o', color=behavior_type2color[behavior_type], ls='-',
                    )

                    ax.fill_between(xs, np.array(ys) - yerrs[0], np.array(ys) + yerrs[1], color=behavior_type2color[behavior_type], 
                        linewidth=0, alpha=0.2)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlabel('Inference-time scaling of adapter weights (Percentage)')
                ax.set_ylabel('P("Yes")')
                ax.set_title(lora_direction.title()+' adapter')
            else:
                for behavior_type in behavior_types:

                    xs = coefs

                    y_all_list = []
                    
                    for coef in coefs:
                        if coef == 0:
                            y_all = [eval_result_all[f'lora_{dataset_name}'][model_identifier]['letter'][selected_layer_id][0][group][behavior_type]['question_infraction'] for group in groups]
                        else:
                            coef_str = f'{coef:.1f}'
                            y_all = [eval_result_all[f'contrastive_{dataset_name}'][model_identifier][selected_layer_id][coef_str][group][behavior_type] for group in groups]
                        y_all_list.append(y_all)

                    ys = [np.mean(y_all) for y_all in y_all_list]
                    yerrs = [1.96*scipy.stats.sem(y_all) for y_all in y_all_list]
                    ax.plot(
                        xs, ys, marker='o', color=behavior_type2color[behavior_type], ls='-',
                    )
                    ax.fill_between(xs, np.array(ys) - yerrs, np.array(ys) + yerrs, color=behavior_type2color[behavior_type], 
                        linewidth=0, alpha=0.2)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xlabel(r'Steering coefficient $\beta$')
                ax.set_ylabel('P("Yes")')
                ax.set_title(f'Contrastive Activation Addition steering')

            if col_idx == 1:
                color_legend = ax.legend(handles=[mpatches.Patch(facecolor=behavior_type2color[behavior_type], label=behavior_type.title()) for behavior_type in behavior_types], 
                                        loc='center', bbox_to_anchor=(0.5, -0.35), ncol=4)
                ax.add_artist(color_legend)

                annotated_title = ax.annotate(f'{model_identifier}', (0.5, 1.15), xycoords='axes fraction', fontsize=14, ha='center')
        
        plt.tight_layout()
        plt.savefig(f'fig/{model_identifier}_method_comparison_panel_{dataset_name}_yes_prob.pdf', bbox_inches='tight', bbox_extra_artists=[color_legend, annotated_title])
        plt.show()


    ##################################
    # Plot panel on loophole stimuli
    ##################################
    dataset_name = 'loophole_eval_stimuli'
    eval_result_all[f'contrastive_{dataset_name}'] = {}
    eval_result_all[f'contrastive_{dataset_name}'][model_identifier] = load_contrastive_activation_eval_result(dataset_name, layer_ids, model_identifier)
    eval_result_all[f'lora_{dataset_name}'] = {}
    eval_result_all[f'lora_{dataset_name}'][model_identifier] = load_adapter_eval_result(dataset_name, layer_ids, scales, model_identifier)

    action_conditions = ['compliance', 'loophole', 'noncompliance']
    power_relations = ['up', 'equal', 'down']
    stimuli_df = pd.read_csv('stimuli/loophole_scenarios.csv')
    story_names = stimuli_df['story name'].tolist()

    a_colors = ['#66A182', 'orange', "red"]
    action2color = dict(zip(action_conditions, a_colors))


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))

    for col_idx, ax in enumerate(axes):
        method = methods[col_idx]

        if col_idx < 2:
            lora_direction = method
            for action_condition in action_conditions:
                xs = scales

                ys = [np.mean(
                    [
                    np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier][lora_direction][selected_layer_id][scale][story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names
                    ]
                    ) for scale in scales]
                yerrs = [1.96*scipy.stats.sem(
                    [
                    np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier][lora_direction][selected_layer_id][scale][story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names
                    ]
                ) for scale in scales]
                ax.plot(
                    xs, ys, marker='o', color=action2color[action_condition], ls='-',
                )
                ax.fill_between(xs, np.array(ys) - yerrs, np.array(ys) + yerrs, color=action2color[action_condition], 
                    linewidth=0, alpha=0.2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xlabel('Inference-time scaling of adapter weights (Percentage)')
            ax.set_ylabel('P("Yes")')

            ax.set_title(lora_direction.title()+' adapter')
        else:
            for action_condition in action_conditions:

                xs = coefs

                y_all_list = []
                
                for coef in coefs:
                    if coef == 0:
                        y_all = [np.mean([eval_result_all[f'lora_{dataset_name}'][model_identifier]['letter'][selected_layer_id][0][story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names]
                    else:
                        coef_str = f'{coef:.1f}'
                        y_all = [np.mean([eval_result_all[f'contrastive_{dataset_name}'][model_identifier][selected_layer_id][coef_str][story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names]
                    y_all_list.append(y_all)

                ys = [np.mean(y_all) for y_all in y_all_list]
                yerrs = [1.96*scipy.stats.sem(y_all) for y_all in y_all_list]
                ax.plot(
                    xs, ys, marker='o', color=action2color[action_condition], ls='-',
                )
                ax.fill_between(xs, np.array(ys) - yerrs, np.array(ys) + yerrs, color=action2color[action_condition], 
                    linewidth=0, alpha=0.2)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel(r'Steering coefficient $\beta$')
            ax.set_ylabel('P("Yes")')
            ax.set_title(f'Contrastive Activation Addition steering')

        if col_idx == 1:
            color_legend = ax.legend(handles=[mpatches.Patch(facecolor=action2color[action_condition], label=action_condition.title()) for action_condition in action_conditions], 
                                    loc='center', bbox_to_anchor=(0.5, -0.3), ncol=3)
            ax.add_artist(color_legend)

    plt.savefig(f'fig/{model_identifier}_method_comparison_panel_{dataset_name}_yes_prob.pdf', bbox_inches='tight', bbox_extra_artists=[color_legend])
    plt.show()