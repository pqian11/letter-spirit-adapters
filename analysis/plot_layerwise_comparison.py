import numpy as np
import scipy.stats
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
plt.rcParams['font.family'] = 'Arial'
import os


model_identifiers = ['Llama-3.1-70B-Instruct', 'Qwen3.5-27B']
model_identifier2layer_ids = {
    'Llama-3.1-70B-Instruct': np.arange(2, 81, 2), 
    'Qwen3.5-27B': np.arange(2, 65, 2)
}

OUTPUT_DIR = 'results/model_eval/loophole'
action_conditions = ['compliance', 'loophole', 'noncompliance']
power_relations = ['up', 'equal', 'down']

for model_identifier in model_identifiers:
    # Load base model results
    model_judgment_dict = json.load(open(os.path.join(OUTPUT_DIR, '{model_identifier}_baseline_loophole_eval_stimuli_yes_probs.json'.format(model_identifier=model_identifier))))
    story_names = list(model_judgment_dict.keys())

    baseline_action_rating_dict = {}

    for action_condition in action_conditions:
        ratings = [np.mean([model_judgment_dict[story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names]
        print('{}: {:.3f} (\u00B1{:.3f})'.format(action_condition, np.mean(ratings), 1.96*scipy.stats.sem(ratings)))

        baseline_action_rating_dict[action_condition] = dict(zip(story_names, ratings))

    layer_ids = model_identifier2layer_ids[model_identifier]

    lora_directions = ['letter', 'spirit']

    delta_rating_dict = {}
    lora_rating_dict = {}

    for lora_direction in lora_directions:
        delta_rating_dict[lora_direction] = {}
        lora_rating_dict[lora_direction] = {}
        for layer_id in layer_ids:
            delta_rating_dict[lora_direction][layer_id] = {}
            lora_rating_dict[lora_direction][layer_id] = {}
            model_judgment_dict = json.load(open(os.path.join(OUTPUT_DIR, "{}/{}_{:02d}_{}_loophole_eval_stimuli_yes_probs.json".format(
                model_identifier, model_identifier, layer_id, lora_direction))))
            for action_condition in action_conditions:
                delta_rating_dict[lora_direction][layer_id][action_condition] = {}
                lora_rating_dict[lora_direction][layer_id][action_condition] = {}
                ratings = [np.mean([model_judgment_dict[story_name][power_relation][action_condition] for power_relation in power_relations]) for story_name in story_names]
                
                delta_ratings = np.array(ratings) - np.array([baseline_action_rating_dict[action_condition][story_name] for story_name in story_names ])
                delta_rating_mean = np.mean(delta_ratings)
                delta_rating_sem = scipy.stats.sem(delta_ratings)
                delta_rating_dict[lora_direction][layer_id][action_condition]['mean'] = delta_rating_mean
                delta_rating_dict[lora_direction][layer_id][action_condition]['sem'] = delta_rating_sem
                delta_rating_dict[lora_direction][layer_id][action_condition]['delta_ratings'] = delta_ratings
                lora_rating_dict[lora_direction][layer_id][action_condition] = dict(zip(story_names, ratings))


    a_colors = ['#66A182', 'orange', "red"]
    action2color = dict(zip(action_conditions, a_colors))

    direction2style = dict(zip(lora_directions, ['-', 'dotted']))
    approach2pretty_label = dict(zip(lora_directions, ['Letter', 'Spirit']))

    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    # action_condition = 'loophole'
    for action_condition in action_conditions:
        for lora_direction in lora_directions:
            plt.errorbar(
                layer_ids, 
                [delta_rating_dict[lora_direction][layer_id][action_condition]['mean'] for layer_id in layer_ids],
                yerr=[delta_rating_dict[lora_direction][layer_id][action_condition]['sem']*1.96 for layer_id in layer_ids],
                color=action2color[action_condition],
                ls=direction2style[lora_direction],
                capsize=3,
                elinewidth=1
            )

            plt.axhline(y=0, color='lightgray', linestyle='-', linewidth=1)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel(r"$\Delta$" + ' P("Yes")')

    ax.set_title(f'Layerwise comparison of generalization on Loophole stimuli ({model_identifier})')


    color_legend = ax.legend(handles=[mpatches.Patch(facecolor=action2color[action], label=action.title()) for action in action_conditions], 
                            loc='center', bbox_to_anchor=(1.15, 0.7), ncol=1, title='Behavior')

    ax.add_artist(color_legend)

    line_legend = ax.legend(handles=[Line2D([0], [0],linestyle=direction2style[approach], color='gray', label=approach.title()) for approach in ['letter', 'spirit']], 
                            loc='center', bbox_to_anchor=(1.15, 0.3), ncol=1, title='Adapter direction')

    ax.add_artist(line_legend)

    savepath = f'fig/{model_identifier}_loophole_anchored_lora_effect_across_layers.pdf'
    plt.savefig(savepath, bbox_inches='tight', bbox_extra_artists=[color_legend, line_legend])

    plt.show()