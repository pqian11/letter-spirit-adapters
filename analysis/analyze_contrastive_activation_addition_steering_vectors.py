import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
plt.rcParams['font.family'] = 'Arial'


def plot_steering_vec_similarity_comparison_heatmap(cos_dists, savepath=None, add_highlight=False):
    plt.imshow(cos_dists, vmin=-1, vmax=1, interpolation='none')
    ax = plt.gca()

    # Annotate the heatmap
    # Add labels on the left vertical axis
    # Annotate prompt style labels
    label_x = -10

    path_ys = [42*i for i in range(7)]
    path_xs = [label_x]*len(path_ys)

    plt.plot(path_xs, path_ys, color='k', linewidth=1, marker='_', zorder=-100)

    label_ys = [21 + 42*i for i in range(6)]
    label_texts = ['A', 'B', 'C', 'A', 'B', 'C']

    label_y_text_offset = 0.5
    radius = 6.5

    for label_idx, label_y in enumerate(label_ys):
        label_text = label_texts[label_idx]

        circle = patches.Circle((label_x, label_y), radius, color='w', fill=True, ec='black')
        ax.add_patch(circle)
        ax.text(label_x, label_y+label_y_text_offset, label_text, ha='center', va='center', fontsize=9, color='black')

    # Annotate behavior type labels
    label_x = -25
    path_ys = [42*3*i for i in range(3)]
    path_xs = [label_x]*len(path_ys)
    plt.plot(path_xs, path_ys, color='k', linewidth=1, marker='_', zorder=-100)

    label_ys = [(len(groups)*len(prompt_variants))*(0.5+i) for i in range(2)]

    label_texts = ['Underinclusion', 'Overinclusion']

    for label_idx, label_y in enumerate(label_ys):
        label_text = label_texts[label_idx]

        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12, color='black', rotation=90,
                bbox={"facecolor":"w", 'edgecolor':'w'}, zorder=-50)


    if add_highlight == False:
        # Add labels on the top horizontal axis
        # Annotate prompt style labels
        label_y = -10

        path_xs = [42*i for i in range(7)]
        path_ys = [label_y]*len(path_xs)

        plt.plot(path_xs, path_ys, color='k', linewidth=1, marker='|', zorder=-100)

        label_xs = [21 + 42*i for i in range(6)]
        label_texts = ['A', 'B', 'C', 'A', 'B', 'C']

        label_y_text_offset = 0.5
        radius = 6.5

        for label_idx, label_x in enumerate(label_xs):
            label_text = label_texts[label_idx]

            circle = patches.Circle((label_x, label_y), radius, color='w', fill=True, ec='black')
            ax.add_patch(circle)
            ax.text(label_x, label_y + label_y_text_offset, label_text, ha='center', va='center', fontsize=9, color='black')

        label_y = -25
        path_xs = [42*3*i for i in range(3)]
        path_ys = [label_y]*len(path_xs)
        plt.plot(path_xs, path_ys, color='k', linewidth=1, marker='|', zorder=-100)

        # Annotate behavior type labels
        label_xs = [(len(groups)*len(prompt_variants))*(0.5+i) for i in range(2)]

        label_texts = ['Underinclusion', 'Overinclusion']

        for label_idx, label_x in enumerate(label_xs):
            label_text = label_texts[label_idx]

            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12, color='black',
                    bbox={"facecolor":"w", 'edgecolor':'w'}, zorder=-50)
    else:
        rect = patches.Rectangle((42*3, 0), 42, 42,
                        linewidth=2,
                        edgecolor='r', 
                        facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((0, 0), 42, 42,  # (x, y) bottom-left, width, height
                        linewidth=2,
                        edgecolor='r',
                        facecolor='none')
        ax.add_patch(rect)

        plt.plot([20, 20, 20+42*3, 20+42*3], [0, -7, -7, 0], color='r', linewidth=1.5, marker='none', zorder=-100)
        plt.plot([20+42*1.5, 20+42*1.5], [-7, -14], color='r', linewidth=1.5, marker='none', zorder=-100)

        annotated_text = ax.text(0, -20, 'Contrastive activation vectors across scenarios of the same behavior type point to a converging direction, and oppose those of the other behavior type.', 
                    fontsize=10, wrap=True, ha='left',
                    )

        annotated_text._get_wrap_line_width = lambda: 240 


    color_bar = plt.colorbar(fraction=0.033, pad=0.05)
    color_bar.set_label('Cosine similarity', fontsize=11)

    ax.axis('off')

    if savepath != None:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()



MODEL_IDENTIFIERS = ['Llama-3.1-70B-Instruct', 'Qwen3.5-27B']
LAYER_IDS = [26, 32]

for MODEL_IDENTIFIER, LAYER_ID in zip(MODEL_IDENTIFIERS, LAYER_IDS):
    print(MODEL_IDENTIFIER, LAYER_ID)

    fpath = f"results/hidden_states/steering/{MODEL_IDENTIFIER}/{MODEL_IDENTIFIER}_hidden_states_layer_{LAYER_ID}.npz"

    hidden_state_dict = np.load(fpath)

    # print(hidden_state_dict.files)

    state_labels= hidden_state_dict['state_labels']

    rule_stimuli_df = pd.read_csv('stimuli/novel_stories.csv')
    groups = rule_stimuli_df['group'].to_list()


    prompt_variants = ['orig', 'narrative', 'brief']
    responses = ['Yes', 'No']
    target_behavior_types = ['underinclusion', 'overinclusion']

    contrastive_state_pair_dict = {}

    for group in groups:
        contrastive_state_pair_dict[group] = {}
        for prompt_variant in prompt_variants:
            contrastive_state_pair_dict[group][prompt_variant] = {}
            for behavior_type in target_behavior_types:
                contrastive_state_pair_dict[group][prompt_variant][behavior_type] = {}
                for response in responses:
                    contrastive_state_pair_dict[group][prompt_variant][behavior_type][response] = None

    for state_idx, state_label in enumerate(state_labels):
        group, behavior_type, response, approach, prompt_variant = state_label
        contrastive_state_pair_dict[group][prompt_variant][behavior_type][response] = state_idx

    state_diff_vecs = []

    for behavior_type in target_behavior_types:

        for prompt_variant in prompt_variants:
            for group in groups:
                Yes_idx = contrastive_state_pair_dict[group][prompt_variant][behavior_type]['Yes']
                No_idx = contrastive_state_pair_dict[group][prompt_variant][behavior_type]['No']
                if behavior_type == 'underinclusion':
                    state_diff_vec = hidden_state_dict['state_vecs'][No_idx] - hidden_state_dict['state_vecs'][Yes_idx]
                else:
                    state_diff_vec = hidden_state_dict['state_vecs'][Yes_idx] - hidden_state_dict['state_vecs'][No_idx]
                state_diff_vecs.append(state_diff_vec)

    # Compute similarity matrix
    cos_dists = cosine_similarity(state_diff_vecs)

    savepath = f'fig/{MODEL_IDENTIFIER}_layer_{LAYER_ID}_contrastive_activation_steering_vec_cosine_sim_comparison.pdf'
    plot_steering_vec_similarity_comparison_heatmap(cos_dists, savepath=savepath, add_highlight=False)

    savepath = f'fig/{MODEL_IDENTIFIER}_layer_{LAYER_ID}_contrastive_activation_steering_vec_cosine_sim_comparison_with_highlight.pdf'
    plot_steering_vec_similarity_comparison_heatmap(cos_dists, savepath=savepath, add_highlight=True)