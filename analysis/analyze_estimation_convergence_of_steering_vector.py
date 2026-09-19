import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
plt.rcParams['font.family'] = 'Arial'

np.random.seed(42)


MODEL_IDENTIFIERS = ['Llama-3.1-70B-Instruct', 'Qwen3.5-27B']
LAYER_IDS = [26, 32]

for MODEL_IDENTIFIER, LAYER_ID in zip(MODEL_IDENTIFIERS, LAYER_IDS):

    fpath = f"results/hidden_states/steering/{MODEL_IDENTIFIER}/{MODEL_IDENTIFIER}_hidden_states_layer_{LAYER_ID}.npz"

    hidden_state_dict = np.load(fpath)

    # print(hidden_state_dict)

    print(hidden_state_dict.files)


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


    def cosine_sim(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity


    state_diff_vecs = []

    # Randomly shuffle the ordering of the groups so that we can use it to simulate different random 
    # draw of a subset K out of the N examples. Then we can average across the number of random draws
    # to get an estimate of the steering vector. Then we can calculate the cosine similarities among
    # different random draws.
    n_shuffle = 10

    subset_Ks = np.linspace(1, len(groups), 20)

    cutoffs = [int(2*3*np.floor(subset_K)) for subset_K in subset_Ks]
    print(cutoffs)

    layer_ids = [LAYER_ID]
    cos_sim_to_final_vec_multiple_shuffles_across_layers = {}

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
        
    for layer_id in layer_ids:

        fpath = f"results/hidden_states/steering/{MODEL_IDENTIFIER}/{MODEL_IDENTIFIER}_hidden_states_layer_{layer_id:02}.npz"

        hidden_state_dict = np.load(fpath)

        shuffled_groups = rule_stimuli_df['group'].to_list()

        steering_vec_from_subset_multiple_shuffles = []
        cos_sim_to_final_vec_multiple_shuffles = []

        for k in range(n_shuffle):
            print(f'Layer {layer_id} shuffle {k}')
            np.random.shuffle(shuffled_groups)

            state_diff_vecs = []

            for group_idx, group in enumerate(shuffled_groups):
                for behavior_type in target_behavior_types:
                    for prompt_variant in prompt_variants:
                        Yes_idx = contrastive_state_pair_dict[group][prompt_variant][behavior_type]['Yes']
                        No_idx = contrastive_state_pair_dict[group][prompt_variant][behavior_type]['No']
                        if behavior_type == 'underinclusion':
                            state_diff_vec = hidden_state_dict['state_vecs'][No_idx] - hidden_state_dict['state_vecs'][Yes_idx]
                        else:
                            state_diff_vec = hidden_state_dict['state_vecs'][Yes_idx] - hidden_state_dict['state_vecs'][No_idx]
                        state_diff_vecs.append(state_diff_vec)

            steering_vec_from_subset_per_shuffle = [list(np.mean(state_diff_vecs[:L], axis=0)) for L in cutoffs]
            steering_vec_from_subset_multiple_shuffles.append(steering_vec_from_subset_per_shuffle)

            cos_sims = []
            for subset_K_idx in range(len(subset_Ks)):
                vec1 = steering_vec_from_subset_per_shuffle[subset_K_idx]
                vec2 = steering_vec_from_subset_per_shuffle[-1]
                # vec2 = steering_vec_from_subset_per_shuffle[subset_K_idx-1]
                cos_sims.append(cosine_sim(vec1, vec2))

            cos_sim_to_final_vec_multiple_shuffles.append(cos_sims)

        cos_sim_to_final_vec_multiple_shuffles_across_layers[layer_id] = cos_sim_to_final_vec_multiple_shuffles

    plt.figure(figsize=(4,3.2))
    ax = plt.gca()
    for layer_id in layer_ids:
        cos_sim_to_final_vec_multiple_shuffles = cos_sim_to_final_vec_multiple_shuffles_across_layers[layer_id]
        ys = np.mean(cos_sim_to_final_vec_multiple_shuffles, axis=0)
        yerrs = scipy.stats.sem(cos_sim_to_final_vec_multiple_shuffles, axis=0)

        plt.errorbar(100*subset_Ks/len(groups), ys, yerr=yerrs, color='k', alpha=0.7, label=layer_id, capsize=4)

        plt.xlabel('Percentage of contrastive pairs')
        plt.ylabel('Cosine similarity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title('Convergence of estimated direction')

    plt.savefig(f'fig/{MODEL_IDENTIFIER}_Layer_{LAYER_ID}_steering_vec_estimation_convergence.pdf', bbox_inches='tight')
    plt.show()
