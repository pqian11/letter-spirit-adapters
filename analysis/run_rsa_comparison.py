import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from argparse import ArgumentParser
import scipy.stats
plt.rcParams['font.family'] = 'Arial'

np.random.seed(42)
rng = np.random.default_rng(seed=42)

parser = ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="Model identifier.", type=str)
args = parser.parse_args()

MODEL_IDENTIFIER = args.model


def get_triu_values(matrix):
    # Extracts upper triangle values excluding the diagonal.
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def bootstrap_rsa_correlation_diff(matrix1, matrix2, matrix3, num_bootstraps=1000):
    n_conditions = matrix1.shape[0]
    bootstrapped_correlation_diffs = []
    
    # Calculate original empirical correlation
    orig_v1 = get_triu_values(matrix1)
    orig_v2 = get_triu_values(matrix2)
    orig_v3 = get_triu_values(matrix3)
    empirical_corr_1v2, _ = scipy.stats.spearmanr(orig_v1, orig_v2)
    empirical_corr_1v3, _ = scipy.stats.spearmanr(orig_v1, orig_v3)
    empirical_corr_diff = empirical_corr_1v2 - empirical_corr_1v3
    
    for _ in range(num_bootstraps):
        # Resample condition indices with replacement
        boot_indices = np.random.choice(n_conditions, size=n_conditions, replace=True)
        
        # Re-index both matrices simultaneously to maintain alignment
        boot_mat1 = matrix1[np.ix_(boot_indices, boot_indices)]
        boot_mat2 = matrix2[np.ix_(boot_indices, boot_indices)]
        boot_mat3 = matrix3[np.ix_(boot_indices, boot_indices)]
        
        # Extract upper triangles of the resampled matrices
        v1_boot = get_triu_values(boot_mat1)
        v2_boot = get_triu_values(boot_mat2)
        v3_boot = get_triu_values(boot_mat3)
        
        # Compute correlation
        with np.errstate(invalid='ignore'):
            corr_1v2, _ = scipy.stats.spearmanr(v1_boot, v2_boot)
            corr_1v3, _ = scipy.stats.spearmanr(v1_boot, v3_boot)

        if not np.isnan(corr_1v2) and not np.isnan(corr_1v3):
            bootstrapped_correlation_diffs.append(corr_1v2 - corr_1v3)
            
    bootstrapped_correlation_diffs = np.array(bootstrapped_correlation_diffs)
    
    # Compute 95% Confidence Interval
    ci_lower = np.percentile(bootstrapped_correlation_diffs, 2.5)
    ci_upper = np.percentile(bootstrapped_correlation_diffs, 97.5)
    
    return empirical_corr_diff, (ci_lower, ci_upper), bootstrapped_correlation_diffs




behavior_types = ['core', 'overinclusion', 'underinclusion', 'compliance']
# approach2pretty_name = {'letter': 'Textualism', 'spirit': 'Purposivism', 'base':'Base', 'both': 'Compatible with either'}
# approach2plain_name = {'letter': 'By letter', 'spirit': 'By spirit'}


if MODEL_IDENTIFIER == 'Qwen3.5-27B':
    LORA_LAYER_ID = 32
    N_LAYER = 65
elif MODEL_IDENTIFIER == 'Llama-3.1-70B-Instruct':
    LORA_LAYER_ID = 26
    N_LAYER = 81
else:
    raise NotImplementedError

print(f"{MODEL_IDENTIFIER}, adapter applied at Layer {LORA_LAYER_ID}, total number of layers: {N_LAYER}.\n")

dataset_name = 'novel_rule_stories'
HIDDEN_STATE_DIR = f'results/hidden_states/scaled_lora/{MODEL_IDENTIFIER}'

state_labels= json.load(open(f'{HIDDEN_STATE_DIR}/scaled_lora_state_labels_{dataset_name}_all.json'))

state_labels_all = state_labels + state_labels
approach_all = ['letter']*(len(state_labels)) + ['spirit']*(len(state_labels))

groups = list(set([group for group, _ in state_labels]))
group2state_vec_idx_list = {}

for group in groups:
    group2state_vec_idx_list[group] = {}
    for behavior_type in behavior_types:
        group2state_vec_idx_list[group][behavior_type] = {}

for state_idx, state_label in enumerate(state_labels_all):
    group, behavior_type = state_label
    state_set_idx = state_idx // len(state_labels)
    group2state_vec_idx_list[group][behavior_type][state_set_idx] = state_idx


behavior_approach2feature_dict = {
    'H1': {
        "compliance_letter": [-1, 1, 1],
        "underinclusion_letter": [-1, -1, 1],
        "overinclusion_letter": [1, -1, 1],
        "core_letter": [1, 1, 1],
        "compliance_spirit": [-1, 1, -1],
        "underinclusion_spirit": [1, -1, -1],
        "overinclusion_spirit": [-1, -1, -1],
        "core_spirit": [1, 1, -1],
    },
    'H2': {
        "compliance_letter": [-1, 1, 1],
        "underinclusion_letter": [-1, -1, 1],
        "overinclusion_letter": [1, -1, 1],
        "core_letter": [1, 1, 1],
        "compliance_spirit": [-1, 1, -1],
        "underinclusion_spirit": [-1, -1, -1],
        "overinclusion_spirit": [1, -1, -1],
        "core_spirit": [1, 1, -1],
    },
    'H3': {
        "compliance_letter": [-1, -1, 1],
        "underinclusion_letter": [-1, 1, 1],
        "overinclusion_letter": [1, -1, 1],
        "core_letter": [1, 1, 1],
        "compliance_spirit": [-1, -1, -1],
        "underinclusion_spirit": [-1, 1, -1],
        "overinclusion_spirit": [1, -1, -1],
        "core_spirit": [1, 1, -1],
    }
}


delta_rho_by_layer_dict = {'H1 vs. H2':{}, 'H1 vs. H3':{}}
hyp_comparisons = ['H1 vs. H2', 'H1 vs. H3']
hyps = ['H1', 'H2', 'H3']

for layer_id in np.arange(LORA_LAYER_ID, N_LAYER):
    print('Layer', layer_id)
    for hyp_comparison in hyp_comparisons:
        delta_rho_by_layer_dict[hyp_comparison][layer_id] = {}

    state_vecs_letter_lora_100 = np.load(open(f'{HIDDEN_STATE_DIR}/{MODEL_IDENTIFIER}_letter_{LORA_LAYER_ID}_scale_100_activation_layer_{layer_id}_{dataset_name}_all.npy', 'rb'))
    state_vecs_spirit_lora_100 = np.load(open(f'{HIDDEN_STATE_DIR}/{MODEL_IDENTIFIER}_spirit_{LORA_LAYER_ID}_scale_100_activation_layer_{layer_id}_{dataset_name}_all.npy', 'rb'))
    state_vecs = np.concatenate((state_vecs_letter_lora_100, state_vecs_spirit_lora_100), axis=0)

    # Run PCA on residual stream activations
    pca = PCA(n_components=5, random_state=rng.integers(low=1, high=10000))
    transformed_state_vecs = pca.fit_transform(np.array(state_vecs))

    # Construct the list of low-dimensional state vectors and feature vectors for each vignette
    pca_vecs = []

    feature_vecs_all_dict = {}
    for hyp in hyps:
        feature_vecs_all_dict[hyp] = []

    for state_set_idx, approach in enumerate(['letter', 'spirit']):
        for behavior_type in behavior_types:
            for group in groups:
                state_idx = group2state_vec_idx_list[group][behavior_type][state_set_idx]
                pca_vecs.append(transformed_state_vecs[state_idx, :3])

                for hyp in hyps:
                    feature_vecs_all_dict[hyp].append(behavior_approach2feature_dict[hyp]["{}_{}".format(behavior_type, approach)])

    # Compute RSA matrix for the low-dimensional projection of LLM internal states
    rsa_matrix_data = cosine_similarity(pca_vecs)

    # Compute RSA matrix for each hypothetical organization of the conceptual space
    rsa_matrix_hyp_dict = {}
    for hyp in hyps:
        rsa_matrix_hyp_dict[hyp] = cosine_similarity(feature_vecs_all_dict[hyp])

    for hyp_comparison in hyp_comparisons:
        hyp_A, _, hyp_B = hyp_comparison.split()
        emp_corr_diff, ci, _ = bootstrap_rsa_correlation_diff(
            rsa_matrix_data, rsa_matrix_hyp_dict[hyp_A], rsa_matrix_hyp_dict[hyp_B], num_bootstraps=5000)
        print(f"Empirical correlation diff ({hyp_comparison}): {emp_corr_diff}")
        print(f"95% Bootstrap CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

        delta_rho_by_layer_dict[hyp_comparison][layer_id] =  {
            'val': emp_corr_diff, 
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }

with open(f"fig/rsa/{MODEL_IDENTIFIER}_rsa_delta_rho_by_layer_dict.json", "w") as file:
    json.dump(delta_rho_by_layer_dict, file, indent=4)

conditions = ['H1 vs. H2', 'H1 vs. H3']
condition_colors = ['maroon', 'coral']
layer_ids = np.arange(LORA_LAYER_ID, N_LAYER)

plt.figure(figsize=(9,3))
ax = plt.gca()

for cond_idx, condition in enumerate(conditions):
    yerrs = [
            [delta_rho_by_layer_dict[condition][layer_id]['val'] - delta_rho_by_layer_dict[condition][layer_id]['ci_lower'] for layer_id in layer_ids],
            [delta_rho_by_layer_dict[condition][layer_id]['ci_upper'] - delta_rho_by_layer_dict[condition][layer_id]['val'] for layer_id in layer_ids],
             ]
    plt.errorbar(layer_ids, [delta_rho_by_layer_dict[condition][layer_id]['val'] for layer_id in layer_ids], 
                 yerr=yerrs, marker='o', ls='none', ms=5, capsize=3.5, elinewidth=1, color=condition_colors[cond_idx], label=condition)

plt.legend(frameon=False, fontsize=9)
plt.annotate('+adapter', xy=(LORA_LAYER_ID, 0.06), xytext=(LORA_LAYER_ID, 0.2), ha='center', arrowprops=dict(facecolor='black', 
                                                                            headwidth=6, headlength=6, 
                                                                            width=0.7, shrink=0.1))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_ylabel(r'$\Delta(\rho)$')
ax.set_xlabel('Layer Index')
    
plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)
plt.savefig(f'fig/{MODEL_IDENTIFIER}_rsa_analysis_delta_rho_rs_across_layers.pdf', bbox_inches='tight')
plt.show()