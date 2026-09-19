import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib.path as mpath
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


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_triu_values(matrix):
    """Extracts upper triangle values excluding the diagonal."""
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


width = 1.0
height = 0.33
vertices = np.array([
    [-width/2, -height/2],  # Bottom left
    [ width/2, -height/2],  # Bottom right
    [ width/2,  height/2],  # Top right
    [-width/2,  height/2],  # Top left
    [-width/2, -height/2]   # Close the path
])

rect_marker = mpath.Path(vertices, closed=True)

behavior_types = ['core', 'overinclusion', 'underinclusion', 'compliance']
behavior_type_shapes = ['X', 'o', '^', rect_marker]
behavior_type2shape = dict(zip(behavior_types, behavior_type_shapes))
approach2color = {'letter': 'navy', 'spirit':'orangered', 'base': 'gray'} # Both is used for core and compliance
approach2pretty_name = {'letter': 'Textualism', 'spirit': 'Purposivism', 'base':'Base', 'both': 'Compatible with either'}
approach2plain_name = {'letter': 'By letter', 'spirit': 'By spirit'}


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

hyps = ['H1', 'H2', 'H3']


if MODEL_IDENTIFIER == 'Qwen3.5-27B':
    LORA_LAYER_ID = 32
    N_LAYER = 65
elif MODEL_IDENTIFIER == 'Llama-3.1-70B-Instruct':
    LORA_LAYER_ID = 26
    N_LAYER = 81
else:
    raise NotImplementedError


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


for layer_id in np.arange(LORA_LAYER_ID, N_LAYER):
    state_vecs_letter_lora_100 = np.load(open(f'{HIDDEN_STATE_DIR}/{MODEL_IDENTIFIER}_letter_{LORA_LAYER_ID}_scale_100_activation_layer_{layer_id}_{dataset_name}_all.npy', 'rb'))
    state_vecs_spirit_lora_100 = np.load(open(f'{HIDDEN_STATE_DIR}/{MODEL_IDENTIFIER}_spirit_{LORA_LAYER_ID}_scale_100_activation_layer_{layer_id}_{dataset_name}_all.npy', 'rb'))

    state_vecs = np.concatenate((state_vecs_letter_lora_100, state_vecs_spirit_lora_100), axis=0)

    pca = PCA(n_components=5, random_state=rng.integers(low=1, high=10000))
    transformed_state_vecs = pca.fit_transform(np.array(state_vecs))

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

    fig = plt.figure(figsize=(12, 6))

    axes = []

    for m in range(2):
        axes.append([])
        for j in range(4):
            if m == 0:
                if j == 0:
                    axes[m].append(None)
                    continue
                ax = fig.add_subplot(2, 4, 1+j+m*4, projection='3d')
            else:
                ax = fig.add_subplot(2, 4, 1+j+m*4)
            axes[m].append(ax)

    axes = np.array(axes)


    for j in range(4):
        ax = axes[0, j]

        if j == 0:
            # ax.set_axis_off()
            continue

        ax.view_init(elev=20, azim=-40)
        
        behavior_type2linecolor = {'compliance': lighten_color('k', 0.8), 'underinclusion': lighten_color('k', 0.8), 
                                    'overinclusion':lighten_color('k', 0.8), 'core':lighten_color('k', 0.8)}

        behavior_approach2coords = {}

        for behavior_type in ['compliance', 'underinclusion', 'overinclusion', 'core']:
            xs = []
            ys = []
            zs = []

            for approach in ['letter', 'spirit']:
                if j >= 1:
                    hyp = hyps[j-1]
                    x, y, z = behavior_approach2feature_dict[hyp][f"{behavior_type}_{approach}"]

                xs.append(x)
                ys.append(y)
                zs.append(z)

                marker_color = lighten_color(approach2color[approach], 1)

                ax.plot(x, y, z, color=marker_color, marker=behavior_type2shape[behavior_type], ms=8)

                behavior_approach2coords[behavior_type+'_'+approach] = [x, y, z]

            linecolor = behavior_type2linecolor[behavior_type]

            ax.plot(xs, ys, zs, '-', color='lightblue', lw=4, zorder=-90)

        solid_edge_vertex_pairs = [
            ([-1, -1, -1], [-1, -1, 1]), ([-1, -1, -1], [1, -1, -1]), ([1, -1, -1], [1, -1, 1]), ([1, -1, -1], [1, 1, -1]),
            ([1, 1, -1], [1, 1, 1]), ([-1, -1, 1], [-1, 1, 1]),  ([-1, -1, 1], [1, -1, 1]),
            ([1, 1, 1], [-1, 1, 1]),  ([1, 1, 1], [1, -1, 1]),
        ]

        for vertex1, vertex2 in solid_edge_vertex_pairs:
            xs = [vertex1[0], vertex2[0]]
            ys = [vertex1[1], vertex2[1]]
            zs = [vertex1[2], vertex2[2]]
            ax.plot(xs, ys, zs, '-', color='k', lw=1.2, zorder=-50)

        hidden_edge_vertex_pairs = [
            ([-1, 1, -1], [-1, -1, -1]), ([-1, 1, -1], [1, 1, -1]), ([-1, 1, -1], [-1, 1, 1])
        ]

        for vertex1, vertex2 in hidden_edge_vertex_pairs:
            xs = [vertex1[0], vertex2[0]]
            ys = [vertex1[1], vertex2[1]]
            zs = [vertex1[2], vertex2[2]]
            ax.plot(xs, ys, zs, ':', color='k', lw=1.2, zorder=-50)

        ax.set_facecolor('white') # Sets the main axes background
        fig.patch.set_facecolor('white') # Sets the outer figure background

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.xaxis.set_tick_params(colors='w')
        ax.yaxis.set_tick_params(colors='w')
        ax.zaxis.set_tick_params(colors='w')

        if j == 2:
            behavior_type_legend_handles = [mlines.Line2D([], [], mec='k', mfc='none', ls='none', marker=behavior_type2shape[behavior_type], label=behavior_type.title()) for behavior_type in behavior_types]
            behavior_type_legend = ax.legend(handles=behavior_type_legend_handles, loc='center', ncol=4, bbox_to_anchor=(-0.7, 1.15), title='Behavior type')

            approach_legend_handles = [Patch(color=approach2color[approach], label=approach2plain_name[approach]) for approach in ['letter', 'spirit']]
            approach_legend = ax.legend(handles=approach_legend_handles, loc='center', ncol=2, bbox_to_anchor=(1.35, 1.15), title='Judgment approach (Adapter direction)')

            ax.add_artist(behavior_type_legend)
            ax.add_artist(approach_legend)

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_zlim(-1.6, 1.6)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()

        ax.set_box_aspect([1, 1, 1])

        ax.set_title(f"Hypothesis {j}", loc="center", pad=0, x=0.5, y=0.95)

    # Plot the RSA of model hidden states after PCA
    rsa_matrix_data = cosine_similarity(pca_vecs)
    axes[1, 0].imshow(rsa_matrix_data, vmin=-1, vmax=1, interpolation='none')
    ax = axes[1, 0]
    ax.set_title(f'{MODEL_IDENTIFIER}\n(Layer {layer_id})', x=0.5, y=1.1)

    ax.set_xticks([])
    ax.set_yticks([])

    n_group = 42

    approaches = ['letter', 'spirit']

    # Annotate Representational Similarity Matrix with symbols to mark
    # the clustering structure of the items in the matrix
    for k in range(2):
        for j in range(4):
            # Plot marker symbol on the x-axis
            ax.plot(
                n_group*(0.5 + j + 4*k),
                1.06,
                marker=behavior_type2shape[behavior_types[j]],
                color=approach2color[approaches[k]],
                markersize=7,
                clip_on=False,
                zorder=100,
                transform=ax.get_xaxis_transform(),
            )

            # Plot marker symbol on the y-axis
            ax.plot(
                -0.06,
                n_group*(0.5 + j + 4*k),
                marker=behavior_type2shape[behavior_types[j]],
                color=approach2color[approaches[k]],
                markersize=7,
                clip_on=False,
                zorder=100,
                transform=ax.get_yaxis_transform(),
        )

    triu_indices = np.triu_indices_from(rsa_matrix_data, k=1)
    vector_data = rsa_matrix_data[triu_indices]

    for hyp_idx, hyp in enumerate(hyps):
        rsa_matrix_hyp = cosine_similarity(feature_vecs_all_dict[hyp])
        ax = axes[1, hyp_idx + 1]
        im = ax.imshow(rsa_matrix_hyp, vmin=-1, vmax=1, interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        vector_hyp = rsa_matrix_hyp[triu_indices]
        spearman_r, p_value = scipy.stats.spearmanr(vector_data, vector_hyp)
        ax.set_title(r"$\rho={:.3f}$".format(spearman_r), fontsize=10)

    cbar_ax = fig.add_axes([0.915, 0.19, 0.01, 0.31])
    color_bar = plt.colorbar(im, cax=cbar_ax)
    color_bar.set_label('Cosine similarity', fontsize=11)

    plt.subplots_adjust(hspace=-0.3)

    plt.savefig(f'fig/rsa/{MODEL_IDENTIFIER}_{LORA_LAYER_ID}_rsa_layer_{layer_id}.pdf', bbox_inches='tight', bbox_extra_artists=[behavior_type_legend, approach_legend], pad_inches=0.3)
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
