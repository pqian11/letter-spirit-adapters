import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
plt.rcParams['font.family'] = 'Arial'


behavior_types = ['compliance', 'underinclusion', 'overinclusion', 'core']
behavior_colors = ['#66A182', 'orange', 'violet', 'red']
behavior_type2color = dict(zip(behavior_types, behavior_colors))

conditions = ['spirit', 'base', 'letter']
pretty_conds = ['Spirit', 'Base', 'Letter']

meta_info_dict_list = [
    {'model_identifier':'Llama-3.1-70B-Instruct', 'thinking_mode':'no-thinking', 'legend_anchor':(0.4, 0.85)},
    {'model_identifier':'Qwen3.5-27B', 'thinking_mode':'no-thinking', 'legend_anchor':(0.4, 0.1)},
    {'model_identifier':'Qwen3.5-27B', 'thinking_mode':'thinking', 'legend_anchor':(0.4, 0.1)},
]

caselaw_df = pd.read_csv("stimuli/legal_case_dataset.csv")

for meta_info_dict in meta_info_dict_list:
    model_identifier = meta_info_dict['model_identifier']
    thinking_mode = meta_info_dict['thinking_mode']
    legend_anchor = meta_info_dict['legend_anchor']

    rs_df = pd.read_csv(f"results/model_eval/generation_scaled_lora/caselaw/{model_identifier}/caselaw_rs_df_free-form_{model_identifier}_{thinking_mode}.csv")

    rs_df['response'] = (rs_df['yes_prob'] > 0.5).astype(int)

    rs_df_summary = rs_df.groupby(['direction', 'case_id', 'behavior_type'], as_index=False).agg(
        mean_value=('response', 'mean')
    )

    # print(rs_df_summary)

    rs_df_summary["direction"] = pd.Categorical(
        rs_df_summary["direction"], categories=["spirit", "base", "letter"], ordered=True
    )

    rs_df_mean_summary = (
        rs_df_summary.groupby(["behavior_type", "direction"], as_index=False)
        .agg(
            mean_yes_prob=("mean_value", "mean"),
            sd_value=("mean_value", "std"),
            valid_count=("mean_value", "count"),
        )
    )

    rs_df_mean_summary["sem"] = rs_df_mean_summary["sd_value"] / np.sqrt(
        rs_df_mean_summary["valid_count"]
    )


    plt.figure(figsize=(2.8, 2.5))
    ax = plt.gca()

    for behavior_type in ['overinclusion', 'underinclusion']:
        xs = [-1, 0, 1]
        ys = [rs_df_mean_summary.loc[(rs_df_mean_summary['direction'] == cond) & (rs_df_mean_summary['behavior_type'] == behavior_type.title()), "mean_yes_prob"].item() for cond in conditions]
        yerrs = [1.96*rs_df_mean_summary.loc[(rs_df_mean_summary['direction'] == cond) & (rs_df_mean_summary['behavior_type'] == behavior_type.title()), "sem"].item() for cond in conditions]

        ax.errorbar(xs, ys, yerr=yerrs, color=behavior_type2color[behavior_type], capsize=5, marker='o', label=behavior_type.title())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(0, 1.02)

    ax.set_xticks(xs)
    ax.set_xticklabels(pretty_conds)

    ax.set_ylabel('Unfavorable legal decision')

    color_legend = ax.legend(handles=[Line2D([], [], marker='o', color=behavior_type2color[behavior_type], label=behavior_type.title()) for behavior_type in ['overinclusion', 'underinclusion']], 
                            loc='center left', bbox_to_anchor=legend_anchor, ncol=1, frameon=False, fontsize=8)
    ax.add_artist(color_legend)

    plt.savefig(f"fig/{model_identifier}_{thinking_mode}_caselaw_free-form.pdf", bbox_inches='tight')
    plt.show()

