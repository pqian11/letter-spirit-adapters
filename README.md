# Directing Large Language Models to Follow the Letter and Spirit of the Law

This repository accompanies a research project on targeted adaptation that pushes a Large Language Model to prioritize the spirit or letter of the law in judgment.

## Environment

Download the results and stimuli folders and place them in the root of the repository. Create the following directory structure from the root folder:
```
mkdir -p fig/rsa
```

## Visualization

Plot adaptater evaluation results on the Rule stimuli set.
```
python analysis/plot_eval_rule_rs.py
```

Plot adaptater evaluation results on the Loophole stimuli set.
```
python analysis/plot_eval_loophole_rs.py
```

Plot adaptater evaluation results on the Reddit r/MaliciousCompliance stimuli set.
```
python analysis/plot_eval_reddit_rs.py
```

Plot adaptater evaluation results on the Historical Legal Case stimuli set.
```
python analysis/plot_eval_caselaw_rs.py
```

Plot layerwise comparison of adapter generalization effect.
```
python analysis/plot_layerwise_comparison.py
```

Plot the effect of scaled adapters as well as comparison to the steering effect with Contrastive Activation Addition approach.
```
python analysis/plot_scaled_adapter_effect.py
```

## Representational similarity analyses

Visualize representational similarity matrices of projected residual streams as well as different hypothetical models of the conceptual space:
```
python analysis/plot_rsa_analysis_panel.py -m Qwen3.5-27B
python analysis/plot_rsa_analysis_panel.py -m Llama-3.1-70B-Instruct
```

Compare RSA results across different hypotheses and plot the results across layers.
```
python analysis/run_rsa_comparison.py -m Qwen3.5-27B
python analysis/run_rsa_comparison.py -m Llama-3.1-70B-Instruct
```

## Supplemental analyses and visualization
Analyze the convergence of the estimated steering vector based on Contrastive Activation Addition approach.
```
python analysis/analyze_estimation_convergence_of_steering_vector.py
```
Analyze the similarity structure of the activation difference vectors across behavior types:
```
python analysis/analyze_contrastive_activation_addition_steering_vectors.py
```

## Statistical analyses

The `r_script` folder contains the R markdown files for statistical analyses.