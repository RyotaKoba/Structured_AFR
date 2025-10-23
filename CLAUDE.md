# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference

**重要: すべての応答は日本語で行ってください。**
- コードの説明は日本語で
- エラーメッセージの解釈も日本語で
- コメントを追加する際は日本語で記述

## Project Overview

This is a research implementation of **Structured Activation Feature-based Reweighting (AFR)** for neural network pruning in Large Language Models (LLMs). The project focuses on structured pruning of MLP layers in transformer models, particularly targeting LLaMA and Vicuna models.

## Core Architecture

### Pruning Methods

The codebase implements several pruning strategies:

1. **Structured AFR** (`structured_afr`): The main method combining Feature Orthogonality (FO) scores and SNIP scores
   - FO score: Based on singular value decomposition of activations
   - SNIP score: Based on gradient-weighted importance
   - Final score: Standardized combination of both metrics

2. **Structured ReFer Methods** (`structured_refer_svd`, `structured_refer_l1`): SVD-based and L1-based reference pruning

3. **Structured SNIP** (`structured_snip`): Gradient-based pruning using cross-entropy loss

4. **CFSP** (`cfsp`): Custom feature-based structured pruning with configurable global/local metrics

All structured methods work at the neuron level by computing scores across three MLP projection matrices (gate_proj, up_proj, down_proj) and pruning entire neurons rather than individual weights.

### Key Components

**lib/prune.py**: Core pruning logic
- `Structured_AFR()`: Main AFR implementation at lib/prune.py:1197
- `compress()`: Physically removes pruned neurons and updates model architecture at lib/prune.py:237
- `prepare_calibration_input()`: Captures intermediate activations for pruning decisions at lib/prune.py:180
- Global variables like `P_SVD_loss` and `SVD_loss` accumulate feature-based loss via forward hooks

**lib/block_metrics.py**: Layer importance calculation
- `block_influence()`: Computes layer-wise influence using angular distance, cosine similarity, MSE, or MAE

**lib/data.py**: Dataset handling
- Uses WikiText-2 dataset (loaded locally from `./data_local/wiki_all`)
- `get_loaders()`: Returns calibration samples for pruning

**lib/eval.py**: Model evaluation
- `eval_ppl()`: Computes perplexity on WikiText-2 test set

**main.py**: Entry point for pruning and evaluation

**analyzer.py**: Post-pruning analysis of FO and SNIP score distributions across layers

### Global/Local Metrics System (CFSP)

The CFSP method uses a two-level importance scoring:

**Global metrics** (layer-level importance):
- `angular`: Angular distance between input/output hidden states (default)
- `cosine`: Cosine similarity
- `mse`: Mean squared error
- `mae`: Mean absolute error

**Local metrics** (neuron-level importance):
- `three_w_one_wa`: Combines all three MLP weights with activation-weighted scoring (default)
- `wanda_base`: Weight magnitude × activation
- `mag_base`: Weight magnitude only
- `one_wa`, `one_a`: Simplified variants

Control parameters `a`, `b`, `c` adjust the balance between global and local importance.

### Weight Score Logging

The `WeightScoreLogger` class (lib/prune.py:952 and :1231) saves per-layer FO and SNIP scores to disk during structured pruning for analysis:
- `fo_weight_scores.pt`: FO scores per layer
- `snip_weight_scores.pt`: SNIP scores per layer
- `metadata.json`: Pruning configuration and timestamp

## Common Commands

### Run Pruning

```bash
bash start.sh
```

This executes structured AFR pruning with default parameters:
- Model: Meta-Llama-3-8B
- Pruning ratio: 0.5
- Method: structured_afr
- Samples: 128
- Global metric: angular
- Local metric: three_w_one_wa

### Evaluate Pruned Model

```bash
bash eval.sh
```

Loads a pruned model from disk and evaluates perplexity on WikiText-2.

### Analyze Score Distribution

```bash
python analyzer.py
```

Generates correlation analysis and statistics for FO vs SNIP scores across layers. Outputs PDFs and JSON to `afr_analysis_results/`.

### Direct Python Invocation

```bash
python main.py \
  --model meta-llama/Meta-Llama-3-8B \
  --prune_method structured_afr \
  --pruning_ratio 0.5 \
  --nsamples 128 \
  --a 1 --b 1 --c 1 \
  --cuda \
  --global_metrics angular \
  --local_metrics three_w_one_wa \
  --save_model "./pruned_model/my_pruned_model" \
  --eval
```

Key flags:
- `--cuda`: Use GPU (required for model loading)
- `--eval`: Run perplexity evaluation after pruning
- `--sample`: Show sample model outputs
- `--save_model`: Save pruned model to specified path
- `--all`: Prune attention layers in addition to MLP (for unstructured methods)
- `--cuda_friendly`: Round pruning to multiples of 128 for hardware efficiency

### SLURM Execution

```bash
sbatch analysis.sh
```

Runs analyzer in Singularity container on A6000 GPU partition.

## Model Structure Assumptions

The code is designed for LLaMA/Vicuna-style architectures with:
- `model.model.layers`: List of transformer blocks
- Each layer has `.mlp` with `gate_proj`, `up_proj`, `down_proj` (SwiGLU MLP)
- Each layer has `.self_attn` with `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Config attributes: `intermediate_size`, `hidden_size`, `use_cache`

## Important Implementation Details

### Pruning Ratio Calculation

For structured methods, pruning operates at the neuron level:
- `sorted_mlp_metric`: Scores sorted per-layer in descending order
- `thresholds[i] = sorted_mlp_metric[i][int(shape * (1 - pruning_ratio))]`
- This ensures uniform sparsity across layers (flat pruning)

For CFSP, pruning ratios are adaptive per layer based on sigmoid-transformed layer importance.

### Hook-based Feature Collection

Forward hooks accumulate global loss (`P_SVD_loss` or `SVD_loss`) during a single forward pass. This enables gradient computation w.r.t. intermediate activations without storing full activation tensors.

### Neuron Score Aggregation

`calculate_neuron_score_v2()` (lib/prune.py:1003, :1205) computes per-neuron scores from weight matrices:
- Trims top/bottom 2% of weights per neuron
- Computes mean and standard deviation
- Returns SNR (signal-to-noise ratio) or mean absolute value

### Model Compression

`compress()` function (lib/prune.py:237):
1. Selects neurons to keep using boolean mask
2. Physically removes rows from `gate_proj` and `up_proj`
3. Removes columns from `down_proj`
4. Updates `out_features` and `intermediate_size` attributes
5. Optionally adds bias compensation (see `compress_bias()` for FLAP-style pruning)

## Data Dependencies

- WikiText-2 dataset must be pre-downloaded to `./data_local/wiki_all`
- Models are cached in `./llm_weights` (configurable via `--cache_dir`)

## Output Structure

- `./pruned_model/`: Saved pruned models
- `./weight_scores_{pruning_ratio}/`: FO/SNIP score logs
- `./afr_analysis_results/`: Analysis outputs (PDFs, JSON)
- `result_analysis.txt` / `error_analysis.txt`: SLURM job outputs
