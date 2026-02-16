# DAM Learning

## Task Summary
This pipeline builds syntactically parsed and semantically labeled corpora, applies Differential Argument Marking (DAM) perturbations, and includes evaluation benchmarks for controlled experiments on how language models learn DAM.

---

## Setup

### 1. Create the environment
``` bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m benepar.download benepar_en3
```

### 2. Configure Paths

Edit paths in `utils.py`:

- `DATA_PATH` -- dataset directory
- `MODEL_PATH` -- trained models
- `CONFIG_PATH` -- model configuration files
- `CHECKPOINT_PATH` -- training checkpoints
- `CACHE_PATH` -- cached preprocessing artifacts
- `EVALUATION_PATH` -- generated evaluation files

These paths specify where intermediate artifacts from the pipeline will be stored.

### 3. Datasets

We use the English side of OpenSubtitles (OPUS release), available at:

- https://opus.nlpl.eu/datasets/OpenSubtitles

Download the English corpus and place a single `.txt` file under `<DATA_PATH>`/raw.

You may also use any English `.txt` corpus instead. Make sure there is **only one** `.txt` file under `<DATA_PATH>/raw`.

---

## Pipeline

### 1. Preprocessing

```bash
# Split into train / valid / test
python -m data_processing.split_corpus

# Filter sentences (default: 3–30 tokens)
python -m data_processing.filter_sentences

# Constituency parsing (spaCy + benepar)
python -m data_processing.parse

# Extract verb–argument structures
python -m data_processing.extract_verb
```

### 2. Train Label Classifier

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Then:
```bash
# Generate label data
python -m classifiers.generate_data --task animacy
# also: definiteness / pronominality

# Train classifier
python -m classifiers.train_classifier --task animacy
```

### 3. Generate Perturbed Data

```bash
# Prelabel NP features using the trained classifiers
python -m perturbation.prelabel

# Run a single condition
python -m perturbation.run_perturb --rule localA --feature animacy --direction natural

# Or run all conditions
bash perturbation/run_all_perturb.sh
```

### 4. Train Model

``` bash
# Generate a YAML config for a perturbed condition
python -m training.generate_config --input_dir <DATA_PATH>/perturbed/<condition_dir>

# Train GPT-2-small from scratch
python training/train_lm_from_scratch.py --config <CONFIG_PATH>/<run_id>.yaml
```

### 5. Evaluation

#### 5.1 Rule Mastery

``` bash
# Build rule-mastery minimal pairs:
python -m evaluation.build_rule_mastery_minpairs   --run_id localA_animacy_natural

# Evaluate a checkpoint:
python -m evaluation.eval_minpairs_acc   --checkpoint <CHECKPOINT_PATH>/<run_id>/checkpoint-XXXX   --minpairs <EVALUATION_PATH>/rule_mastery/minimal_pairs/<run_id>/test_minimal_pairs.jsonl
```

#### 5.2 Marker Placement Robustness

``` bash
# Build marker-placement minimal pairs:
python -m evaluation.build_marker_placement_minpairs   --run_id localA_animacy_natural

# Evaluate a checkpoint:
python -m evaluation.eval_minpairs_acc   --checkpoint <CHECKPOINT_PATH>/<run_id>/checkpoint-XXXX   --minpairs <EVALUATION_PATH>/marker_placement/minimal_pairs/<run_id>/test_marker_position_minpairs.jsonl
```

#### 5.3 Semantic Probing

``` bash
# Build probing datasets:
python -m evaluation.build_probe_heads --split test

# Run probing on a checkpoint:
python -m evaluation.run_probing   --ckpt <CHECKPOINT_PATH>/<run_id>/checkpoint-XXXX   --data <EVALUATION_PATH>/probing/data/probe_subject_heads_test.jsonl   --out <EVALUATION_PATH>/probing/results.json
```

#### 5.4 BLiMP Generalization

We use the BLiMP benchmark, available from its official repository:
https://github.com/alexwarstadt/blimp

``` bash
# Perturb a BLiMP dataset:
python -m evaluation.perturb_blimp   --run_id localA_animacy_natural   --blimp_in <path/to/blimp.jsonl>   --blimp_out <path/to/perturbed.jsonl>

# Evaluate a checkpoint:
python -m evaluation.eval_minpairs_acc   --checkpoint <CHECKPOINT_PATH>/<run_id>/checkpoint-XXXX   --minpairs <path/to/perturbed.jsonl>
```

