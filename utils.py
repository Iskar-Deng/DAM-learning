# utils.py

# === Constants ===

DATA_PATH = "/path/to/data"
MODEL_PATH = "/path/to/models"
CONFIG_PATH = "/path/to/configs"
CHECKPOINT_PATH = "/path/to/checkpoints"
CACHE_PATH = "/path/to/cache"
EVALUATION_PATH = "/path/to/evaluation"

AGENT_MARK = "🄰"
PATIENT_MARK = "🄿"

SEMANTIC_HIERARCHIES = {
    "animacy": ["inanimate", "animate"],
    "definiteness": ["indef", "definite"],
    "pronominality": ["common", "pronoun"],
}

# === Perturbed Rules ===
def _rank(label: str, feature: str) -> int:
    """Return the index of label in the feature's scale; unknown label = -1."""
    if label is None:
        return -1
    try:
        return SEMANTIC_HIERARCHIES[feature].index(label)
    except ValueError:
        return -1

def should_mark_local_A(A_labels, feature, direction="natural"):
    """
    Local–A:
      natural: mark A if A is LOW (low rank)
      inverse: mark A if A is HIGH (high rank)
    """
    r = _rank(A_labels.get(feature), feature)
    if r < 0:
        return False
    if direction == "natural":
        return r == 0
    else:  # inverse
        return r == 1

def should_mark_local_P(P_labels, feature, direction="natural"):
    """
    Local–P:
      natural: mark P if P is HIGH (high rank)
      inverse: mark P if P is LOW (low rank)
    """
    r = _rank(P_labels.get(feature), feature)
    if r < 0:
        return False
    if direction == "natural":
        return r == 1
    else:
        return r == 0

def should_mark_global(A_labels, P_labels, feature, direction="natural"):
    """
    Global:
      natural: mark if A <= P
      inverse: mark if A > P
    """
    rA = _rank(A_labels.get(feature), feature)
    rP = _rank(P_labels.get(feature), feature)
    if rA < 0 or rP < 0:
        return False, False

    if direction == "natural":
        mark = (rA <= rP)
    else:  # inverse
        mark = (rA > rP)

    return mark, mark

# === Classifer Prompt ===

NP_CLASSIFICATION_PROMPTS = {
    "animacy": """Classify the animacy of the following noun phrase (NP).
Answer strictly in one word: "animate" or "inanimate".

Sentence: {sentence}
{np_type} NP: {np}

Definitions:
- animate: humans and animals (e.g., the teacher, a cat, he, they)
- inanimate: objects, places, concepts, non-living entities (e.g., the book, the school, water)

Return exactly one of: animate, inanimate.
""",

    "definiteness": """Classify the definiteness of the following noun phrase (NP).
Answer strictly in one word: "definite" or "indef".

Sentence: {sentence}
{np_type} NP: {np}

Definitions:
- definite: identifiable or unique referent (e.g., the teacher, this book, my car)
- indef: non-specific or non-unique referent (e.g., a student, some dog, any book)

Return exactly one of: definite, indef.
""",

    "pronominality": """Classify the following noun phrase (NP) by whether it is a pronoun.
Answer strictly in one word: "pronoun" or "common".

Sentence: {sentence}
{np_type} NP: {np}

Definitions:
- pronoun: personal or demonstrative pronouns (e.g., I, you, he, she, it, they, this, that)
- common: all other noun phrases including names and full NPs (e.g., John, the teacher, a dog, some water, the book on the table)

Return exactly one of: pronoun, common.
""",
}

# === Model / Training Defaults ===

MODEL_NAME = "gpt2"
SEED = 42
BLOCK_SIZE = 1024

TRAINING_ARGUMENTS_DEFAULTS = {
    "max_steps": 15000,
    "learning_rate": 3.0e-04,
    "weight_decay": 0.01,
    "warmup_steps": 0,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",

    "per_device_train_batch_size": 48,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "optim": "adamw_torch",
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": False,
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
    "max_grad_norm": 0.5,

    "logging_steps": 50,

    "eval_strategy": "steps",
    "eval_steps": 1000,

    "save_strategy": "no",
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "prediction_loss_only": True,
    "remove_unused_columns": False,
    "report_to": ["tensorboard"],
}

CHECKPOINT_FREQUENCY_DEFAULTS = [
    [500, 8000], 
    [1000, 20000],
]

RESUME_DEFAULT = False