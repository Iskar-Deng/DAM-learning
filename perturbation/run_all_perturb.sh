#!/bin/bash
set -euo pipefail

echo "Running perturbation: baseline"
python -m perturbation.run_perturb --rule "baseline"

echo "Running perturbation: full"
python -m perturbation.run_perturb --rule "full"

RULES=("localA" "localP" "global")
FEATURES=("animacy" "definiteness" "pronominality")
DIRECTIONS=("natural" "inverse")

for rule in "${RULES[@]}"; do
  for feature in "${FEATURES[@]}"; do
    for direction in "${DIRECTIONS[@]}"; do
      echo "Running perturbation: $rule / $feature / $direction"
      python -m perturbation.run_perturb \
        --rule "$rule" \
        --feature "$feature" \
        --direction "$direction"
    done
  done
done

echo "All perturbations completed."
