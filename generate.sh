python -m bayes_lti.cli generate --out data/dataset_long.npz --n 50 \
  --M-train 1000 --M-val 0 --M-test 0 \
  --T-min 25 --T-max 25 \
  --sigma-true 0.01 --v-true 0.5 --rho0-gen 0.95 --seed 123

python -m bayes_lti.cli select-tasks \
  --data-in data/dataset_long.npz \
  --out data/dataset_selected.npz \
  --train 100 --val 0 --test 20 \
  --k 1.0 \
  --select-strategy spread \
  --shuffle-selected \
  --edge-out data/dataset_edge_selected.npz \
  --seed 123

