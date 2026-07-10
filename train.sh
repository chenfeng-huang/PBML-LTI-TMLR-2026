python -m bayes_lti.cli train \
  --data data/dataset_selected.npz \
  --steps 6000 --batch 32 --lr 1e-3 \
  --tauW 5.0 --lambdaV 1e-2 --gamma 1.0 --eta 1.0 \
  --rho0 0.98 --device cpu --seed 1

