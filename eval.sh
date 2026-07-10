python -m bayes_lti.cli eval \
  --data data/dataset_selected.npz \
  --methods "meta,ols,ridge,pooled_prior_ridge, shared_subspace" \
  --ckpt runs/last.ckpt \
  --support-len 19  --query-len 5\
  --report outputs/report.json \
  --auto-support --auto-support-metric auto --auto-support-val-len 5