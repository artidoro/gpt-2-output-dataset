#!/bin/bash

source /projects/tir5/users/apagnoni/anaconda3/bin/activate dft

export LOG_DIR=log/baseline_grover_gpt2_model_featurizer

python baseline_grover.py \
    data \
    $LOG_DIR \
    --source "generator=mega~dataset=p0.94" \
    --n_jobs 6 \
    --load_model "log/baseline" \
    --load_featureizer "log/baseline" \
    --save_model \
    --save_features \
    --save_featureizer \

cp $(realpath $0) "$LOG_DIR/script.sh"