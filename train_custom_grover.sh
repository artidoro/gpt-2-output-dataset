#!/bin/bash

source /projects/tir5/users/apagnoni/anaconda3/bin/activate dft

export LOG_DIR=log/custom_features_grover_only_multiprocess

python baseline.py \
    data \
    $LOG_DIR \
    --source "generator=mega~dataset=p0.94" \
    --n_jobs 25 \
    --n_train 10000 \
    --n_valid 12000 \
    --n_test 10000 \
    --save_model \
    --save_features \
    --save_featureizer \
    --custom_features_only \

cp $(realpath $0) "$LOG_DIR/script.sh"