#!/bin/bash

source /projects/tir5/users/apagnoni/anaconda3/bin/activate dft

export LOG_DIR=log/baseline

python baseline.py \
    data \
    $LOG_DIR \
    --n_jobs 5 \
    --save_model \
    --save_features \
    --save_featureizer

cp $(realpath $0) "$LOG_DIR/script.sh"