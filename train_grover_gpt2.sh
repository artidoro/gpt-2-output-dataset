#!/bin/bash

source /projects/tir5/users/apagnoni/anaconda3/bin/activate dft

export LOG_DIR=log/baseline_gpt2_xl-1542M-k40_grover_hard_combined

python baseline.py \
    data \
    $LOG_DIR \
    --source "xl-1542M-k40;generator=mega~dataset=p0.90;generator=mega~dataset=p0.92;generator=mega~dataset=p0.94;generator=mega~dataset=p0.96;generator=mega~dataset=p0.98;generator=mega~dataset=p1.00" \
    --n_train 500000 \
    --n_valid 10000 \
    --n_jobs 7 \
    --save_model \
    --save_featureizer \

cp $(realpath $0) "$LOG_DIR/script.sh"