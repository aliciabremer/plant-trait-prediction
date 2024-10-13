#!/bin/bash

python training_code.py --type 'dino_big' --lr 0.0001 --lr-all 0.0001 \
    --frozen-epochs 20 --max-epochs 0 --decay 0.01 --decay-all 0.01 \
    --dropout-fc 0.5 --dropout-t 0 \
    --batch 64 --seed 0 > lr_scheduler_dino_seed0.txt

python training_code.py --type 'swin_v2_b_batch' --lr 0.001 --lr-all 0.00001 \
    --frozen-epochs 21 --max-epochs 0 --decay 0.01 --decay-all 0.01 \
    --dropout-fc 0.2 --dropout-t 0 --lr-sched 7 \
    --batch 64 --seed 1 > swin_batchnorm_testing_seed1.txt
