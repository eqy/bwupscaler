#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --name fusedtest --model rcan --batch-size 6 --val-batch-size 6
