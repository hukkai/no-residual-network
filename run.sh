#!/bin/bash

OMP_NUM_THREADS=12 python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPU} --master_port=6606 train.py "$@"



