#!/bin/bash

data_dir=

python ./bin/main.py \
    model=ds2 \
    train=ds2_train \
    train.dataset_path=$data_dir