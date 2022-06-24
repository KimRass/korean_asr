#!/bin/bash

data_dir='/Users/jongbeom.kim/Documents/ksponspeech/data/KsponSpeech_01'

python3 train_ds2.py \
    model=ds2 \
    train=ds2_train \
    train.dataset_path=$data_dir