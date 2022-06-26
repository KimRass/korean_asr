# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from kospeech.trainer.supervised_trainer import SupervisedTrainer
from pathlib import Path


@dataclass
class TrainConfig:
    dataset: str = "kspon"
    dataset_path: str = "???"
    transcripts_path: str = "/Users/jongbeom.kim/Documents/ksponspeech/data/transcripts.txt"
    output_unit: str = "character"

    batch_size: int = 32
    save_result_every: int = 1000
    checkpoint_every: int = 5000
    print_every: int = 10
    mode: str = "train"

    num_workers: int = 4
    use_cuda: bool = True
    num_threads: int = 2

    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.05
    max_grad_norm: int = 400
    weight_decay: float = 1e-05
    total_steps: int = 200000

    seed: int = 777
    resume: bool = False


@dataclass
class DeepSpeech2TrainConfig(TrainConfig):
    optimizer: str = "adam"
    init_lr: float = 1e-06
    final_lr: float = 1e-06
    peak_lr: float = 1e-04
    warmup_steps: int = 1000
    num_epochs: int = 70
    reduction: str = "mean"
    lr_scheduler: str = 'tri_stage_lr_scheduler'
