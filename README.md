# Packages
```sh
pip install torch
pip install omegaconf
pip install Levenshtein
pip install hydra-core --upgrade
```

# Install
```sh
pip install -e .
```

# Preprocess
## Run
- Set options in "preprocess_data.sh"
```sh
bash preprocess_data.sh
```
- Output: "vocab.csv", "transcripts.txtg"


# Train
## Configurations
- "kospeech/models/__init__.py": `ModelConfig`
- "kospeech/models/deepspeech2/__init__.py": `DeepSpeech2Config`
- "kospeech/trainer/__init__.py": `TrainConfig`
- "kospeech/trainer/__init__.py": `DeepSpeech2TrainConfig`
## Run
- Set options in "/bin/train_ds2.sh"
```sh
# /bin
bash train_ds2.sh
```

# Interence
```sh
python3 bin/inference.py --model_path $model_path --audio_path $audio_path --device 'cpu'
```

audio:
  audio_extension: pcm
  transform_method: fbank
  sample_rate: 16000
  frame_length: 20
  frame_shift: 10
  n_mels: 80
  normalize: true
  del_silence: true
  feature_extract_by: kaldi
  freq_mask_para: 18
  time_mask_num: 4
  freq_mask_num: 2
  spec_augment: true
  input_reverse: false
model:
  architecture: deepspeech2
  use_bidirectional: true
  hidden_dim: 1024
  dropout: 0.3
  num_encoder_layers: 3
  rnn_type: gru
  max_len: 400
  activation: hardtanh
  teacher_forcing_ratio: 1.0
  teacher_forcing_step: 0.0
  min_teacher_forcing_ratio: 1.0
  joint_ctc_attention: false
train:
  dataset: kspon
  dataset_path: /Users/jongbeom.kim/Documents/ksponspeech/data/KsponSpeech_01
  transcripts_path: /Users/jongbeom.kim/Documents/ksponspeech/data/transcripts.txt
  output_unit: character
  num_epochs: 70
  batch_size: 32
  save_result_every: 1000
  checkpoint_every: 5000
  print_every: 10
  mode: train
  seed: 777
  resume: false
  num_workers: 4
  use_cuda: true
  optimizer: adam
  init_lr: 1.0e-06
  final_lr: 1.0e-06
  peak_lr: 0.0001
  init_lr_scale: 0.01
  final_lr_scale: 0.05
  max_grad_norm: 400
  warmup_steps: 400
  weight_decay: 1.0e-05
  reduction: mean