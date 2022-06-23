# Packages
```sh
pip install torch
pip install omegaconf
pip install Levenshtein
pip install hydra-core --upgrade
```

# Data Preparation
- "configs/audio/fbank.yaml"
    ```yaml
    audio_extension: pcm
    transform_method: fbank
    sample_rate: 16000
    frame_length: 20
    frame_shift: 10
    n_mels: 80
    normalize: True
    del_silence: True
    feature_extract_by: kaldi
    freq_mask_para: 18
    time_mask_num: 4
    freq_mask_num: 2
    spec_augment: True
    input_reverse: false
    ```
- "bin/kospeech/data/audio/parser.py"
    ```py
    def __init__(
        ...
        audio_extension: str = 'pcm'
    )
    ```

# Install
```sh
pip install -e .
```

# Preprocess
- Set options in "preprocess_data.sh"
```sh
bash preprocess_data.sh
```
- Output: vocab.csv


# Train
- Set `data_dir` in "/bin/train_ds2.sh" then
```sh
# /bin
bash train_ds2.sh
```

# Interence
```sh
python3 bin/inference.py --model_path $model_path --audio_path $audio_path --device 'cpu'
```