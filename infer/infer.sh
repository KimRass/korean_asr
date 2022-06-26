model_path='/Users/jongbeom.kim/Documents/ksponspeech/model_ds2.pt'
audio_path='/Users/jongbeom.kim/Documents/ksponspeech/data/KsponSpeech_01/KsponSpeech_0109/KsponSpeech_108999.pcm'
vocabs_csv_path='/Users/jongbeom.kim/Documents/ksponspeech/aihub_character_vocabs.csv'
# $device

python3 inference.py \
    --model_path $model_path \
    --audio_path $audio_path \
    --vocabs_csv_path $vocabs_csv_path
    # --device $device