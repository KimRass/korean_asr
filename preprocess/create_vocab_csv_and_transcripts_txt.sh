data_dir='/Users/jongbeom.kim/Documents/ksponspeech/data'
vocab_dir='/Users/jongbeom.kim/Documents/ksponspeech/data'
save_path='/Users/jongbeom.kim/Documents/ksponspeech/data'
OUTPUT_UNIT='character'
PREPROCESS_MODE='phonetic'
VOCAB_SIZE=5000

echo "Pre-process KsponSpeech Dataset.."

python3 preprocess_data.py \
    --data_dir $data_dir \
    --vocab_dir $vocab_dir \
    --save_path $save_path \
    --output_unit $OUTPUT_UNIT \
    --preprocess_mode $PREPROCESS_MODE \
    --vocab_size $VOCAB_SIZE \
