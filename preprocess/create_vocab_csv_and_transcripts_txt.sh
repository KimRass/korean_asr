data_dir='/Users/jongbeom.kim/Documents/ksponspeech/data'
save_dir='/Users/jongbeom.kim/Documents/ksponspeech/data'
OUTPUT_UNIT='character'
PREPROCESS_MODE='phonetic'
VOCAB_SIZE=5000

python3 create_vocab_csv_and_transcripts_txt.py \
    --data_dir $data_dir \
    --save_dir $save_dir \
    --output_unit $OUTPUT_UNIT \
    --preprocess_mode $PREPROCESS_MODE \
    --vocab_size $VOCAB_SIZE \
