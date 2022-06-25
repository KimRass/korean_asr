import argparse
import re
from pathlib import Path

from grapheme import *
from character import *
from subword import *


def get_args():
    parser = argparse.ArgumentParser(description="KsponSpeech Preprocess")
    parser.add_argument(
        "--data_dir", type=str,
    )
    parser.add_argument(
        "--save_dir", type=str
    )
    parser.add_argument(
        "--output_unit", type=str, default="character"
    )
    parser.add_argument(
        "--save_path", type=str
    )
    parser.add_argument(
        "--preprocess_mode", type=str, default="phonetic"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=5000
    )
    return parser.parse_args()


def bracket_filter(sentence, mode="phonetic"):
    new_sentence = ""
    if mode == "phonetic":
        flag = False
        for char in sentence:
            if char == "(" and flag is False:
                flag = True
                continue
            if char == "(" and flag is True:
                flag = False
                continue
            if char != ")" and flag is False:
                new_sentence += char
    elif mode == "spelling":
        flag = True
        for char in sentence:
            if char == "(":
                continue
            if char == ")":
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if char != ")" and flag is True:
                new_sentence += char
    else:
        raise ValueError(f"Unsupported mode: {preprocess_mode}")
    return new_sentence


def special_filter(sentence, preprocess_mode="phonetic", replace=None):
    SENTENCE_MARK = ["?", "!", "."]
    NOISE = ["o", "n", "u", "b", "l"]
    EXCEPT = ["/", "+", "*", "-", "@", "$", "^", "&", "[", "]", "=", ":", ";", ","]

    new_sentence = ""
    for idx, char in enumerate(sentence):
        if char not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and char in NOISE and sentence[idx + 1] == "/":
                continue
        if char == "#":
            new_sentence += "샾"
        elif char == "%":
            if preprocess_mode == "phonetic":
                new_sentence += replace
            elif preprocess_mode == "spelling":
                new_sentence += "%"
        elif char not in EXCEPT:
            new_sentence += char

    pattern = re.compile(r"\s\s+")
    new_sentence = re.sub(pattern, " ", new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, preprocess_mode, replace=""):
    return special_filter(
        bracket_filter(raw_sentence, preprocess_mode), preprocess_mode, replace
    )


def preprocess(data_dir, preprocess_mode="phonetic"):
    print("Preprocessing data...")
    
    percent_files = {
        "087797": "퍼센트",
        "215401": "퍼센트",
        "284574": "퍼센트",
        "397184": "퍼센트",
        "501006": "프로",
        "502173": "프로",
        "542363": "프로",
        "581483": "퍼센트"
    }

    ls_audio_path = list()
    ls_transcript = list()
    for file in data_dir.glob("*/*/*"):
        if file.suffix == ".txt":
            with open(file, mode="r", encoding="cp949") as f:
                sent = f.read()
                data_idx = file.name.split("_")[-1]
                if data_idx in percent_files:
                    sent = sentence_filter(sent, preprocess_mode, replace=percent_files[data_idx])
                else:
                    sent = sentence_filter(sent, preprocess_mode)
            ls_audio_path.append(str(file))
            ls_transcript.append(sent)
        else:
            continue
    return ls_audio_path, ls_transcript


def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    vocab_size = args.vocab_size
    output_unit = args.output_unit
    preprocess_mode = args.preprocess_mode

    ls_audio_path, ls_transcript = preprocess(
        data_dir=data_dir,
        preprocess_mode=preprocess_mode
    )
    if args.output_unit == "character":
        create_vocabs_csv(ls_transcript, vocab_size, save_dir)
        create_transcripts_txt(ls_audio_path, ls_transcript, save_dir)
    # elif args.output_unit == "subword":
    #     train_sentencepiece(ls_transcript, save_path, vocab_size)
    #     sentence_to_subwords(ls_audio_path, ls_transcript, save_path)
    # elif args.output_unit == "grapheme":
    #     sentence_to_grapheme(ls_audio_path, ls_transcript, vocab_dir)
    # else:
    #     raise ValueError(f"Unsupported preprocess method: {output_unit}")


if __name__ == "__main__":
    main()
