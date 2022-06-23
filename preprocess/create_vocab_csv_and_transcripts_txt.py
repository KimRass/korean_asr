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
        "--vocab_dir", type=str
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
    new_sentence = str()

    if mode == "phonetic":
        flag = False

        for ch in sentence:
            if ch == "(" and flag is False:
                flag = True
                continue
            if ch == "(" and flag is True:
                flag = False
                continue
            if ch != ")" and flag is False:
                new_sentence += ch

    elif mode == "spelling":
        flag = True

        for ch in sentence:
            if ch == "(":
                continue
            if ch == ")":
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ")" and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode: {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode="phonetic", replace=None):
    SENTENCE_MARK = ["?", "!", "."]
    NOISE = ["o", "n", "u", "b", "l"]
    EXCEPT = ["/", "+", "*", "-", "@", "$", "^", "&", "[", "]", "=", ":", ";", ","]

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == "/":
                continue

        if ch == "#":
            new_sentence += "샾"

        elif ch == "%":
            if mode == "phonetic":
                new_sentence += replace
            elif mode == "spelling":
                new_sentence += "%"

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r"\s\s+")
    new_sentence = re.sub(pattern, " ", new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


def preprocess(data_dir, mode="phonetic"):
    data_dir = Path(data_dir)
    print("preprocess started..")

    ls_audio_path = list()
    ls_transcript = list()
    for file in data_dir.glob("*/*/*"):
        if file.suffix == ".txt":
            with open(file, mode="r", encoding="cp949") as f:
                sent = f.read()
            ls_audio_path.append(file)
            ls_transcript.append(sent)
        else:
            continue
    return ls_audio_path, ls_transcript


def main():
    args = get_args()
    data_dir = Path(args.data_dir)
    vocab_dir = Path(args.vocab_dir)
    save_path = Path(args.save_path)
    vocab_size = args.vocab_size
    output_unit = args.output_unit
    preprocess_mode = args.preprocess_mode

    ls_audio_path, ls_transcript = preprocess(
        data_dir=data_dir,
        mode=preprocess_mode
    )
    if args.output_unit == "character":
        create_vocab_csv(ls_transcript, vocab_dir, vocab_size)
        create_transcripts_txt(ls_audio_path, ls_transcript, vocab_dir)
    elif args.output_unit == "subword":
        train_sentencepiece(ls_transcript, save_path, vocab_size)
        sentence_to_subwords(ls_audio_path, ls_transcript, save_path)
    elif args.output_unit == "grapheme":
        sentence_to_grapheme(ls_audio_path, ls_transcript, vocab_dir)
    else:
        raise ValueError(f"Unsupported preprocess method: {output_unit}")


if __name__ == "__main__":
    main()
