from typing import Tuple
import re


def rule(x):
    # 괄호
    a = re.compile(r"\([^)]*\)")
    # 문장 부호
    b = re.compile("[^가-힣0-9 ]")
    x = re.sub(pattern=a, repl="", string= x)
    x = re.sub(pattern=b, repl="", string= x)
    return x


def load_dataset(ls_sequence_of_indices_path: str) -> Tuple[list, list]:
    ls_audio_path = list()
    ls_sequence_of_indices = list()
    with open(ls_sequence_of_indices_path, encoding="utf-8") as f:
        for line in f.readlines():
            audio_path, transcript, sequence_of_indices = line.split("\t")
            sequence_of_indices = sequence_of_indices.replace("\n", "")

            ls_audio_path.append(audio_path)
            ls_sequence_of_indices.append(sequence_of_indices)
    return ls_audio_path, ls_sequence_of_indices
