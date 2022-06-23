import os
import pandas as pd

# id/char 사전 만들기
def load_vocab_csv(vocab_csv_path):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(vocab_csv_path, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"]

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


# 문장을 id로
def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        # 사전에 없는 경우 넘어가라 -> 그냥 묵음처리나 마찬가지.
        except KeyError as e:
            continue

    return target[:-1]


def create_vocab_csv(ls_transcript, vocab_dir, vocab_size):
    print("Creating 'vocab.csv'...", end=" ")

    label_list = list()
    label_freq = list()
    for transcript in ls_transcript:
        for char in transcript:
            if char not in label_list:
                label_list.append(char)
                label_freq.append(1)
            else:
                label_freq[label_list.index(char)] += 1

    label_freq, label_list = zip(
        *sorted(zip(label_freq, label_list), reverse=True)
    )
    label = {
        'id': [0, 1, 2],
        'char': ['<pad>', '<sos>', '<eos>'],
        'freq': [0, 0, 0]
    }

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    label['id'] = label['id'][:vocab_size]
    label['char'] = label['char'][:vocab_size]
    label['freq'] = label['freq'][:vocab_size]

    label_df = pd.DataFrame(label)
    label_df.to_csv(
        vocab_dir / "vocab.csv", encoding="utf-8", index=False
    )
    
    print("completed!")


def create_transcripts_txt(ls_audio_path, ls_transcript, vocab_dir):
    print("Creating 'transcripts.txt'...", end=" ")

    char2id, id2char = load_vocab_csv(vocab_csv_path=vocab_dir / "vocab.csv")

    with open(vocab_dir / "transcripts.txt", mode="w") as f:
        for audio_path, transcript in zip(ls_audio_path, ls_transcript):
            char_id_transcript = sentence_to_target(transcript, char2id)
            audio_path = str(audio_path).replace("txt", "wav")

            f.write(f"{audio_path}\t{transcript}\t{char_id_transcript}\n")
    print("completed!")
