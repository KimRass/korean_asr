import pandas as pd


def create_vocabs_csv(ls_transcript, vocab_size, save_dir):
    print("Creating 'vocabs.csv'...", end=" ")

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
        "idx": [0, 1, 2],
        "char": ["<pad>", "<sos>", "<eos>"],
        "freq": [0, 0, 0]
    }

    for idxx, (char, freq) in enumerate(zip(label_list, label_freq)):
        label["idx"].append(idxx + 3)
        label["char"].append(char)
        label["freq"].append(freq)

    label["idx"] = label["idx"][:vocab_size]
    label["char"] = label["char"][:vocab_size]
    label["freq"] = label["freq"][:vocab_size]

    label_df = pd.DataFrame(label)
    label_df.to_csv(
        save_dir / "vocabs.csv", encoding="utf-8", index=False
    )
    
    print("completed!")


def get_character_to_index(vocabs_csv_path):
    char2idx = dict()
    idx2char = dict()

    df_vocabs = pd.read_csv(vocabs_csv_path, encoding="utf-8")
    for _, row in df_vocabs.iterrows():
        char2idx[row["char"]] = row["idx"]
    return char2idx


def convert_transcript_to_sequence_of_indices(transcript, char2idx):
    sequence_of_indices = ""
    for char in transcript:
        if char in char2idx:
            sequence_of_indices += (str(char2idx[char]) + " ")
    return sequence_of_indices[:-1]


def create_transcripts_txt(ls_audio_path, ls_transcript, save_dir):
    print("Creating 'transcripts.txt'...", end=" ")

    char2idx = get_character_to_index(vocabs_csv_path=save_dir / "vocabs.csv")

    with open(save_dir / "transcripts.txt", mode="w") as f:
        for audio_path, transcript in zip(ls_audio_path, ls_transcript):
            sequence_of_indices = convert_transcript_to_sequence_of_indices(transcript, char2idx)
            audio_path = str(audio_path).replace("txt", "pcm")

            f.write(f"{audio_path}\t{transcript}\t{sequence_of_indices}\n")
    print("completed!")
