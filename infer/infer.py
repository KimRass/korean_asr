import argparse
from pathlib import Path
from typing_extensions import Required
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor
import os
# from tools import revise

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import DeepSpeech2


def get_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--vocabs_csv_path", required=True)
    parser.add_argument("--device", required=False, default="cpu")
    return parser.parse_args()


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = "pcm") -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type="hamming"
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


def revise(sentence):
    words = sentence[0].split()
    result = []
    for word in words:
        tmp = ''    
        for t in word:
            if not tmp:
                tmp += t
            elif tmp[-1]!= t:
                tmp += t
        if tmp == '스로':
            tmp = '스스로'
        result.append(tmp)
    return ' '.join(result)


def main():
    args = get_args()
    model_path = Path(args.model_path)
    audio_path = Path(args.audio_path)
    vocabs_csv_path = Path(args.vocabs_csv_path)
    device = args.device

    feature = parse_audio(audio_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])
    vocab = KsponSpeechVocabulary(vocabs_csv_path)

    model = torch.load(model_path, map_location=lambda storage, loc: storage).to(device)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    if isinstance(model, DeepSpeech2):
        model.device = device
        y_hats = model.recognize(feature.unsqueeze(0), input_length)

    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
    print(sentence[0])
    # print(revise(sentence))


if __name__ == "__main__":
    main()
