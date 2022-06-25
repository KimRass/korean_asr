# import os
# import random
# import warnings
# import torch
# import torch.nn as nn
# import hydra
# from hydra.core.config_store import ConfigStore
# from omegaconf import OmegaConf, DictConfig

# from kospeech.data.data_loader import split_dataset
# from kospeech.optim import Optimizer
# from kospeech.model_builder import build_model
# from kospeech.utils import (
#     check_envirionment,
#     get_optimizer,
#     get_criterion,
#     logger,
#     get_lr_scheduler,
# )
# from kospeech.vocabs import (
#     KsponSpeechVocabulary
# )
# from kospeech.data.audio import (
#     FilterBankConfig,
#     MelSpectrogramConfig,
#     MfccConfig,
#     SpectrogramConfig,
# )
# from kospeech.models import DeepSpeech2Config
# from kospeech.trainer import (
#     SupervisedTrainer,
#     DeepSpeech2TrainConfig
# )
# # from kospeech.trainer.supervised_trainer import SupervisedTrainer


# # cs = ConfigStore.instance()
# # cs.store(group="audio", name="fbank", node=FilterBankConfig, package="audio")
# # cs.store(group="audio", name="melspectrogram", node=MelSpectrogramConfig, package="audio")
# # cs.store(group="audio", name="mfcc", node=MfccConfig, package="audio")
# # cs.store(group="audio", name="spectrogram", node=SpectrogramConfig, package="audio")
# # cs.store(name="ds2_train", group="train", node=DeepSpeech2TrainConfig, package="train")
# # cs.store(name="ds2", group="model", node=DeepSpeech2Config, package="model")


# # "configs/train.yaml"
# @hydra.main(version_base=None, config_path="../configs", config_name="train")
# def main(config: DictConfig) -> None:
#     print(OmegaConf.to_yaml(config))
def main():
    transcripts_path = "/Users/jongbeom.kim/Documents/ksponspeech/data/transcripts.txt"
    # with open(transcripts_path, encoding="utf-8") as f:
    with open(transcripts_path) as f:
        for line in f.readlines():
            print(line.split("\t"))
            # audio_path, korean_transcript, transcript = line.split("\t")
            transcript = transcript.replace('\n', '')
# line

if __name__ == "__main__":
    main()
