import os
import random
import warnings
import torch
import torch.nn as nn
import hydra
# from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig

from kospeech.data.data_loader import split_dataset
from kospeech.optim import Optimizer
from kospeech.model_builder import build_model
from kospeech.utils import (
    check_envirionment,
    get_optimizer,
    get_criterion,
    logger,
    get_lr_scheduler,
)
from kospeech.vocabs import (
    KsponSpeechVocabulary
)
# from kospeech.data.audio import (
#     FilterBankConfig,
#     MelSpectrogramConfig,
#     MfccConfig,
#     SpectrogramConfig,
# )
# from kospeech.models import DeepSpeech2Config
from kospeech.trainer import SupervisedTrainer
# from kospeech.trainer import SupervisedTrainer, DeepSpeech2TrainConfig
# from kospeech.trainer.supervised_trainer import SupervisedTrainer

KSPONSPEECH_VOCAB_PATH = '../data/vocab/kspon_sentencepiece.vocab'
KSPONSPEECH_SP_MODEL_PATH = '../data/vocab/kspon_sentencepiece.model'


def train(config: DictConfig) -> nn.DataParallel:
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)

    device = check_envirionment(config.train.use_cuda)
    if hasattr(config.train, "num_threads") and int(config.train.num_threads) > 0:
        torch.set_num_threads(config.train.num_threads)

    vocab = KsponSpeechVocabulary(
        vocab_path=f"/Users/jongbeom.kim/Documents/ksponspeech/data/vocabs.csv",
        output_unit=config.train.output_unit,
    )

    if not config.train.resume:
        epoch_time_step, trainset_list, validset = split_dataset(
            config=config,
            transcripts_path=config.train.transcripts_path,
            vocab=vocab
        )
        model = build_model(config, vocab, device)

        optimizer = get_optimizer(model, config)
        lr_scheduler = get_lr_scheduler(config, optimizer, epoch_time_step)

        optimizer = Optimizer(
            optimizer, lr_scheduler, config.train.total_steps, config.train.max_grad_norm
        )
        criterion = get_criterion(config, vocab)
    else:
        trainset_list = None
        validset = None
        model = None
        optimizer = None
        epoch_time_step = None
        criterion = get_criterion(config, vocab)

    trainer = SupervisedTrainer(
        optimizer=optimizer,
        criterion=criterion,
        trainset_list=trainset_list,
        validset=validset,
        num_workers=config.train.num_workers,
        device=device,
        teacher_forcing_step=config.model.teacher_forcing_step,
        min_teacher_forcing_ratio=config.model.min_teacher_forcing_ratio,
        print_every=config.train.print_every,
        save_result_every=config.train.save_result_every,
        checkpoint_every=config.train.checkpoint_every,
        architecture=config.model.architecture,
        vocab=vocab,
        joint_ctc_attention=config.model.joint_ctc_attention,
    )
    model = trainer.train(
        model=model,
        batch_size=config.train.batch_size,
        epoch_time_step=epoch_time_step,
        num_epochs=config.train.num_epochs,
        teacher_forcing_ratio=config.model.teacher_forcing_ratio,
        resume=config.train.resume,
    )
    return model


@hydra.main(version_base=None, config_path="../configs")
def main(config: DictConfig) -> None:
    warnings.filterwarnings("ignore")

    logger.info(OmegaConf.to_yaml(config))

    last_model_checkpoint = train(config)
    torch.save(
        last_model_checkpoint, "last_model_checkpoint.pt"
    )


if __name__ == "__main__":
    main()
