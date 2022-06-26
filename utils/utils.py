import torch
import torch.nn as nn
import logging
import platform
from omegaconf import DictConfig

from utils.optim.lr_scheduler.lr_scheduler import LearningRateScheduler
from utils.vocabs import Vocabulary
from torch import optim
from utils.optim import (
    RAdam,
    AdamP,
    Novograd,
)
# from utils.criterion import (
#     LabelSmoothedCrossEntropyLoss,
#     JointCTCCrossEntropyLoss,
#     TransducerLoss,
# )
from utils.optim.lr_scheduler import (
    TriStageLRScheduler,
    TransformerLRScheduler,
)


logger = logging.getLogger(__name__)


def check_envirionment(use_cuda: bool) -> torch.device:
    """
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return device


def get_optimizer(model: nn.Module, config: DictConfig):
    supported_optimizer = {
        'adam': optim.Adam,
        'radam': RAdam,
        'adamp': AdamP,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'novograd': Novograd,
    }
    assert config.train.optimizer.lower() in supported_optimizer.keys(), \
        f"Unsupported Optimizer: {config.train.optimizer}\n" \
        f"Supported Optimizer: {supported_optimizer.keys()}"

    # if config.model.architecture == 'conformer':
    #     return optim.Adam(
    #         model.parameters(),
    #         betas=config.train.optimizer_betas,
    #         eps=config.train.optimizer_eps,
    #         weight_decay=config.train.weight_decay,
    #     )
    return supported_optimizer[config.train.optimizer](
        model.module.parameters(),
        lr=config.train.init_lr,
        weight_decay=config.train.weight_decay,
    )


def get_criterion(config: DictConfig, vocab: Vocabulary) -> nn.Module:
    if config.model.architecture in ('deepspeech2', 'jasper'):
        criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.train.reduction, zero_infinity=True)
    else:
        criterion = LabelSmoothedCrossEntropyLoss(
            num_classes=len(vocab),
            ignore_index=vocab.pad_id,
            smoothing=config.train.label_smoothing,
            reduction=config.train.reduction,
            dim=-1,
        )

    return criterion


def get_lr_scheduler(config: DictConfig, optimizer, epoch_time_step) -> LearningRateScheduler:
    if config.train.lr_scheduler == "tri_stage_lr_scheduler":
        lr_scheduler = TriStageLRScheduler(
            optimizer=optimizer,
            init_lr=config.train.init_lr,
            peak_lr=config.train.peak_lr,
            final_lr=config.train.final_lr,
            init_lr_scale=config.train.init_lr_scale,
            final_lr_scale=config.train.final_lr_scale,
            warmup_steps=config.train.warmup_steps,
            total_steps=int(config.train.num_epochs * epoch_time_step),
        )
    elif config.train.lr_scheduler == "transformer_lr_scheduler":
        lr_scheduler = TransformerLRScheduler(
            optimizer=optimizer,
            peak_lr=config.train.peak_lr,
            final_lr=config.train.final_lr,
            final_lr_scale=config.train.final_lr_scale,
            warmup_steps=config.train.warmup_steps,
            decay_steps=config.train.decay_steps,
        )
    else:
        raise ValueError(f"Unsupported Learning Rate Scheduler: {config.train.lr_scheduler}")

    return lr_scheduler
