import torch
import torch.nn as nn
from omegaconf import DictConfig
from astropy.modeling import ParameterError

from utils.vocabs import Vocabulary
from train.ensemble import (
    BasicEnsemble,
    WeightedEnsemble,
)
from utils.models import DeepSpeech2


def build_model(
        config: DictConfig,
        vocab: Vocabulary,
        device: torch.device,
) -> nn.DataParallel:
    """ Various model dispatcher function. """
    if config.audio.transform_method.lower() == 'spect':
        if config.audio.feature_extract_by == 'kaldi':
            input_size = 257
        else:
            input_size = (config.audio.frame_length << 3) + 1
    else:
        input_size = config.audio.n_mels

    if config.model.architecture.lower() == 'deepspeech2':
        model = build_deepspeech2(
            input_size=input_size,
            num_classes=len(vocab),
            rnn_type=config.model.rnn_type,
            num_rnn_layers=config.model.num_encoder_layers,
            rnn_hidden_dim=config.model.hidden_dim,
            dropout_p=config.model.dropout,
            bidirectional=config.model.use_bidirectional,
            activation=config.model.activation,
            device=device,
        )
    elif config.model.architecture.lower() == 'las':
        model = build_las(input_size, config, vocab, device)
    else:
        raise ValueError('Unsupported model: {0}'.format(config.model.architecture))

    print(model)

    return model


def build_deepspeech2(
        input_size: int,
        num_classes: int,
        rnn_type: str,
        num_rnn_layers: int,
        rnn_hidden_dim: int,
        dropout_p: float,
        bidirectional: bool,
        activation: str,
        device: torch.device,
) -> nn.DataParallel:
    if dropout_p < 0.0:
        raise ParameterError("dropout probability should be positive")
    if input_size < 0:
        raise ParameterError("input_size should be greater than 0")
    if rnn_hidden_dim < 0:
        raise ParameterError("hidden_dim should be greater than 0")
    if num_rnn_layers < 0:
        raise ParameterError("num_layers should be greater than 0")
    # if rnn_type.lower() not in EncoderRNN.supported_rnns.keys():
    #     raise ParameterError("Unsupported RNN Cell: {0}".format(rnn_type))

    return nn.DataParallel(DeepSpeech2(
        input_dim=input_size,
        num_classes=num_classes,
        rnn_type=rnn_type,
        num_rnn_layers=num_rnn_layers,
        rnn_hidden_dim=rnn_hidden_dim,
        dropout_p=dropout_p,
        bidirectional=bidirectional,
        activation=activation,
        device=device,
    )).to(device)


# def build_las(
#         input_size: int,
#         config: DictConfig,
#         vocab: Vocabulary,
#         device: torch.device,
# ) -> nn.DataParallel:
#     model = ListenAttendSpell(
#         input_dim=input_size,
#         num_classes=len(vocab),
#         encoder_hidden_state_dim=config.model.hidden_dim,
#         decoder_hidden_state_dim=config.model.hidden_dim << (1 if config.model.use_bidirectional else 0),
#         num_encoder_layers=config.model.num_encoder_layers,
#         num_decoder_layers=config.model.num_decoder_layers,
#         bidirectional=config.model.use_bidirectional,
#         extractor=config.model.extractor,
#         activation=config.model.activation,
#         rnn_type=config.model.rnn_type,
#         max_length=config.model.max_len,
#         pad_id=vocab.pad_id,
#         sos_id=vocab.sos_id,
#         eos_id=vocab.eos_id,
#         attn_mechanism=config.model.attn_mechanism,
#         num_heads=config.model.num_heads,
#         encoder_dropout_p=config.model.dropout,
#         decoder_dropout_p=config.model.dropout,
#         joint_ctc_attention=config.model.joint_ctc_attention,
#     )
#     model.flatten_parameters()
#     return nn.DataParallel(model).to(device)


# "evaluate_ds2.py"로 옮김.
# def load_test_model(config: DictConfig, device: torch.device):
#     model = torch.load(
#         config.model_path, map_location=lambda storage, loc: storage
#     ).to(device)

#     if isinstance(model, nn.DataParallel):
#         model.module.decoder.device = device
#         model.module.encoder.device = device
#     else:
#         model.encoder.device = device
#         model.decoder.device = device
#     return model


# 미사용
# def load_language_model(path: str, device: torch.device):
#     model = torch.load(path, map_location=lambda storage, loc: storage).to(device)

#     if isinstance(model, nn.DataParallel):
#         model = model.module

#     model.device = device
#     return model


# 미사용
# def build_ensemble(model_paths: list, method: str, device: torch.device):
#     models = list()

#     for model_path in model_paths:
#         models.append(torch.load(model_path, map_location=lambda storage, loc: storage))

#     if method == 'basic':
#         ensemble = BasicEnsemble(models).to(device)
#     elif method == 'weight':
#         ensemble = WeightedEnsemble(models).to(device)
#     else:
#         raise ValueError("Unsupported ensemble method : {0}".format(method))
#     return ensemble
