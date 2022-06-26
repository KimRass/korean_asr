# import os
import hydra
import warnings
import torch
import torch.nn as nn
# from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from kospeech.data.audio import FilterBankConfig
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.label_loader import load_dataset
from kospeech.data.data_loader import SpectrogramDataset
from evaluate.evaluator import Evaluator
from utils.utils import check_envirionment, logger
# from configs.train.model_builder import load_test_model


def load_test_model(config: DictConfig, device: torch.device):
    model = torch.load(
        config.model_path, map_location=lambda storage, loc: storage
    ).to(device)

    if isinstance(model, nn.DataParallel):
        model.module.decoder.device = device
        model.module.encoder.device = device
    else:
        model.encoder.device = device
        model.decoder.device = device
    return model


def infer(config: DictConfig):
    device = check_envirionment(config.eval.use_cuda)
    model = load_test_model(config.eval, device)

    if config.eval.dataset == "kspon":
        vocab = KsponSpeechVocabulary(
            f"../../../data/vocab/cssiri_{config.eval.output_unit}_vocabs.csv", output_unit=config.eval.output_unit
        )
    else:
        raise ValueError("Unsupported Dataset : {0}".format(config.eval.dataset))

    audio_paths, transcripts = load_dataset(config.eval.transcripts_path)

    testset = SpectrogramDataset(audio_paths=audio_paths, transcripts=transcripts,
                                 sos_id=vocab.sos_id, eos_id=vocab.eos_id,
                                 dataset_path=config.eval.dataset_path,  config=config, spec_augment=False)

    evaluator = Evaluator(
        dataset=testset,
        vocab=vocab,
        batch_size=config.eval.batch_size,
        device=device,
        num_workers=config.eval.num_workers,
        print_every=config.eval.print_every,
        decode=config.eval.decode,
        beam_size=config.eval.k,
    )
    evaluator.evaluate(model)


# cs = ConfigStore.instance()
# cs.store(name="evalaute", group="evalaute", node=EvalConfig, package="eval")
# cs.store(name="fbank", group="audio", node=FilterBankConfig, package="audio")


@hydra.main(config_path="../configs")
def main(config: DictConfig) -> None:
    warnings.filterwarnings("ignore")

    logger.info(OmegaConf.to_yaml(config))
    infer(config)


if __name__ == "__main__":
    main()
