from dataclasses import dataclass
from kospeech.models import ModelConfig


@dataclass
class DeepSpeech2Config(ModelConfig):
    architecture: str = "deepspeech2"
    use_bidirectional: bool = True
    rnn_type: str = "gru"
    hidden_dim: int = 1024
    activation: str = "hardtanh"
    num_encoder_layers: int = 3
