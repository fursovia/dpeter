from allennlp.common import FromParams
from allennlp.nn.util import weighted_sum
import torch


class SpatialAttention(torch.nn.Module, FromParams):
    """
    Attention-based Extraction of Structured Information from Street View Imagery
    """

    def __init__(self, num_channels: int, embedding_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self._num_channels = num_channels
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim

        self._vector = torch.nn.Parameter(torch.Tensor(self._hidden_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self._vector)
        self._vector = self._vector.flatten()
        self._wf = torch.nn.Linear(self._num_channels, self._hidden_dim)
        self._ws = torch.nn.Linear(self._embedding_dim, self._hidden_dim)

    def forward(self, embeddings: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # embeddings are of shape (batch_size, num_tokens, emb_dim)
        # features are of shape (batch_size, dim1, dim2, num_channels)
        batch_size, num_tokens = embeddings.size(0), embeddings.size(1)

        wff = self._wf(features).unsqueeze(1)
        wss = self._ws(embeddings).unsqueeze(2).unsqueeze(2)

        out = torch.tanh(wff + wss)
        attention_values = torch.matmul(out, self._vector)

        attention_values = attention_values.reshape(batch_size, num_tokens, -1)
        attention_weights = torch.softmax(attention_values, dim=-1)

        return attention_weights

    def get_attention_features(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        # (batch_size, num_tokens, self._num_channels)
        return weighted_sum(features.reshape(batch_size, -1, self._num_channels), weights)
