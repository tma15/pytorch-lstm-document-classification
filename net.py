import gensim
import numpy as np
import torch
import torch.nn.functional as F


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', torch.nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        >>> from torchnlp.nn import WeightDrop
        >>> import torch
        >>>
        >>> torch.manual_seed(123)
        <torch._C.Generator object ...
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)
        >>>
        >>> input_ = torch.randn(3, 2)
        >>> hidden_state = torch.randn(3, 2)
        >>> weight_drop_gru(input_, hidden_state)
        tensor(... grad_fn=<AddBackward0>)
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, weight_names_to_drop=[], **kwargs):
        super().__init__(*args, **kwargs)
        _weight_drop(self, weight_names_to_drop, weight_dropout)


def loss_fn(model, batch):
    return F.nll_loss(
        torch.log_softmax(model(batch), dim=-1),
        batch['label']
    )


class Classifier(torch.nn.Module):
    def __init__(
        self,
        feature_vocab,
        category_vocab,
        embedding_size=128,
        embedding_path=None,
        hidden_size=256,
        num_layers=1,
        weight_dropout=0.1,
        **kwargs,
    ):
        super().__init__()

        self.feature_vocab = feature_vocab
        self.category_vocab = category_vocab

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(
            len(feature_vocab),
            embedding_size,
            padding_idx=feature_vocab.get_index('<pad>')
        )

        self.key_vector = None
        if embedding_path:
            self.key_vector = gensim.models.KeyedVectors.load(
                embedding_path,
                mmap='r',
            )

        self.num_layers = num_layers
        self.weight_names_to_drop = [
            f'weight_hh_l{i}' for i in range(self.num_layers)
        ]
        self.weight_names_to_drop += [
            f'weight_hh_l{i}_reverse' for i in range(self.num_layers)
        ]

        self.bidirectional = True
        self.lstm = WeightDropLSTM(
            input_size=embedding_size + self.key_vector.vector_size if embedding_path
                else embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            weight_dropout=weight_dropout,
            weight_names_to_drop=self.weight_names_to_drop,
        )

        self.out = torch.nn.Linear(
            8 * hidden_size,
            len(category_vocab),
            bias=True)

    def static_embeedding(self, inputs):
        bsz = len(inputs['raw_words'])
        seq_len = 0
        for batch_idx in range(len(inputs['raw_words'])):
            seq_len = max(seq_len, len(inputs['raw_words'][batch_idx]))

        embed = torch.zeros((bsz, seq_len, self.key_vector.vector_size))
        for batch_idx in range(len(inputs['raw_words'])):
            for t, word in enumerate(inputs['raw_words'][batch_idx]):
                if word in self.key_vector:
                    embed[batch_idx, t, :] = torch.from_numpy(
                        np.array(self.key_vector.get_vector(word))
                    )
        return embed

    def load_state_dict(self, state_dict):
        state_dict = super().load_state_dict(state_dict, strict=False)
        return state_dict

    def __call__(self, inputs):
        x = inputs['words']
        bsz, seq_len = x.size()

        lengths = (x != self.feature_vocab.get_index('<pad>')).sum(dim=1)

        # (bsz, seq_len, embedding_size)
        x = self.embedding(x)

        if self.key_vector:
            static_x = self.static_embeedding(inputs).to(inputs['words'].device)
            x = torch.cat([x, static_x], dim=-1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        rnn_out, (ht, ct) = self.lstm(packed)
        ht = ht.view(self.num_layers, 2, bsz, -1)

        ht = ht[-1]  # (2, bsz, hidden_size)
        ht = (
            ht.transpose(0, 1)  # (bsz, 2, hidden_size)
            .contiguous()
            .view(bsz, 2 * self.hidden_size)
        )

        # (bsz, seq_len, 2 * hidden_size)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # (bsz, 1, 2 * hidden_size)
        h_max = F.adaptive_max_pool2d(
            x.view(bsz, 1, seq_len, -1),
            (1, 2 * self.hidden_size)
        ).squeeze(1)
        h_avg = F.adaptive_avg_pool2d(
            x.view(bsz, 1, seq_len, -1),
            (1, 2 * self.hidden_size)
        ).squeeze(1)

        # (bsz, 4 * 2 * hidden_size)
        x = torch.cat([x[:, 0], ht, h_max.squeeze(1), h_avg.squeeze(1)], dim=-1)

        # (bsz, category_size)
        x = self.out(x)
        return x
