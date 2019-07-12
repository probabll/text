"""

* batch of strings to batch of tensors
* batch of tensors to batch of strings
* a sorted data loader for RNNs

"""

import numpy as np
import torch
from text.vocabulary import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from text.vocabulary import Vocabulary


def create_batch(sentences, vocab, device, word_dropout=0.):
    """
    Converts a list of sentences to a padded batch of word ids. Returns
    an input batch, an output batch shifted by one, a sequence mask over
    the input batch, and a tensor containing the sequence length of each
    batch element.
    :param sentences: a list of sentences, each a string
    :param vocab: a Vocabulary object for this dataset
    :param device:
    :param word_dropout: rate at which we omit words from the context (input)
    :returns: a batch of padded inputs, a batch of padded outputs,
              mask, lengths
    """
    tok = np.array([[SOS_TOKEN] + sen.split() + [EOS_TOKEN]
                    for sen in sentences])
    seq_lengths = [len(sen)-1 for sen in tok]
    max_len = max(seq_lengths)
    pad_id = vocab[PAD_TOKEN]
    pad_id_input = [
        [vocab[sen[t]] if t < seq_lengths[idx] else pad_id
         for t in range(max_len)]
        for idx, sen in enumerate(tok)]

    # Replace words of the input with <unk> with p = word_dropout.
    if word_dropout > 0.:
        unk_id = vocab[UNK_TOKEN]
        word_drop = [
            [unk_id if (
                np.random.random() < word_dropout
                and t < seq_lengths[idx])
                else word_ids[t] for t in range(max_len)]
            for idx, word_ids in enumerate(pad_id_input)]
    else:
        word_drop = pad_id_input

    # The output batch is shifted by 1.
    pad_id_output = [
        [vocab[sen[t+1]] if t < seq_lengths[idx] else pad_id
         for t in range(max_len)]
        for idx, sen in enumerate(tok)]

    # Convert everything to PyTorch tensors.
    batch_input = torch.tensor(pad_id_input)
    batch_output = torch.tensor(pad_id_output)
    batch_noisy_input = torch.tensor(word_drop)
    seq_mask = (batch_input != vocab[PAD_TOKEN])
    seq_length = torch.tensor(seq_lengths)

    # Move all tensors to the given device.
    batch_input = batch_input.to(device)
    batch_output = batch_output.to(device)
    batch_noisy_input = batch_noisy_input.to(device)
    seq_mask = seq_mask.to(device)
    seq_length = seq_length.to(device)

    return batch_input, batch_output, seq_mask, seq_length, batch_noisy_input


def batch_to_sentences(tensors, vocab: Vocabulary):
    """
    Converts a batch of word ids back to sentences.
    :param tensors: [B, T] word ids
    :param vocab: a Vocabulary object for this dataset
    :param tokenizer: necessary for joining strings
    :returns: an array of strings (each a sentence).
    """
    sentences = []
    batch_size = tensors.size(0)
    for idx in range(batch_size):
        sentence = [vocab.word(t.item()) for t in tensors[idx, :]]

        # Filter out the start-of-sentence and padding tokens.
        sentence = list(
            filter(lambda t: t != PAD_TOKEN and t != SOS_TOKEN, sentence))

        # Remove the end-of-sentence token and all tokens following it.
        if EOS_TOKEN in sentence:
            sentence = sentence[:sentence.index(EOS_TOKEN)]

        joined = " ".join(sentence)
        sentences.append(joined)

    return np.array(sentences)


class SortingTextDataLoader:
    """
    A wrapper for the DataLoader class that sorts a list of sentences by their
    lengths in descending order.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.it = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        sentences = None
        for s in self.it:
            sentences = s
            break

        if sentences is None:
            self.it = iter(self.dataloader)
            raise StopIteration

        sentences = np.array(sentences)
        sort_keys = sorted(range(len(sentences)),
                           key=lambda idx: len(sentences[idx].split()),
                           reverse=True)
        sorted_sentences = sentences[sort_keys]
        return sorted_sentences
