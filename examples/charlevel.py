import os
import sys

import torch
from torch.utils.data import DataLoader

from text.datasets import readlines
from text.recipes import MonolingualCharLevel
from text.batching import SortingTextDataLoader, create_batch, batch_to_sentences


src = "de"
bilingual_prefix = "data/training"
monolingual_prefix = "data/comparable"
output_prefix = "output/text.pt.corpus"
max_length = -1
batch_size = 5
shuffle = True

try:
    os.mkdir("output")
except:
    pass

corpus_builder = MonolingualCharLevel(
    lang_code=src,  # irrelevant for now
    normalize_blanks=True,
    lowercase=False,
    separator="@@",
    recase=False
)

print("Building vocabulary...", file=sys.stderr)
vocab = corpus_builder.make_vocabulary([f"{bilingual_prefix}.{src}", f"{monolingual_prefix}.{src}"])
vocab.print_statistics(stdout=sys.stderr)

print("Building corpus...", file=sys.stderr)
corpus = corpus_builder.make_corpus(
    [f"{bilingual_prefix}.{src}", f"{monolingual_prefix}.{src}"],
    vocab,
    f"{output_prefix}.mono_src",
    max_length=max_length,
    reuse=False,
    verbose=True
)
print(f"Corpus: sentences={len(corpus)}", file=sys.stderr)


if max_length == -1:
    print("Assertions...", file=sys.stderr)
    assert all(original == processed for original, processed in
        zip(readlines([f"{bilingual_prefix}.{src}", f"{monolingual_prefix}.{src}"]), corpus_builder.post(corpus)))


# Here is how you can use data loaders (and possibly also sorted data loaders for RNNs)

print("Batching...", file=sys.stderr)
device = torch.device("cpu")
dl = SortingTextDataLoader(DataLoader(corpus, batch_size=batch_size, shuffle=shuffle))
for sentences in dl:
    x_in, x_out, x_mask, x_len, x_noisy_in = create_batch(sentences, vocab, device=device, word_dropout=0.)
    y = batch_to_sentences(x_out, vocab)

    for s, i, o, p in zip(sentences, x_in, x_out, corpus_builder.post(y)):
        print(s)
        print(i)
        print(o)
        print(p)
        print()

    break
