import os
import sys

import torch
from torch.utils.data import DataLoader

from text.recipes import BilingualWordLevel
from text.batching import create_batch, batch_to_sentences


src = "de"
tgt = "en"
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

corpus_builder = BilingualWordLevel(
    src_code=src,
    tgt_code=tgt,
    normalize_blanks=True,
    lowercase=False,
    recase=False
)

print("Building source vocabulary...", file=sys.stderr)
src_vocab = corpus_builder.make_src_vocabulary([f"{bilingual_prefix}.{src}", f"{monolingual_prefix}.{src}"])
src_vocab.print_statistics(stdout=sys.stderr)
print("Building target vocabulary...", file=sys.stderr)
tgt_vocab = corpus_builder.make_src_vocabulary([f"{bilingual_prefix}.{tgt}", f"{monolingual_prefix}.{tgt}"])
tgt_vocab.print_statistics(stdout=sys.stderr)

print("Building parallel corpus...", file=sys.stderr)
corpus = corpus_builder.make_corpus(
    [f"{bilingual_prefix}.{src}"],
    [f"{bilingual_prefix}.{tgt}"],
    src_vocab, tgt_vocab,
    f"{output_prefix}.src-tgt",
    max_length=max_length,
    reuse=False,
    verbose=True
)
print(f"Parallel corpus: sentences={len(corpus)}", file=sys.stderr)


# Here is how you can use data loaders (and possibly also sorted data loaders for RNNs)

print("Batching...", file=sys.stderr)
device = torch.device("cpu")
# TODO: get a sorted data loader for parallel text
dl = DataLoader(corpus, batch_size=batch_size, shuffle=shuffle)
for src_sentences, tgt_sentences in dl:

    x_in, x_out, x_mask, x_len, x_noisy_in = create_batch(src_sentences, src_vocab, device=device, word_dropout=0.)
    x = batch_to_sentences(x_out, src_vocab)

    print("Source")
    for s, i, o, p in zip(src_sentences, x_in, x_out, corpus_builder.src_builder.post(x)):
        print(s)
        print(i)
        print(o)
        print(p)
        print()

    y_in, y_out, y_mask, y_len, y_noisy_in = create_batch(tgt_sentences, tgt_vocab, device=device, word_dropout=0.)
    y = batch_to_sentences(y_out, tgt_vocab)
    print("Target")
    for t, i, o, p in zip(tgt_sentences, y_in, y_out, corpus_builder.tgt_builder.post(y)):
        print(t)
        print(i)
        print(o)
        print(p)
        print()

    break
