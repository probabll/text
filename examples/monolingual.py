# load bilingual corpora
# no preprocessing
# MemMap
import os
import sys
from random import shuffle

src = "de"
tgt = "en"
bilingual_prefix = "data/training"
monolingual_prefix = "data/comparable"
output_prefix = "output/text.pt.corpus"
max_length = 50

try:
    os.mkdir("output")
except:
    pass

from text import Vocabulary
from text import MemMappedCorpus

vocab_src = Vocabulary.from_data([f"{bilingual_prefix}.{src}", f"{monolingual_prefix}.{src}"])
vocab_src.print_statistics(stdout=sys.stderr)

vocab_tgt = Vocabulary.from_data([f"{bilingual_prefix}.{tgt}", f"{monolingual_prefix}.{tgt}"])
vocab_tgt.print_statistics(stdout=sys.stderr)

mono_src = MemMappedCorpus(
    generator=MemMappedCorpus.get_generator(f"{monolingual_prefix}.{src}", max_length=max_length),
    output_path=f"{output_prefix}.mono_src",
    vocab=vocab_src,
    reuse=False,
    verbose=True
)


print(f"Source monolingual corpus: sentences={len(mono_src)}", file=sys.stderr)

print("Corpus in given order")
for i in range(len(mono_src)):
    print(f"{i+1} ||| {mono_src[i]}")

ids = list(range(len(mono_src)))
shuffle(ids)

mono_src = MemMappedCorpus(
    generator=MemMappedCorpus.get_generator(f"{bilingual_prefix}.{src}", max_length=max_length),
    output_path=f"{output_prefix}.mono_src",
    vocab=vocab_src,
    reuse=True,
    verbose=True
)

print(f"Source monolingual corpus: sentences={len(mono_src)}", file=sys.stderr)


print("Corpus in random order")
for i in ids:
    print(f"{i+1} ||| {mono_src[i]}")