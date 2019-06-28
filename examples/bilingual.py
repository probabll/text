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
reuse = False

try:
    os.mkdir("output")
except:
    pass

from text import Vocabulary
from text import MemMappedParallelCorpus

vocab_src = Vocabulary.from_data([f"{bilingual_prefix}.{src}", f"{monolingual_prefix}.{src}"])
vocab_src.print_statistics(stdout=sys.stderr)

vocab_tgt = Vocabulary.from_data([f"{bilingual_prefix}.{tgt}", f"{monolingual_prefix}.{tgt}"])
vocab_tgt.print_statistics(stdout=sys.stderr)

parallel = MemMappedParallelCorpus(
    generator=MemMappedParallelCorpus.get_generator(
        [f"{bilingual_prefix}.{src}", f"{bilingual_prefix}.{tgt}"], max_length=max_length),
    output_path=f"{output_prefix}.bilingual",
    vocabs=[vocab_src, vocab_tgt],
    reuse=False,
    verbose=True
)
print(f"Parallel corpus: sentences={len(parallel)}", file=sys.stderr)

print("Corpus in given order")
for i in range(len(parallel)):
    print(f"{i+1} ||| {parallel[i][0]} ||| {parallel[i][1]}")

ids = list(range(len(parallel)))
shuffle(ids)

parallel = MemMappedParallelCorpus(
    generator=MemMappedParallelCorpus.get_generator(
        [f"{bilingual_prefix}.{src}", f"{bilingual_prefix}.{tgt}"], max_length=max_length),
    output_path=f"{output_prefix}.bilingual",
    vocabs=[vocab_src, vocab_tgt],
    reuse=True,
    verbose=True
)
print(f"Parallel corpus: sentences={len(parallel)}", file=sys.stderr)

print("Corpus in random order")
for i in ids:
    print(f"{i+1} ||| {parallel[i][0]} ||| {parallel[i][1]}")