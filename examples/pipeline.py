from itertools import tee

from text import SentenceSplit, Preprocess, EnsureMaxLength, Postprocess
from text import SentenceSegmenter, Pipeline, Tokenizer, Lowercaser, WordSegmenter


path = "raw.en"

pipeline = Pipeline(
    [
        Tokenizer("en", "en"),
        Lowercaser(),
        WordSegmenter("data/bpe_codes.en", separator="@@")
    ]
)

generator = iter(open(path))
generator, it = tee(generator, 2)
print("# After open()")
for sentence in it:
    print(sentence.strip())

generator = SentenceSplit(SentenceSegmenter("en"), read_n=1)(generator)
generator, it = tee(generator, 2)
print("# After SentenceSplit")
for sentence in it:
    print(sentence)
print()

generator = Preprocess(pipeline)(generator)
generator, it = tee(generator, 2)
print("# After Preprocess")
for sentence in it:
    print(sentence)
print()

length_adapter = EnsureMaxLength(max_length=10, split=True)
generator = length_adapter(generator)
generator, it = tee(generator, 2)
print("# After EnsureMaxLength")
for sentence in it:
    print(sentence)
print()

generator = length_adapter.join(generator)
generator, it = tee(generator, 2)
print("# After EnsureMaxLength.join")
for sentence in it:
    print(sentence)
print()

generator = Postprocess(pipeline)(generator)
print("# After Postprocess")
for sentence in generator:
    print(sentence)
print()
