"""
Here we provide recipes to load vocabularies and corpora.
"""

from text import Vocabulary
from text import MemMappedCorpus, MemMappedParallelCorpus
from text.textprocessing import BlankNormalizer, Lowercaser, Recaser
from text.textprocessing import CharLevelSegmenter, CharLevelDesegmenter
from text.datasets import readlines, basic_tokenize, basic_tokenize_parallel
from text.lazy import Pipeline, Preprocess, Postprocess


class MonolingualCharLevel:

    def __init__(self, lang_code, normalize_blanks=True, lowercase=False, separator="@@", recase=False):
        pre_modules = []
        if normalize_blanks:
            pre_modules.append(BlankNormalizer())
        if lowercase:
            pre_modules.append(Lowercaser(lang_code))
        pre_modules.append(CharLevelSegmenter(separator))
        post_modules = [CharLevelDesegmenter(separator)]
        if recase:
            post_modules.append(Recaser(lang_code))
        self.pipeline = Pipeline(pre_modules, post_modules)
        self.pre = Preprocess(self.pipeline)
        self.post = Postprocess(self.pipeline)

    def make_vocabulary(self, filenames: list):
        vocab = Vocabulary.from_generator(
             self.pre(readlines(filenames))
        )
        return vocab

    def make_corpus(self, filenames: list, vocab: Vocabulary, output_path,
                    max_length=-1, reuse=False, verbose=False, dtype="int64"):
        corpus = MemMappedCorpus(
            generator=basic_tokenize(  # str -> list[str]
                self.pre(readlines(filenames)),
                max_length=max_length
            ),
            output_path=output_path,
            vocab=vocab,
            reuse=reuse,
            verbose=verbose,
            dtype=dtype
        )
        return corpus


class MonolingualWordLevel:

    def __init__(self, lang_code, normalize_blanks=True, lowercase=False, recase=False):
        pre_modules = []
        if normalize_blanks:
            pre_modules.append(BlankNormalizer())
        if lowercase:
            pre_modules.append(Lowercaser(lang_code))
        post_modules = []
        if recase:
            post_modules.append(Recaser(lang_code))
        self.pipeline = Pipeline(pre_modules, post_modules)
        self.pre = Preprocess(self.pipeline)
        self.post = Postprocess(self.pipeline)

    def make_vocabulary(self, filenames: list):
        vocab = Vocabulary.from_generator(
             self.pre(readlines(filenames))
        )
        return vocab

    def make_corpus(self, filenames: list, vocab: Vocabulary, output_path,
                    max_length=-1, reuse=False, verbose=False, dtype="int64"):
        corpus = MemMappedCorpus(
            generator=basic_tokenize(  # str -> list[str]
                self.pre(readlines(filenames)),
                max_length=max_length
            ),
            output_path=output_path,
            vocab=vocab,
            reuse=reuse,
            verbose=verbose,
            dtype=dtype
        )
        return corpus


class BilingualWordLevel:

    def __init__(self, src_code, tgt_code, normalize_blanks=True, lowercase=False, recase=False):
        self.src_builder = MonolingualWordLevel(src_code, normalize_blanks, lowercase, recase)
        self.tgt_builder = MonolingualWordLevel(tgt_code, normalize_blanks, lowercase, recase)

    def make_src_vocabulary(self, filenames: list):
        return self.src_builder.make_vocabulary(filenames)

    def make_tgt_vocabulary(self, filenames: list):
        return self.tgt_builder.make_vocabulary(filenames)

    def make_corpus(self, src_filenames: list, tgt_filenames: list,
                    src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                    output_path,
                    max_length=-1, reuse=False, verbose=False, dtype="int64"):
        corpus = MemMappedParallelCorpus(
            generator=basic_tokenize_parallel(  # str -> list[str]
                [  # two streams/generators
                    self.src_builder.pre(readlines(src_filenames)),
                    self.tgt_builder.pre(readlines(tgt_filenames))
                ],
                max_length=max_length
            ),
            output_path=output_path,
            vocabs=[src_vocab, tgt_vocab],
            reuse=reuse,
            verbose=verbose,
            dtype=dtype
        )
        return corpus
