"""
Text processing tools that work as generators including:

1. ListToList tools: input is a list of strings, output is a list of strings, number of entries can change

* SentenceSegmenter

2. StringToString tools: input is a string, output is a string, length can change

* lowercasing
* tokenization (as in separating tokens by an empty space)
* case normalisation
* a Pipeline applies a sequence of such modules


"""

import sacremoses
from subword_nmt.apply_bpe import BPE
from mosestokenizer import MosesSentenceSplitter
import codecs
import re
from typing import List


class ListToList:

    def __call__(self, data: List[str]) -> List[str]:
        raise NotImplementedError("Implement me!")


class SentenceSegmenter(ListToList):
    """
    Processes a list of strings into a list of strings where each entry of the list
    is considered a sentence on its on.

    For now this only supports the Moses sentence splitter (via mosestokenizer wrapper).
    """

    def __init__(self, language_code):
        self.language_code = language_code

    def __call__(self, data: List[str]) -> List[str]:
        data = [line for line in data if line.strip()]
        if len(data):
            with MosesSentenceSplitter(self.language_code) as splitter:
                data = splitter(data)
        return data


class StringToString:
    """
    A generic interface for pre-processing and post-processing strings.
    """

    def __call__(self, line: str) -> str:
        return line


class BlankNormalizer(StringToString):
    """Wrapper around re.sub(r"\s+", " ", line)"""

    def __init__(self):
        self.pattern = re.compile(r"\s+")

    def __call__(self, line: str) -> str:
        return self.pattern.sub(" ", line)


class Tokenizer(StringToString):
    """
    Wrapper around sacremoses.MosesTokenizer
    """

    def __init__(self, lang):
        self.tokenizer = sacremoses.MosesTokenizer(lang)

    def __call__(self, line: str) -> str:
        return self.tokenizer.tokenize(line, return_str=True)


class Detokenizer(StringToString):
    """
    Wrapper around sacremoses.MosesDetokenizer
    """

    def __init__(self, lang):
        self.detokenizer = sacremoses.MosesDetokenizer(lang)

    def __call__(self, line: str) -> str:
        return self.detokenizer.detokenize(line.split(), return_str=True)


class Lowercaser(StringToString):
    """
    Wrapper around python string.lower()
    """

    def __init__(self, lang):
        pass

    def __call__(self, line: str) -> str:
        return line.lower()


class Truecaser(StringToString):
    """
    Wrapper around sacremoses.MosesTruecaser
    """

    def __init__(self, truecase_model):
        self.truecaser = sacremoses.MosesTruecaser(truecase_model)

    def __call__(self, line: str) -> str:
        return self.truecaser.truecase(line, return_str=True)


class Recaser(StringToString):
    """
    Wrapper around sacremoses.MosesDetruecaser
    """

    def __init__(self, lang):
        self.recaser = sacremoses.MosesDetruecaser()

    def __call__(self, line: str) -> str:
        return self.recaser.detruecase(line, return_str=True)


class BPESegmenter(StringToString):
    """
    Wrapper around subword_nmt.apply_bpe.BPE
    """

    def __init__(self, bpe_codes, separator="@@", encoding='utf-8'):
        self.separator = separator.strip()
        self.bpe = BPE(
            codecs.open(bpe_codes, encoding=encoding),
            separator=self.separator)

    def __call__(self, line: str) -> str:
        return self.bpe.process_line(line)


class BPEDesegmenter(StringToString):
    """
    Wrapper around a regex expression.
    """

    def __init__(self, separator="@@", encoding='utf-8'):
        self.separator = separator.strip()

    def __call__(self, line: str) -> str:
        return re.sub(f"({self.separator} )|({self.separator} ?$)|( {self.separator})|(^ ?{self.separator})", "", line)


class CharLevelSegmenter(StringToString):
    """
    Turns a string such as "this is a string" into
        "t h i s @@ i s @@ a @@ s t r i n g"
    using a separator (such as @@).
    """

    def __init__(self, separator="@@", encoding='utf-8'):
        self.separator = separator.strip()

    def __call__(self, line: str) -> str:
        return f" {self.separator} ".join(' '.join(tok) for tok in line.strip().split())


class CharLevelDesegmenter(StringToString):
    """
    Turns a string such as "t h i s @@ i s @@ a @@ s t r i n g" into
        "this is a string"
    assuming a separator such as @@.
    """

    def __init__(self, separator="@@", encoding='utf-8'):
        self.separator = separator.strip()

    def __call__(self, line: str) -> str:
        return ''.join(" " if char == self.separator else char for char in line.split())


class Pipeline(StringToString):

    def __init__(self, modules=[]):
        self._modules = modules

    def __call__(self, line: str) -> str:
        for module in self._modules:
            line = module(line)
        return line


class PrePostPipeline:
    """
    TODO: get rid of this

    Applies a sequence of preprocessing or postprocessing steps
    """

    def __init__(self, pre=[], post=[]):
        self._pre = pre
        self._post = post

    def pre(self, line: str) -> str:
        for module in self._pre:
            line = module(line)
        return line

    def post(self, line: str) -> str:
        for module in self._post:
            line = module(line)
        return line
