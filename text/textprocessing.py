import sacremoses
from subword_nmt.apply_bpe import BPE
from mosestokenizer import MosesSentenceSplitter
import codecs
import re


class SentenceSegmenter:
    """
    Processes a list of strings into a list of strings where each entry of the list
    is considered a sentence on its on.

    For now this only supports the Moses sentence splitter (via mosestokenizer wrapper).
    """

    def __init__(self, language_code):
        self.language_code = language_code

    def split(self, data: list) -> list:
        data = [line for line in data if line.strip()]
        if len(data):
            with MosesSentenceSplitter(self.language_code) as splitter:
                data = splitter(data)
        return data


class TextProcessingModule:
    """
    A generic interface for pre-processing and post-processing strings.
    """

    def __call__(self, line: str) -> str:
        return line


class Tokenizer(TextProcessingModule):
    """
    Wrapper around sacremoses.MosesTokenizer
    """

    def __init__(self, lang):
        self.tokenizer = sacremoses.MosesTokenizer(lang)

    def __call__(self, line: str) -> str:
        return self.tokenizer.tokenize(line, return_str=True)


class Detokenizer(TextProcessingModule):
    """
    Wrapper around sacremoses.MosesDetokenizer
    """

    def __init__(self, lang):
        self.detokenizer = sacremoses.MosesDetokenizer(lang)

    def __call__(self, line: str) -> str:
        return self.detokenizer.detokenize(line.split(), return_str=True)


class Lowercaser(TextProcessingModule):
    """
    Wrapper around python string.lower()
    """

    def __init__(self, lang):
        pass

    def __call__(self, line: str) -> str:
        return line.lower()


class Truecaser(TextProcessingModule):
    """
    Wrapper around sacremoses.MosesTruecaser
    """

    def __init__(self, truecase_model):
        self.truecaser = sacremoses.MosesTruecaser(truecase_model)

    def __call__(self, line: str) -> str:
        return self.truecaser.truecase(line, return_str=True)


class Recaser(TextProcessingModule):
    """
    Wrapper around sacremoses.MosesDetruecaser
    """

    def __init__(self, lang):
        self.recaser = sacremoses.MosesDetruecaser()

    def __call__(self, line: str) -> str:
        return self.recaser.detruecase(line, return_str=True)


class WordSegmenter(TextProcessingModule):
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


class WordDesegmenter(TextProcessingModule):
    """
    Wrapper around a regex expression.
    """

    def __init__(self, separator="@@", encoding='utf-8'):
        self.separator = separator.strip()

    def __call__(self, line: str) -> str:
        return re.sub(f"({self.separator} )|({self.separator} ?$)|( {self.separator})|(^ ?{self.separator})", "", line)


class Pipeline:
    """
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
