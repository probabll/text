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

    def __init__(self, language_code: str):
        """

        :param language_code:
        """
        self.language_code = language_code

    def split(self, data: list):
        data = [line for line in data if line.strip()]  # empty strings are not accepted by the wrapper
        if len(data):
            with MosesSentenceSplitter(self.language_code) as splitter:
                data = splitter(data)
        return data


class TextProcess:
    """
    A generic interface for pre-processing and post-processing strings.
    """

    def pre(self, line: str) -> str:
        return line

    def post(self, line: str) -> str:
        return line


class Tokenizer(TextProcess):
    """
    Wrapper around sacremoses.MosesTokenizer and MosesDetokenizer.

    This takes two parameters src_tokenizer_lang and tgt_tokenizer_lang and provides:
    * pre: source tokenization
    * post: target tokenization

    Note that this can be used monolingually,
        simply use the same language id for both src_tokenizer_lang and tgt_tokenizer.
    """

    def __init__(self, src_tokenizer_lang, tgt_tokenizer_lang):
        self.src_tokenizer = sacremoses.MosesTokenizer(src_tokenizer_lang) if src_tokenizer_lang else None
        self.tgt_detokenizer = sacremoses.MosesDetokenizer(tgt_tokenizer_lang) if tgt_tokenizer_lang else None

    def pre(self, line: str) -> str:
        if self.src_tokenizer:
            return self.src_tokenizer.tokenize(line, return_str=True)
        else:
            return line

    def post(self, line: str) -> str:
        if self.tgt_detokenizer:
            return self.tgt_detokenizer.detokenize(line.split(), return_str=True)
        else:
            return line


class Lowercaser(TextProcess):
    """
    Wrapper around str.lower and sacremoses.MosesDetruecaser.

    * pre: lowercasing
    * post: detruecasing
    """

    def __init__(self):
        self.tgt_detruecaser = sacremoses.MosesDetruecaser()

    def pre(self, line: str) -> str:
        return line.lower()

    def post(self, line: str) -> str:
        if self.tgt_detruecaser:
            return self.tgt_detruecaser.detruecase(line, return_str=True)
        else:
            return line


class Truecaser(TextProcess):
    """
    Wrapper around sacremoses.MosesTruecaser.

    This takes two parameters src_tokenizer_lang and tgt_tokenizer_lang and provides:
    * pre: source truecasing
    * post: target detruecasing

    Note that this can be used monolingually,
        simply use the same file for both src_truecase_model and tgt_truecase_model.
    """

    def __init__(self, src_truecase_model, tgt_truecase_model):
        self.src_truecaser = sacremoses.MosesTruecaser(src_truecase_model) if src_truecase_model else None
        self.tgt_detruecaser = sacremoses.MosesDetruecaser() if tgt_truecase_model else None

    def pre(self, line: str) -> str:
        if self.src_truecaser:
            return self.src_truecaser.truecase(line, return_str=True)
        else:
            return line

    def post(self, line: str) -> str:
        if self.tgt_detruecaser:
            return self.tgt_detruecaser.detruecase(line, return_str=True)
        else:
            return line


class WordSegmenter(TextProcess):
    """
        Wrapper around subword_nmt.apply_bpe.BPE

        This takes two parameters src_bpe_codes and separator and provides:
        * pre: BPE segmentation
        * post: regex for merging segments
        """

    def __init__(self, src_bpe_codes, separator="@@", encoding='utf-8'):
        separator = separator.strip() if separator else separator
        self.src_bpe = BPE(
            codecs.open(src_bpe_codes, encoding=encoding), separator=separator) if src_bpe_codes else None
        self.separator = separator

    def pre(self, line: str) -> str:
        if self.src_bpe:
            return self.src_bpe.process_line(line)
        else:
            return line

    def post(self, line: str) -> str:
        if self.separator:
            return re.sub(f"({self.separator} )|({self.separator} ?$)|( {self.separator})|(^ ?{self.separator})", "", line)
        else:
            return line


class Pipeline(TextProcess):
    """
    A generic pipeline takes a list of TextProcess objects and provides:

    * pre: applies pipeline in order making calls to pre(str)
    * post: applies pipeline in reversed order making calls to post(str)
    """

    def __init__(self, pipeline=[]):
        self.pipeline = pipeline

    def pre(self, line: str) -> str:
        for module in self.pipeline:
            line = module.pre(line)
        return line

    def post(self, line: str) -> str:
        for module in reversed(self.pipeline):
            line = module.post(line)
        return line
