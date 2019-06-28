import numpy as np

import os
import sys
from itertools import tee

from torch.utils.data import Dataset

from text.vocabulary import Vocabulary


class MemMappedCorpus(Dataset):
    """
    A memory mapped Dataset:
    - this never loads the entire data to memory
    - and still allows for random memory access
    """

    @staticmethod
    def make_offsets(lengths):
        offsets = np.zeros_like(lengths)
        offset = 0
        # populate memory map
        for i, seq_len in enumerate(lengths):
            offsets[i] = offset
            offset += seq_len
        return offsets

    @staticmethod
    def map_tokens_to_ids(generator, vocab: Vocabulary):
        """
        Wraps a generator around a mapping function from string tokens to token ids.
        :param generator:
        :param vocab:
        :return: generator
        """
        for sentence in generator:
            yield [vocab[word] for word in sentence]

    @staticmethod
    def iterate_view(generator, stream):
        """Returns one element of the tuple"""
        for t in generator:
            yield t[stream]

    @staticmethod
    def construct_memmap(generator, lengths, output_path, dtype):
        """Stores a np.array with shape [nb_tokens] in disk"""
        nb_tokens = np.sum(lengths)

        # construct memory mapped array
        mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=nb_tokens)

        # prepare for populating memmap
        offset = 0
        offsets = []

        # populate memory map
        for seq, seq_len in zip(generator, lengths):
            offsets.append(offset)
            # here we have a valid sequence, thus we memory map it
            mmap[offset:offset + seq_len] = seq
            offset += seq_len

        del mmap
        return np.array(offsets, dtype=dtype)

    @staticmethod
    def basic_tokenize_parallel(files: list, max_length=-1):
        """
        Reads in parallel tuples and returns a generator of tuples of sentences (each a list of string tokens).
        Tokenization is done by string.split()

        :param files: each file in this list is one stream of the corpus.
        :param max_length: if more than 0, discards a tuple if any of its sentences is longer than the value specified
        :return: a generator of tuples of sentences (lists of string tokens)
        """

        handlers = []
        for path in files:
            handlers.append(open(path))

        for lines in zip(*handlers):
            sentences = [line.strip().split() for line in lines]
            if 0 < max_length < max(len(sentence) for sentence in sentences):
                continue
            yield sentences

        for h in handlers:
            h.close()

    @staticmethod
    def get_generator(path: str, max_length=-1):
        generator = MemMappedCorpus.iterate_view(
            MemMappedCorpus.basic_tokenize_parallel([path], max_length=max_length), stream=0)
        return generator

    def __init__(self, generator, output_path: str, vocab: Vocabulary,
                 dtype='int64', reuse=True, verbose=False):
        """

        :param generator: produces sequences of tokens (each a list of strings)
            see MemMappedCorpus.get_generator
        :param memmap_path:
        :param vocab:
        :param dtype:
        :param reuse: by default we reload already constructed memmaps
            switch this to False to reconstruct them
        """
        self.vocab = vocab
        self.lengths_path = f"{output_path}.lengths.npy"  # if we don't have .npy np.save will add it
        self.memmap_path = f"{output_path}.memmap"

        # Try to load existing data structures
        if reuse and os.path.isfile(self.lengths_path) and os.path.isfile(self.memmap_path):
            self.lengths = np.load(self.lengths_path)
            self.offsets = MemMappedCorpus.make_offsets(self.lengths)
        else:  # construct data structures
            generator, iterator1, iterator2 = tee(generator, 3)
            self.lengths = np.array([len(s) for s in iterator1], dtype=dtype)
            np.save(self.lengths_path, self.lengths)
            if verbose:
                print("Constructing memory map", file=sys.stderr)
            self.offsets = MemMappedCorpus.construct_memmap(
                MemMappedCorpus.map_tokens_to_ids(iterator2, vocab),
                self.lengths,
                self.memmap_path,
                dtype=dtype)
        if verbose:
            print(f"Loading memory map from {self.memmap_path}", file=sys.stderr)
        self.mmap = np.memmap(self.memmap_path, dtype=dtype, mode='r')
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def as_memmap(self, idx):
        offset = self.offsets[idx]
        return self.mmap[offset: offset + self.lengths[idx]].tolist()

    def as_string(self, idx):
        return ' '.join(self.vocab.word(t) for t in self.as_memmap(idx))

    def __getitem__(self, idx):
        return self.as_string(idx)


class MemMappedParallelCorpus(Dataset):

    @staticmethod
    def get_generator(paths: list, max_length=-1):
        generator = MemMappedCorpus.basic_tokenize_parallel(paths, max_length=max_length)
        return generator

    def __init__(self, generator, output_path: str, vocabs: 'list[Vocabulary]',
                 dtype='int64', reuse=True, verbose=True):
        """

        :param generator: A generator for parallel data (this should already take care of max_length constraints,
            see MemMappedParallelCorpus.get_generator).
        :param output_path: where to store files (this is considered a prefix)
        :param vocabs: list of Vocabulary objects (one per stream of the data)
        :param reuse: by default we load already stored data structures,
            switch this to False to reconstruct them
        :param dtype:
        """
        self.nb_streams = len(vocabs)
        iterators = tee(generator, self.nb_streams + 1)
        self.generator = iterators[0]
        self.corpora = []
        for i, it in enumerate(iterators[1:]):
            corpus = MemMappedCorpus(
                generator=MemMappedCorpus.iterate_view(it, i),
                output_path=f"{output_path}{i}",
                vocab=vocabs[i],
                dtype=dtype,
                reuse=reuse,
                verbose=verbose)
            self.corpora.append(corpus)

    def __len__(self):
        return len(self.corpora[0])

    def as_memmap(self, idx):
        return [corpus.as_memmap(idx) for corpus in self.corpora]

    def as_string(self, idx):
        return [corpus.as_string(idx) for corpus in self.corpora]

    def __getitem__(self, idx):
        return [corpus[idx] for corpus in self.corpora]
