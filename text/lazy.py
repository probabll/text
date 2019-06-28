from text.textprocessing import TextProcess, SentenceSegmenter


class Preprocess:

    def __init__(self, pipeline: TextProcess):
        self.pipeline = pipeline

    def __call__(self, generator):
        for line in generator:
            yield self.pipeline.pre(line.strip())


class Postprocess:

    def __init__(self, pipeline: TextProcess):
        self.pipeline = pipeline

    def __call__(self, generator):
        for line in generator:
            yield self.pipeline.post(line.strip())


class SentenceSplit:

    def __init__(self, splitter: SentenceSegmenter, read_n=1):
        """

        :param splitter:
        :param read_n: sends a number of lines to the sentence splitter
            use -1 for all lines (WARNING: this means the whole data will be in memory!)
        """
        self.splitter = splitter
        self.read_n = read_n

    def __call__(self, generator):
        if self.read_n > 0:
            batch = []
            for line in generator:
                batch.append(line)
                if len(batch) >= self.read_n:
                    for sentence in self.splitter.split(batch):
                        yield sentence
                    batch = []
            if len(batch) > 0:
                for sentence in self.splitter.split(batch):
                    yield sentence
        else:
            for sentence in self.splitter.split([line for line in generator]):
                yield sentence


class EnsureMaxLength:

    def __init__(self, max_length=-1, split=False):
        self.max_length = max_length
        self.split = split
        self.nb_parts = []

    @staticmethod
    def split_list(x, size):
        output = []
        if size < 0:
            return [x]
        elif size == 0:
            raise ValueError("Use size -1 for no splitting or size more than 0.")
        while True:
            if len(x) > size:
                output.append(x[:size])
                x = x[size:]
            else:
                output.append(x)
                break
        return output

    def __call__(self, generator):
        if self.max_length < 0:
            yield from generator
        for line in generator:
            line = line.strip()  # strip \n
            tokens = line.split()
            if len(tokens) <= self.max_length:
                self.nb_parts.append(1)
                yield line
            elif self.split:  # sentence splitting (saves metadata to restore line-alignment with input)
                parts = EnsureMaxLength.split_list(tokens, self.max_length)
                self.nb_parts.append(len(parts))
                for part in parts:
                    yield ' '.join(part)

    def join(self, generator):
        if self.max_length < 0 or not self.split:
            yield from generator
        else:
            # This is important: note that perhaps the generator hasn't yet generated nb_parts
            # thus we cannot iterate over nb_parts
            # we should rather get a line from the generator and maintain an index
            i = 0
            parts = []
            for line in generator:
                parts.append(line)
                if len(parts) == self.nb_parts[i]:
                    i += 1
                    yield ' '.join(parts)
                    parts = []