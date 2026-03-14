import warnings
import json
import re
import os
from functools import cached_property
import hashlib
from typing import Literal, Mapping, Optional, Union
from pydantic import Field, model_validator
import data_loaders as data
from ..configs.base import BaseConfig
from ..post_process import PostProcessing


def _recursive_getattr(obj, name, default_value):
    for n in name.split('.'):
        res = getattr(obj, n, None)
        if res is None:
            return default_value
        obj = res
    return res


class PostProcessingConfig(BaseConfig):
    bert_name: Literal['hfl/chinese-roberta-wwm-ext-large',
                       'sentence-transformers/all-roberta-large-v1']
    """bert used to calculate cosine similarity.
    Chinese: hfl/chinese-roberta-wwm-ext-large
    En: sentence-transformers/all-roberta-large-v1
    """
    threshold: float
    """threshold for cosine similarity clustering in post processing"""

    def init(
        self, raw_corpus: Mapping[int, list[str]], global_config,
        mapping_out_dir=None, verbose=False
    ):
        self._check_lang(raw_corpus)
        pids = list(raw_corpus)
        raw_corpus = list(raw_corpus.values())
        pp = PostProcessing.from_corpus(raw_corpus, bert_name=self.bert_name)
        corpus = pp.post_process(
            raw_corpus,
            threshold=self.threshold,
            verbose=verbose,
        )
        if mapping_out_dir is not None:
            self._save_mapping(
                dict(zip(pids, raw_corpus)),
                pp.mappings_cache[self.threshold],
                out_dir=mapping_out_dir,
            )
        return dict(zip(pids, corpus))

    def _save_mapping(
        self,
        raw_corpus: Mapping[int, list[str]],
        mapping: dict,
        out_dir: str,
    ):
        if out_dir is None:
            return
        corpus_hash = hashlib.sha1(json.dumps(raw_corpus).encode()).hexdigest()
        suffix = f'{self.threshold:.03f}'[2:]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{corpus_hash}_thr{suffix}.json')

        with open(out_path, 'w', encoding='utf8') as fout:
            json.dump(mapping, fout, ensure_ascii=False, indent=4)
        return

    def _check_lang(self, raw_corpus: Mapping[int, list[str]]):
        chinese_char_r_ = re.compile(r'[\u4E00-\u9FFF]')
        for pid, tags in raw_corpus.items():
            if chinese_char_r_.search(''.join(tags)) is not None:
                assert 'chinese' in self.bert_name
            return


class ParsingConfig(BaseConfig):

    start_line: int
    """line start parsing. Usually use 0.
    >>> lines = raw_predict.read_lines()
    >>> lines = lines[start_line:end_line]
    >>> tags = parse_tags(', '.join(lines))

    E.g.
    given the raw predict:
        1. tag1, tag2, ...             <---- input prompt
        2. taga, tagb, tagc, ...       <---- predicted started from 2
        3. ...
    Set '1' to skip the input prompt.
    """

    end_line: int
    """line to skip parsing. Usually use -1 because the last line is usaully 
    imcomplete.
    >>> lines = raw_predict.read_lines()
    >>> lines = lines[start_line:end_line]
    >>> tags = parse_tags(', '.join(lines))

    E.g.
    given the raw predict:
        1. tag1, tag2, ...             <---- input prompt
        2. taga, tagb, tagc, ...       <---- predicted started from 2
        3. ...
    Set '-1' to drop the last line.
    """

    include_input_line: Optional[bool] = None
    n_beams: Optional[int] = None

    def init(
        self,
        raw_predicts: Union[Mapping[int, str], Mapping[int, list[str]], list[str]],
        global_config,
        verbose=False,
    ):
        sample = list(raw_predicts.values())[0]
        if isinstance(sample, str):
            assert self.n_beams is None
            assert self.include_input_line is None
            return self._parse_single_prediction(raw_predicts)
        assert isinstance(sample, list) and isinstance(sample[0],
                                                       str), str(sample)
        assert self.n_beams is not None
        assert self.include_input_line is not None
        return self._parse_multi_beam_prediction(raw_predicts)

    def _parse_multi_beam_prediction(
        self, raw_predicts: Mapping[int, list[str]]
    ):
        # line2_r_ = re.compile(r'\n2\. ([^\n]+)')
        def line_i_r_(i: int):
            pat = r'\n' + str(i) + r'\. ([^\n]+)'
            return re.compile(pat)

        line_regs = [
            (i, line_i_r_(i + 1))
            for i in range(self.start_line, self.end_line)
        ]
        tag_r_ = re.compile(r'\b[a-zA-Z- \u4E00-\u9FFF\.\']+\b')

        def parse(raws: list[str]):
            if self.include_input_line:
                lines = [re.search(r'1\. ([^\n]+)', raws[0]).group(1)]
            else:
                lines = []
            assert len(raws) >= self.n_beams, (
                f'{len(raws) = }, {self.n_beams = }'
            )
            for raw in raws[:self.n_beams]:
                for i, line_r_ in line_regs:
                    match = line_r_.search(raw)
                    if match is not None:
                        lines.append(match.group(1))
                    else:
                        warnings.warn(
                            f'line: {repr(raw)} does not seem to have line {i + 1}'
                        )
            return lines

        illegal_r_ = re.compile(r'\*+([a-zA-Z- \u4E00-\u9FFF\'\.]+)\*+')

        def parse_line_to_tags(line: str):
            matches = illegal_r_.findall(line.lower())
            if matches:
                # warnings.warn(f'beam: {repr(line)} seems to be illegal')
                pass
            else:
                matches = tag_r_.findall(line.lower())
            return [tag.strip() for tag in matches if tag.strip()]

        raw_tags = {}
        for pid, raw_predict in raw_predicts.items():
            raw_predict: list[str]
            lines = parse(raw_predict)
            tags = []
            for line in lines:
                tags.extend(parse_line_to_tags(line))

            raw_tags[pid] = tags

        return raw_tags

    def _parse_single_prediction(self, raw_predicts: Mapping[int, str]):
        tag_r_ = re.compile(r'\b[a-zA-Z- \u4E00-\u9FFF]+\b')
        line_sep_r_ = re.compile(r'\n\d+\.')

        raw_tags = {}
        for pid, raw_predict in raw_predicts.items():
            raw_predict: str
            _, raw_predict = raw_predict.split(
                '1.', maxsplit=1
            )  # clean messages before '1.'

            tags = []
            for line in line_sep_r_.split(raw_predict
                                          )[self.start_line:self.end_line]:
                tags.extend(
                    tag.strip() for tag in tag_r_.findall(line.lower())
                    if tag.strip()
                )

            raw_tags[pid] = tags

        return raw_tags


class CorpusConfig(BaseConfig):

    name: str
    """one of the following:

    1. 'empty': create a dummy empty corpus, used for baseline predictor not using corpus
    2. data.<name>: preset data that can be found in the data module.
    3. <path>: path to a json. Accept one of following format
        - Mapping[pid, list[str]] raw predicted tags which would be postprocessed
            if postprocessing_config is specified.
        - Mapping[pid, str]: raw prediction file. Must specify parsing_config
             for parsing raw prediction into raw predicted tags. (would then
             get post-processed if postprocessing_config specified)
    """

    postprocessing_config: Optional[PostProcessingConfig] = Field(None)
    parsing_config: Optional[ParsingConfig] = Field(None)

    add_n_pid_tags: Optional[int] = Field(None)
    """add n pid tags to each item. Default to 0."""
    unique: bool
    """Whether only keep unique tags for each item.
    If False, the duplicated tags would weight more.
    """
    _version: int = 2

    def init(
        self, global_config=None, cache_dir=None, force_reload=False,
        verbose=False
    ) -> dict[int, list[str]]:

        if not force_reload and cache_dir is not None:
            corpus = self._load_from_cache(cache_dir)
            if corpus is not None:
                if verbose:
                    print('cache for corpus found, skip init')
                return corpus

        corpus = self._load(
            global_config, cache_dir=cache_dir, verbose=verbose
        )

        if self.unique:
            for pid, tags in corpus.items():
                corpus[pid] = list(set(tags))

        if self.add_n_pid_tags:
            for pid, tags in corpus.items():
                tags.extend([str(pid)] * self.add_n_pid_tags)

        if cache_dir is not None:
            self._save_corpus(corpus, cache_dir, verbose)
        return corpus

    def _load_from_cache(self, cache_dir: Optional[str]):
        if cache_dir is None:
            return None
        path = os.path.join(cache_dir, f'{self.hash}.json')
        if not os.path.exists(path):
            return None

        with open(path, 'r', encoding='utf8') as fin:
            return {int(pid): tags for pid, tags in json.load(fin).items()}

    def _save_corpus(
        self,
        corpus: dict[int, list[str]],
        cache_dir: Optional[str],
        verbose,
    ):
        if cache_dir is None:
            return

        corpus_json = json.dumps(corpus, indent=4, ensure_ascii=False)
        os.makedirs(cache_dir, exist_ok=True)
        p = os.path.abspath(os.path.join(cache_dir, f'{self.hash}.json'))
        with open(p, 'w', encoding='utf8') as fout:
            fout.write(corpus_json)

        if verbose:
            print(f'cache saved in {p}')
        return

    def _load(self, global_config, cache_dir, verbose):

        if self.name == 'empty':
            return {}

        if os.path.isfile(self.name):
            corpus = self._load_corpus_json

            if self.parsing_config is not None:
                corpus = self.parsing_config.init(
                    corpus, global_config, verbose
                )

            if self.postprocessing_config is not None:
                corpus = self.postprocessing_config.init(
                    corpus,
                    global_config,
                    mapping_out_dir=os.path.join(cache_dir, 'mappings'),
                    verbose=verbose,
                )
            return corpus

        corpus_getter = _recursive_getattr(data, self.name, None)
        assert corpus_getter is not None, self.name
        return corpus_getter(use_pid=True)

    @cached_property
    def _load_corpus_json(self) -> dict[int, list[str]]:
        assert os.path.isfile(self.name)
        with open(self.name, 'r', encoding='utf8') as fin:
            corpus = json.load(fin)
            assert isinstance(corpus, dict)
            return {int(pid): tags for pid, tags in corpus.items()}

    @model_validator(mode='after')
    @property
    def check(self):
        if self.name == 'empty':
            return self

        if _recursive_getattr(data, self.name, None) is not None:
            return self

        assert os.path.isfile(self.name), f'{self.name} is not file'
        corpus = self._load_corpus_json

        if isinstance(list(corpus.values())[0], str):
            assert self.parsing_config is not None, (
                'a raw prediction file is given, while the self.parsing_config is None'
            )

        return self

    @cached_property
    def hash(self):
        d = self.model_dump()
        d['_version'] = self._version
        if os.path.isfile(self.name):
            d['name'] = self._load_corpus_json
        return hashlib.sha1(json.dumps(d).encode()).hexdigest()
