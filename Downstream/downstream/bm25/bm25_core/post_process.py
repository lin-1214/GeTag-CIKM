from typing import Optional
import re
from collections import defaultdict
import hashlib
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from . import corpus_utils


def load_raw_predict(
    raw_predict_json: str,
    line_start: int = 0,
    line_end: int = -1,
):
    tag_r_ = re.compile(r'\b[a-zA-Z- \u4E00-\u9FFF]+\b')
    line_sep_r_ = re.compile(r'\n\d+\.')
    with open(raw_predict_json, 'r', encoding='utf8') as fin:
        raw_predicts = json.load(fin)
        raw_predicts = {int(pid): v for pid, v in raw_predicts.items()}

    raw_tags = {}
    for pid, raw_predict in raw_predicts.items():
        raw_predict: str
        _, raw_predict = raw_predict.split(
            '1.', maxsplit=1
        )  # clean messages before '1.'

        tags = []
        for line in line_sep_r_.split(raw_predict)[line_start:line_end]:
            tags.extend(
                tag.strip() for tag in tag_r_.findall(line.lower())
                if tag.strip()
            )

        raw_tags[pid] = tags

    return raw_tags


def tag_similarity(
    tag_domain: list[str],
    ref_domain: Optional[list[str]],
    bert_name: Optional[str] = None,
):
    if bert_name is None:
        bert_name = 'sentence-transformers/all-roberta-large-v1'

    model = SentenceTransformer(bert_name, cache_folder='.cache')
    tag_embs = model.encode(tag_domain)
    if ref_domain is None:
        ref_embs = tag_embs
    else:
        ref_embs = model.encode(ref_domain)

    similarity_mat = cosine_similarity(ref_embs, tag_embs)
    """candidate -> referrer"""
    return similarity_mat


class PostProcessing:

    def __init__(
        self,
        tag_domain: list[str],
        ref_domain: Optional[list[str]] = None,
        bert_name: Optional[str] = None,
    ):

        self.tag_domain = tag_domain
        self.ref_domain = ref_domain or tag_domain
        self.bert_name = bert_name
        self.mappings_cache: dict[int, tuple[dict[str, list[str]],
                                             dict[str, str]]] = {}
        return

    @classmethod
    def from_corpus(
        cls,
        tag_corpus: list[list[str]],
        ref_domain: Optional[list[str]] = None,
        bert_name: Optional[str] = None,
    ):
        return cls(
            tag_domain=corpus_utils.to_domain(tag_corpus, sort=True),
            ref_domain=ref_domain,
            bert_name=bert_name,
        )

    # def _load_from_cache(self, path: str):
    #     print('loading from cached results...')
    #     with open(path, 'r', encoding='utf8') as fout:
    #         tag_post = json.load(fout)
    #     inverse_mapping = {}
    #     for representing_tag, represented_tags in tag_post.items():
    #         for t in represented_tags:
    #             inverse_mapping[t] = representing_tag
    #     return tag_post, inverse_mapping

    def get_mappings(
        self,
        threshold: float,
        out_dir: Optional[str] = None,
        verbose=False,
    ):
        # sim_matrix: candidate -> referrer
        sim_matrix = tag_similarity(
            self.tag_domain,
            self.ref_domain,
            self.bert_name,
        ) > threshold
        for i, row in enumerate(sim_matrix):
            if self.tag_domain is not self.ref_domain or sim_matrix[i, i]:
                # when tag_domain == ref_domain, skip the tag which has been represented by others
                sim_matrix[i + 1:, row] = False

        mapping = defaultdict(list)  # candidate -> referrer
        inverse_mapping = {}  # referrer -> candidate
        for represent_tag_id, tag_id in zip(*sim_matrix.nonzero()):
            mapping[self.ref_domain[represent_tag_id]].append(
                self.tag_domain[tag_id]
            )
            inverse_mapping[self.tag_domain[tag_id]] = \
                self.ref_domain[represent_tag_id]

        if out_dir is not None:
            out_path = os.path.join(
                out_dir, f'tag_ref_table_{threshold:.2f}.json'
            )
            with open(out_path, 'w', encoding='utf8') as fout:
                json.dump(mapping, fout, ensure_ascii=False, indent=4)

        if verbose:
            print(f'total unique tags = {len(self.tag_domain)}')
            print(f'total unique tags after post processing = {len(mapping)}')
            print(f'overlapping_rate = {len(self.tag_domain) / len(mapping)}')
        return mapping, inverse_mapping

    def post_process(
        self,
        tag_corpus: list[list[str]],
        threshold: float,
        out_dir: str = None,
        verbose=False,
    ):
        assert not isinstance(tag_corpus, dict)
        if threshold not in self.mappings_cache:
            self.mappings_cache[threshold] = self.get_mappings(
                threshold, out_dir, verbose
            )

        _tag_mapping, inverse_mapping = self.mappings_cache[threshold]

        def get_tags(tags: list[str]):
            return [inverse_mapping[t] for t in tags if t in inverse_mapping]

        new_tag_corpus = [get_tags(tags) for tags in tag_corpus]
        if out_dir is not None:

            def to_json(corpus: list):
                if isinstance(corpus[0], np.ndarray):
                    corpus = [tags.tolist() for tags in corpus]
                return json.dumps(corpus, ensure_ascii=False, indent=4)

            raw_tags_str = to_json(tag_corpus)
            sha256 = hashlib.sha256(raw_tags_str.encode()).hexdigest()[:6]

            out_dir = os.path.join(out_dir, f'enh_{sha256}')
            os.makedirs(out_dir, exist_ok=True)

            with open(
                os.path.join(out_dir, 'tag_corpus.json'), 'w', encoding='utf8'
            ) as fout:
                fout.write(raw_tags_str)

            with open(
                os.path.join(out_dir, 'enh_tag_corpus.json'), 'w',
                encoding='utf8'
            ) as fout:
                fout.write(to_json(new_tag_corpus))

        return new_tag_corpus


if __name__ == '__main__':
    RAW_PREDICT = 'results/ckpt-Llama-3-Taiwan-8B-Instruct-6/c4760502/'
    import data_loaders
    pp = PostProcessing.from_corpus(
        data_loaders.product_keywords(), bert_name='hfl/chinese-roberta-wwm-ext'
    )
    ref_domain, _ = pp.get_mappings(0.9, 'results')

    raw_raw_tags = load_raw_predict(
        os.path.join(RAW_PREDICT, 'raw_predict.json')
    )
    # pp = PostProcessing.from_corpus(raw_raw_tags, ref_domain=list(ref_domain))
    # tag_mapping, inverse_mapping = pp.get_mappings(0.8, RAW_PREDICT)
    # print(dict(list(tag_mapping.items())[:3]))
    # print(dict(list(inverse_mapping.items())[:3]))
