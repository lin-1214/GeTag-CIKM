import copy
from collections import Counter
import numpy as np

CorpusT = list[list[str] | np.ndarray]


def mask_tags(
    tag_corpus: CorpusT,
    parse_n_tags: int | None = None,
    skip_first_n: int | None = None,
    mask: slice | None = None,
):
    if mask is not None:
        assert parse_n_tags is None and skip_first_n is None
    else:
        skip_first_n = skip_first_n or 0
        assert parse_n_tags is not None
        mask = slice(skip_first_n, skip_first_n + parse_n_tags)

    return [np.array(tags)[mask] for tags in tag_corpus]


def add_id_tag(tag_corpus: CorpusT, count: int = 1):
    if count == 0:
        return tag_corpus
    tag_corpus = copy.deepcopy(tag_corpus)
    for iid, tags in enumerate(tag_corpus):
        tags.extend([str(iid)] * count)
    return tag_corpus


def to_lower(tag_corpus: CorpusT):
    return [[t.lower() for t in tags] for tags in tag_corpus]


def to_domain(tag_corpus: CorpusT, sort: bool = False):
    if isinstance(tag_corpus[0], np.ndarray):
        tag_corpus = [tags.tolist() for tags in tag_corpus]
    counter = Counter(sum(tag_corpus, []))
    counts = np.array(list(counter.values()))
    tags_domain = np.array(list(counter))
    if sort:
        tags_domain = tags_domain[np.argsort(-counts)]
    return tags_domain


def statistics(tag_corpus: CorpusT):

    def density():
        """Density of tag-item graph"""
        domain = set()
        n = 0
        for tags in tag_corpus:
            tags = set(tags)
            domain |= tags
            n += len(tags)
        return n / (len(domain) * len(tag_corpus))

    adj = to_graph(tag_corpus)

    # item->tag

    def avg_items_per_tag():
        return adj.sum(axis=0).mean()

    def avg_tags_per_item():
        return adj.sum(axis=1).mean()

    return {
        'density': density(),
        'avg_items_per_tag': avg_items_per_tag(),
        'avg_tags_per_item': avg_tags_per_item(),
    }


def to_graph(tag_corpus: CorpusT):
    """

    Args:
        tag_corpus (dict[int, list[str]]): _description_

    Returns:
        adj: item->tag
    """
    from scipy import sparse as sp
    domain = to_domain(tag_corpus)
    tag_to_tid = {tag: i for i, tag in enumerate(domain)}
    uv = np.array(
        [
            (iid, tag_to_tid[tag]) for iid, tags in enumerate(tag_corpus)
            for tag in tags
        ]
    )  # (|E|, 2)
    return sp.coo_matrix((np.ones(len(uv)), (uv[:, 0], uv[:, 1])))
