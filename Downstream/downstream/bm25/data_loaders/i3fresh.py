from typing import Literal, Callable, Optional, Union
from typing_extensions import deprecated
from itertools import zip_longest
import requests
import json
import os
from urllib.parse import urlparse
import hashlib
from functools import lru_cache
import zipfile
import pandas as pd
import numpy as np

__all__ = [
    'list_all_available_data',
    'set_password',
    'N_RPODUCTS',
    'AVAILABLE_PRODUCT_IDS',
    'PID_MAPPING',
    'load_df',
    'merged_df',
    # 'default_user_names',
    # 'product_tags_v4',
    # 'product_tags_v5_en',
    # 'product_tags_v5_zh',
    # 'product_keywords',
    # 'enhanced_tags_rev',
    'keywords_v2',
    'product_info',
    'base_tags',
]

_DATASET_URL = 'https://www.dropbox.com/scl/fi/ejkje0mv5wfcaanpgtzui/datasets_20240520.zip?rlkey=muu7ycdkbqyaxlcgfl6psomqe&st=qw5a4p94&dl=0'
_DATASET_URL_HASH = hashlib.md5(_DATASET_URL.encode()).hexdigest()


_CACHE_DIR = 'data/preprocessed/food/bm25'
N_RPODUCTS = 434
AVAILABLE_PRODUCT_IDS = [5, 23, 24, 25, 28, 46, 51, 56, 76, 77, 79, 81, 87, 93, 95, 107, 119, 133, 151, 173, 183, 189, 195, 212, 232, 236, 264, 282, 295, 320, 324, 345, 352, 353, 365, 373, 376, 380, 381, 383, 386, 387, 400, 401, 437, 449, 454, 455, 457, 459, 477, 499, 520, 526, 527, 530, 531, 534, 535, 554, 556, 565, 578, 579, 592, 596, 636, 644, 645, 646, 647, 649, 695, 701, 736, 737, 738, 740, 753, 765, 816, 818, 824, 835, 836, 890, 891, 925, 926, 949, 1014, 1026, 1034, 1036, 1037, 1039, 1065, 1075, 1097, 1098, 1099, 1100, 1120, 1121, 1123, 1125, 1147, 1195, 1197, 1198, 1210, 1216, 1217, 1232, 1233, 1234, 1235, 1236, 1237, 1240, 1242, 1285, 1404, 1418, 1419, 1437, 1440, 1441, 1442, 1452, 1453, 1454, 1455, 1467, 1491, 1494, 1495, 1496, 1497, 1499, 1512, 1515, 1516, 1539, 1575, 1681, 1686, 1687, 1688, 1700, 1701, 1719, 1722, 1729, 1746, 1758, 1767, 1799, 1816, 1857, 1858, 1859, 1860, 1861, 1864, 1893, 1906, 1914, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1997, 1999, 2001, 2005, 2019, 2027, 2048, 2049, 2077, 2083, 2124, 2125, 2239, 2240, 2254, 2265, 2266, 2270, 2312, 2477, 2495, 2570, 2572, 2574, 2576, 2577, 2678, 2679, 2712, 2782, 2799, 2803, 2823, 2832, 2840, 2851, 2886, 2887, 2903, 2907, 2917, 2918, 2947, 2955, 2988, 3021, 3039, 3062, 3227, 3232, 3233, 3239, 3304, 3306, 3319, 3329, 3340, 3346, 3362, 3366, 3367, 3369, 3373, 3395, 3400, 3409, 3427, 3431, 3489, 3521, 3522, 3523, 3524, 3525, 3530, 3532, 3534, 3536, 3537, 3538, 3539, 3540, 3542, 3543, 3545, 3546, 3547, 3549, 3551, 3568, 3650, 3663, 3680, 3728, 3794, 3798, 3812, 3818, 3819, 3820, 3839, 3843, 3864, 3883, 3901, 3915, 3932, 3943, 3972, 3976, 4001, 4026, 4031, 4036, 4043, 4045, 4047, 4054, 4055, 4056, 4059, 4060, 4070, 4072, 4075, 4083, 4092, 4102, 4103, 4125, 4139, 4140, 4141, 4144, 4180, 4181, 4183, 4194, 4196, 4213, 4214, 4216, 4227, 4230, 4231, 4232, 4255, 4261, 4267, 4293, 4302, 4322, 4345, 4356, 4413, 4422, 4473, 4475, 4478, 4479, 4489, 4517, 4518, 4526, 4536, 4538, 4540, 4545, 4554, 4558, 4580, 4589, 4591, 4593, 4594, 4628, 4631, 4634, 4635, 4636, 4648, 4649, 4650, 4651, 4652, 4670, 4675, 4705, 4761, 4762, 4763, 4764, 4777, 4862, 4866, 4884, 4943, 5033, 5034, 5044, 5079, 5177, 5231, 5293, 5306, 5307, 5309, 5325, 5428, 5431, 5436, 5437, 5446, 5489, 5510, 5580, 5612, 5619, 5651, 5658, 5670, 5677, 5693, 5694, 5710, 5741, 5750, 5771, 5834, 5837, 5841, 5844, 5850, 5852, 5861, 5871, 5887, 5914, 5927, 5947, 5949, 5983, 5997, 6022, 6029, 6030, 6088, 6099, 6100, 6104, 6106, 6108, 6110, 6112, 6115, 6118, 6121, 6123] # yapf: disable
PID_MAPPING = {pid: i for i, pid in enumerate(AVAILABLE_PRODUCT_IDS)}


@lru_cache(maxsize=4)
def _download(url: str, cache_dir: str):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_name = os.path.basename(urlparse(url).path)
    ext = os.path.splitext(file_name)[-1]
    assert ext in ('.json', '.pkl')
    file_path = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest())

    if not os.path.exists(file_path):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        res = requests.get(url, headers=headers)
        assert res.status_code == 200
        with open(file_path, 'wb') as fout:
            fout.write(res.content)

    if ext == '.json':
        with open(file_path, 'r', encoding='utf8') as fin:
            return json.load(fin)
    else:
        assert ext == '.pkl'
        return pd.read_pickle(file_path)


def _download_datasets(cache_hashed_dir: str):
    os.makedirs(cache_hashed_dir, exist_ok=True)
    file_name = os.path.basename(urlparse(_DATASET_URL).path)
    file_path = os.path.join(cache_hashed_dir, file_name)

    if not os.path.exists(file_path):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        res = requests.get(_DATASET_URL, headers=headers)
        assert res.status_code == 200
        with open(file_path, 'wb') as fout:
            fout.write(res.content)

    with zipfile.ZipFile(file_path, mode='r') as zipfin:
        pwd = getattr(_download_datasets, 'pwd', None)
        if pwd is None:
            pwd = input(
                'You have not set the password for the dataset, '
                'either set the password using final.data.setpassword(<pwd>) '
                'or prompt the password here: '
            )
        zipfin.setpassword(pwd.encode())
        zipfin.extractall(cache_hashed_dir)


@lru_cache(maxsize=2)
def _get_file(file_name: str, cache_dir):
    # Use ../data/pkl instead of hash-based directory
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to src/tmp/final/data -> src/tmp/final -> src/tmp -> src -> PseudoUser -> data/pkl
    data_pkl_dir = os.path.join(current_dir, '..', '..', '..', '..', 'data', 'pkl')
    file_path = os.path.join(data_pkl_dir, file_name)
    assert os.path.isfile(file_path), (
        f'Cannot find "{file_path}". Please download the dataset manually.'
    )
    # if not os.path.isfile(file_path):
    #     _download_datasets(cache_dir)
    ext = os.path.splitext(file_name)[-1]
    assert ext in ('.json', '.pkl')
    if ext == '.json':
        with open(file_path, 'r', encoding='utf8') as fin:
            return json.load(fin)
    else:
        assert ext == '.pkl'
        return pd.read_pickle(file_path)


def _create_split(
    merged_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
):

    total_time_interval = merged_df['timestamp'].max(
    ) - merged_df['timestamp'].min()
    train_stamp = merged_df['timestamp'].min() + int(
        total_time_interval * train_ratio
    )
    val_stamp = merged_df['timestamp'].min() + int(
        total_time_interval * (train_ratio + val_ratio)
    )

    train_idx = merged_df['timestamp'] <= train_stamp
    val_idx = (~train_idx) & (merged_df['timestamp'] <= val_stamp)
    test_idx = merged_df['timestamp'] > val_stamp

    return {
        'train': merged_df[train_idx],  #.reset_index(drop=True),
        'val': merged_df[val_idx],  #.reset_index(drop=True),
        'test': merged_df[test_idx],  #.reset_index(drop=True),
    }


def set_password(password: str):
    _download_datasets.pwd = password
    return


def inters_df(
    cache_dir: str = _CACHE_DIR,
    split: Optional[Literal['train', 'val', 'test', 'all']] = None,
    use_iid: bool = False,
):
    df = load_df(cache_dir, auto_split=False)
    df = df[df['loaded_pids'].map(len) >= 3]

    if split is None:
        return {
            spt: inters_df(cache_dir, split=spt)
            for spt in ['train', 'val', 'test']
        }
    if split == 'all':
        assert use_iid is False
        return df

    loaded_pids = df['loaded_pids']
    if use_iid:
        pid_to_iid = {pid: iid for iid, pid in enumerate(all_pids(cache_dir))}
        loaded_pids = loaded_pids.map(
            lambda seq: [pid_to_iid[pid] for pid in seq]
        )

    # Create result dataframe with 'u' column for user/session ID
    result_df = loaded_pids.to_frame()
    result_df['u'] = result_df.index

    if split == 'train':
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[:-2])
    elif split == 'val':
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-2:-1])
    else:
        assert split == 'test'
        result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-1:])

    return result_df


@lru_cache(maxsize=1)
def load_df(
    cache_dir: str = _CACHE_DIR, auto_split: bool = True,
    load_raw: bool = False
) -> pd.DataFrame:
    FILE_NAME = 'load_df_v3.pkl'
    CACHE_GROUPED_DF = os.path.join(cache_dir, 'load_df_v3_grouped.pkl')

    load_df = _get_file(FILE_NAME, cache_dir)
    if load_raw:
        if not auto_split:
            return load_df
        merged_dfs = merged_df()
        val_start_stamp = merged_dfs['val']['session_id'].min()
        test_start_stamp = merged_dfs['test']['session_id'].min()
        load_dfs = {}
        load_dfs['train'] = load_df[load_df['session_id'] < val_start_stamp]
        load_dfs['val'] = load_df[(load_df['session_id'] >= val_start_stamp)
                                  & (load_df['session_id'] < test_start_stamp)]
        load_dfs['test'] = load_df[load_df['session_id'] >= test_start_stamp]
        return load_dfs

    def concat(s: pd.Series):
        if len(s) > 1:
            s = s.sort_values(by=['timestamp'])
        pids = s['item_id']
        return [
            pid for pid, shift in zip_longest(pids, pids[1:]) if pid != shift
        ]

    if os.path.isfile(CACHE_GROUPED_DF):
        load_df = pd.read_pickle(CACHE_GROUPED_DF)
    else:
        load_df: pd.DataFrame = load_df.groupby(['session_id']
                                                )[['timestamp', 'item_id'
                                                   ]].apply(concat).to_frame()

        load_df = load_df.rename(columns={0: 'loaded_pids'})
        load_df.to_pickle(CACHE_GROUPED_DF)
    if not auto_split:
        return load_df
    merged_dfs = merged_df(cache_dir=cache_dir)
    val_start_stamp = merged_dfs['val']['session_id'].min()
    test_start_stamp = merged_dfs['test']['session_id'].min()

    load_dfs = {}
    load_dfs['train'] = load_df[load_df.index < val_start_stamp]
    load_dfs['val'] = load_df[(load_df.index >= val_start_stamp)
                              & (load_df.index < test_start_stamp)]
    load_dfs['test'] = load_df[load_df.index >= test_start_stamp]

    return load_dfs


@deprecated(
    'merged_df only include sessions with placed-puchase. We use load_df now.'
)
def merged_df(
    auto_split: bool = True, cache_dir: str = _CACHE_DIR
) -> Union[dict[Literal['train', 'val', 'test'], pd.DataFrame], pd.DataFrame]:
    FILE_NAME = 'merged_df_v3.pkl'
    merged_df: pd.DataFrame = _get_file(FILE_NAME, cache_dir)
    merged_df = merged_df.set_index('order_id')
    merged_df.loc[:, 'loaded_pids'] = merged_df['loads'].map(
        lambda loads: [load['item_id'] for load in loads]
    )
    merged_df = merged_df[merged_df['loaded_pids'].map(len).astype(bool)]

    def products_set_to_array(products: set[int]):
        a = np.zeros(N_RPODUCTS, dtype=int)
        a[[PID_MAPPING[pid] for pid in products]] = 1
        return a

    y_true = merged_df['products'].map(products_set_to_array)
    merged_df.loc[:, 'y_true'] = y_true
    if auto_split:
        return _create_split(merged_df)
    return merged_df


def keywords_v2(cache_dir: str = _CACHE_DIR, use_pid: bool = False):
    path = os.path.join(cache_dir, 'keywords_v2.json')
    assert os.path.isfile(path), os.path.abspath(path)
    with open(path, 'r', encoding='utf8') as fin:
        d = json.load(fin)
    d = {int(pid): tags for pid, tags in d.items()}
    if use_pid:
        return d
    return [d[pid] for pid in AVAILABLE_PRODUCT_IDS]


def base_tags(cache_dir: str = _CACHE_DIR, use_pid: bool = False):
    path = os.path.join(cache_dir, 'base_tags.json')
    assert os.path.isfile(path), os.path.abspath(path)
    with open(path, 'r', encoding='utf8') as fin:
        d = json.load(fin)
    d = {int(pid): tags for pid, tags in d.items()}
    if use_pid:
        return d
    return [d[pid] for pid in AVAILABLE_PRODUCT_IDS]


def product_info(cache_dir: str = _CACHE_DIR, use_pid: bool = False):
    path = os.path.join(cache_dir, 'product_info_ex.json')
    assert os.path.isfile(path), os.path.abspath(path)
    with open(path, 'r', encoding='utf8') as fin:
        d = json.load(fin)
    d = {int(pid): tags for pid, tags in d.items()}
    if use_pid:
        return d
    return [d[pid] for pid in AVAILABLE_PRODUCT_IDS]


def all_pids(cache_dir: str = _CACHE_DIR):
    return AVAILABLE_PRODUCT_IDS


def list_all_available_data():
    return __all__[2:]


def check_integrity(cache_dir: str = _CACHE_DIR):
    for name in list_all_available_data():
        fn = globals()[name]
        if isinstance(fn, Callable):
            fn(cache_dir)
    return None


class _VariantHelper:
    """Helper to load from variant-specific directories like Leave_one_last_item"""

    def __init__(self, variant_name: str):
        self.variant_name = variant_name
        return

    def inters_df(
        self,
        cache_dir: str = _CACHE_DIR,
        split: Optional[str] = None,
        use_iid: bool = False,
    ):
        """Load interactions from variant directory (e.g., Leave_one_last_item)"""
        # Map variant name to directory name
        if self.variant_name == 'leave_one_last_item':
            dir_name = '.'
        else:
            dir_name = self.variant_name.capitalize()

        # Load full sequences from sequences.json
        sequences_file = os.path.join(cache_dir, dir_name, 'sequences.json')
        with open(sequences_file, 'r', encoding='utf8') as f:
            sequences = json.load(f)

        # Convert to DataFrame
        full_sequences = {str(user_id): pids for user_id, pids in sequences.items()}
        df = pd.DataFrame({
            'loaded_pids': list(full_sequences.values())
        }, index=list(full_sequences.keys()))

        # Filter to sequences with at least 3 items
        df = df[df['loaded_pids'].map(len) >= 3]

        if split is None:
            return {
                'train': self.inters_df(cache_dir, split='train', use_iid=use_iid),
                'val': self.inters_df(cache_dir, split='val', use_iid=use_iid),
                'test': self.inters_df(cache_dir, split='test', use_iid=use_iid),
            }

        if split == 'all':
            assert use_iid is False
            return df

        loaded_pids = df['loaded_pids']
        if use_iid:
            pid_to_iid = {pid: iid for iid, pid in enumerate(self.all_pids(cache_dir))}
            loaded_pids = loaded_pids.map(
                lambda seq: [pid_to_iid[pid] for pid in seq]
            )

        # Create result dataframe with 'u' column for user/session ID
        result_df = loaded_pids.to_frame()
        result_df['u'] = result_df.index

        # Perform leave-one-last-item split at load time
        if split == 'train':
            result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[:-2])
        elif split == 'val':
            result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-2:-1])
        else:
            assert split == 'test'
            result_df['loaded_pids'] = result_df['loaded_pids'].map(lambda seq: seq[-1:])

        return result_df

    def all_pids(self, cache_dir: str = _CACHE_DIR):
        """Load all product IDs from smap.json"""
        if self.variant_name == 'leave_one_last_item':
            dir_name = '.'
        else:
            dir_name = self.variant_name.capitalize()

        with open(
            os.path.join(cache_dir, dir_name, 'smap.json'),
            encoding='utf8'
        ) as fin:
            return list(json.load(fin).values())

    def base_tags(self, cache_dir: str = _CACHE_DIR, use_pid: bool = False):
        """Load base tags from base_tags.json"""
        pids = self.all_pids(cache_dir)
        if self.variant_name == 'leave_one_last_item':
            dir_name = '.'
        else:
            dir_name = self.variant_name.capitalize()

        with open(
            os.path.join(cache_dir, dir_name, 'base_tags.json'),
            encoding='utf8'
        ) as fin:
            d = {int(pid): tags for pid, tags in json.load(fin).items()}

        if use_pid:
            return {pid: d[pid] for pid in pids}
        return [d[pid] for pid in pids]

    def __getattr__(self, __name: str):
        fn = globals().get(__name, None)
        assert fn is not None, f'data: {__name} for i3fresh does not exist'
        return fn


leave_one_last_item = _VariantHelper('leave_one_last_item')
loli = leave_one_last_item
