import datetime
import os
import torch

class Config:
    MODEL_NAME = "Qwen/Qwen3-4B"
    
    # ====== Data Paths ======
    # Domain-specific data paths (auto-selected based on DATA_DOMAIN env var)
    _DATA_PATH_MAP = {
        "food": "../data/food/food_commerce_data_labeling.csv",
        "amazon": "../data/amazon/amazon_sessions_labeling.csv",
        "yelp": "../data/yelp/yelp_sessions_labeling.csv",
    }
    DOMAIN = os.getenv("DATA_DOMAIN", "yelp")  # "food" | "amazon" | "yelp"
    INCLUDE_TAG = os.getenv("INCLUDE_TAG", "base")  # "native" | "base"
    PREPROCESSED_DATA_PATH = _DATA_PATH_MAP.get(DOMAIN, "../data/yelp/yelp_sessions_labeling.csv")
    USER_SESSION_DATA_PATH = '../label_data'
    # CLASSIFIED_DATA_PATH now includes domain and tag for easy identification
    CLASSIFIED_DATA_PATH = f'../label_data/classified_data_{DOMAIN}_{INCLUDE_TAG}'

    LOCAL_ROOT_PATH = 'checkpoints'
    CLLM_DATA_PATH = '../cllm_data'
    CLLM_PROCESSED_DATA_PATH = 'user_session_data'

    T_CLUSTERS   = 20
    T_VARIANCE   = 0
    GPU_INDEX    = 0
    MAX_NEW_TOK  = 1024

    # Dynamic augmentation parameters
    INITIAL_K_AUG = 1  # Initial number of augmentations
    TOP_K_PROMPTS = 1  # Number of top prompts to keep
    K_AUG = 1  # Will be dynamically adjusted
    
    SAMPLE_STEP  = 25 # 4900/25=196 
    SEED         = 42

    SAMPLE_START_INDEX = 0  # The starting row index for sampling
    SAMPLE_SIZE = 4900
    BATCH_SIZE = 4

    # Multi-beam classification parameters
    MULTIBEAM_K = 5                # 每个 session 分类次数
    MULTIBEAM_TEMPERATURE = 1.5     # 提高多样性 (1.0 太低，建议 1.5-2.0)
    MULTIBEAM_MIN_FREQ = 0.40       # 最低频率阈值（40% = 2/5次，排除只出现1次的标签）
    MULTIBEAM_TOP_N = 5             # 最多保留几个标签
    
    # Session length filtering (for movie dataset to avoid token overflow)
    MAX_SESSION_LENGTH = 100  # Maximum number of events per session (None = no limit)
    SAFE_GLOBALS = {"datetime": datetime, "__builtins__": {}}

    MAX_RETRIES = 5

    # Multi-label classification controls
    MULTI_LABEL = True            # Set True to enable multi-label classification
    MAX_LABELS_PER_SESSION = 5     # Upper bound on labels per session when multi-label is enabled
    MIN_LABELS_PER_SESSION = 1     # Lower bound; keep 1 for at-least-one label
    LABEL_SEPARATOR = ";"         # If model returns string, split on this; prefer JSON array

    LAMBDA_V = 0.1
    CUDA_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
    DEVICE = f"cuda:{GPU_INDEX}" if CUDA_AVAILABLE else "cpu"

    EXPLORE_PARAM = 1.414

    SHARE_BASE_MODEL = True
    MODEL_TYPE = "rec"


    def __init__(self):
        """
        Initialize Config class and verify/create necessary directories.
        Creates directories if they don't exist.
        """
        paths = [
            self.USER_SESSION_DATA_PATH,
            self.CLLM_DATA_PATH,
            self.LOCAL_ROOT_PATH
        ]
        
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")

        # Check if the preprocessed data file's directory exists
        preproc_dir = os.path.dirname(self.PREPROCESSED_DATA_PATH)
        if preproc_dir and not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir, exist_ok=True)
            print(f"Created directory: {preproc_dir}")

        # Auto-detect domain from data path if not explicitly set
        include_tag_normalized = (self.INCLUDE_TAG or "none").strip().lower().replace(" ", "_")
        if include_tag_normalized in {"base", "base_tag", "base_tags"}:
            self.INCLUDE_TAG = "base"
        elif include_tag_normalized in {"native", "native_tag", "keyword", "keywords"}:
            self.INCLUDE_TAG = "native"
        elif include_tag_normalized in {"betag", "be_tag", "betags", "be_tags"}:
            self.INCLUDE_TAG = "betag"
        else:
            self.INCLUDE_TAG = "none"

        data_path_lower = self.PREPROCESSED_DATA_PATH.lower()
        if "movie" in data_path_lower:
            self.DOMAIN = "movie"
        elif "food" in data_path_lower or "commerce" in data_path_lower:
            self.DOMAIN = "food"
        else:
            # keep env-provided value
            self.DOMAIN = self.DOMAIN.lower()


    def check_overall_config(self):
        if self.CUDA_AVAILABLE:
            print(f"CUDA is available. Found {torch.cuda.device_count()} device(s)")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            print("CUDA is not available. Using CPU")
            
        print("\n\n-----Overall Config-----\n\n")
        print(f"MODEL_NAME: {self.MODEL_NAME}")
        print(f"PREPROCESSED_DATA_PATH: {self.PREPROCESSED_DATA_PATH}")
        print(f"USER_SESSION_DATA_PATH: {self.USER_SESSION_DATA_PATH}")
        print(f"CLASSIFIED_DATA_PATH: {self.CLASSIFIED_DATA_PATH}")
        print(f"CLLM_DATA_PATH: {self.CLLM_DATA_PATH}")
        print(f"CLLM_PROCESSED_DATA_PATH: {self.CLLM_PROCESSED_DATA_PATH}")
        print(f"T_CLUSTERS: {self.T_CLUSTERS}")
        print(f"GPU_INDEX: {self.GPU_INDEX}")
        print(f"MAX_NEW_TOK: {self.MAX_NEW_TOK}")
        print(f"K_AUG: {self.K_AUG}")
        print(f"SAMPLE_STEP: {self.SAMPLE_STEP}")
        print(f"SEED: {self.SEED}")
        print(f"SAMPLE_SIZE: {self.SAMPLE_SIZE}")
        print(f"BATCH_SIZE: {self.BATCH_SIZE}")
        print(f"SAFE_GLOBALS: {self.SAFE_GLOBALS}")
        print(f"MAX_RETRIES: {self.MAX_RETRIES}")
        print(f"LAMBDA_V: {self.LAMBDA_V}")
        print(f"DEVICE: {self.DEVICE}")
        # print(f"NUM_TRAIN_EPOCHS: {self.NUM_TRAIN_EPOCHS}")
        # print(f"NUM_PRETRAIN_EPOCHS: {self.NUM_PRETRAIN_EPOCHS}")
        # print(f"NUM_FINETUNE_EPOCHS: {self.NUM_FINETUNE_EPOCHS}")

        print("\n\n-----End of Config-----\n\n")



