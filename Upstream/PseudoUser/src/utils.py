import os
import json
import random
import numpy as np
import pandas as pd

from config import Config
from openai import OpenAI  # pip install openai>=1.0

config = Config()

class Utils:
    # -------------------------------
    # 基本工具
    # -------------------------------
    def __init__(self):
        base_url = getattr(config, "VLLM_BASE_URL", os.getenv("VLLM_BASE_URL", "http://localhost:1357/v1"))
        model = getattr(config, "VLLM_MODEL", os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B"))

        # vLLM 的 OpenAI 相容端點不會驗 API Key，但欄位要有值
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("VLLM_API_KEY", "DUMMY"))
        self.model = model

        # 推論設定：盡量沿用你原本的溫度/長度
        self.max_new_tokens = getattr(config, "MAX_NEW_TOK", 1024)
        self.temp_gen = 1.0   # 你原本 generate/augment 用 0.7
        self.temp_cls = 0.0   # 分類/結構化建議溫度 0（提高 JSON 穩定性）

        # 如需可重現性，vLLM 支援 seed
        self.seed = getattr(config, "SEED", None)

        # Paths for tag resources (domain-aware)
        self.tag_source = getattr(config, "INCLUDE_TAG", "none")
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._name_to_id = {}
        self._id_to_name = {}
        self._base_tags = {}
        self._keyword_tags = {}
        self._domain = getattr(config, "DOMAIN", "food").lower()

        # Always load product mappings for formatting sessions (even if tags disabled)
        tags_dir = os.path.join(self._project_root, "json", "tags")
        if self._domain == "movie":
            mapping_path = os.path.join(self._project_root, "json", "movie_name_to_id.json")
            reverse_path = os.path.join(self._project_root, "json", "movie_id_to_name.json")
            base_tags_path = os.path.join(tags_dir, "movie_basetag.json")
            native_tags_path = os.path.join(tags_dir, "movie_native.json")
            betag_tags_path = os.path.join(tags_dir, "movie_betags.json")
        elif self._domain == "amazon":
            # Amazon domain: CSV has ASINs, need to map ASIN → PID for tags
            mapping_path = os.path.join(self._project_root, "json", "amazon_title_to_asin.json")
            reverse_path = os.path.join(self._project_root, "json", "amazon_asin_to_title.json")
            # For Amazon tags, we need ASIN → PID mapping
            amazon_mapping_path = os.path.join(tags_dir, "amazon_mapping.json")
            base_tags_path = os.path.join(tags_dir, "amazon_basetag.json")
            native_tags_path = os.path.join(tags_dir, "amazon_native.json")
            betag_tags_path = os.path.join(tags_dir, "amazon_betags.json")
        elif self._domain == "yelp":
            # Yelp domain: CSV has business_ids, need to map business_id → internal_id → name
            mapping_path = os.path.join(self._project_root, "json", "yelp_name_to_id.json")
            reverse_path = os.path.join(self._project_root, "json", "yelp_id_to_name.json")
            base_tags_path = os.path.join(tags_dir, "yelp_basetag.json")
            native_tags_path = os.path.join(tags_dir, "yelp_native.json")
            betag_tags_path = os.path.join(tags_dir, "yelp_betags.json")
        else:  # food domain
            mapping_path = os.path.join(self._project_root, "json", "product_name_to_id.json")
            reverse_path = os.path.join(self._project_root, "json", "product_id_to_name.json")
            base_tags_path = os.path.join(tags_dir, "food_basetag.json")
            native_tags_path = os.path.join(tags_dir, "food_native.json")
            betag_tags_path = os.path.join(tags_dir, "food_betags.json")

        self._name_to_id = self._load_json_file(mapping_path)
        self._id_to_name = self._load_json_file(reverse_path)
        
        # For Amazon, create reverse mapping: ASIN → PID
        if self._domain == "amazon":
            amazon_mapping = self._load_json_file(amazon_mapping_path)
            # amazon_mapping is {PID: ASIN}, we need {ASIN: PID}
            self._asin_to_pid = {asin: pid for pid, asin in amazon_mapping.items()}
        else:
            self._asin_to_pid = {}
        
        if self.tag_source != "none":
            if self.tag_source == "base" and base_tags_path:
                self._base_tags = self._load_json_file(base_tags_path)
            elif self.tag_source == "native" and native_tags_path:
                self._keyword_tags = self._load_json_file(native_tags_path)
            elif self.tag_source == "betag" and betag_tags_path:
                self._keyword_tags = self._load_json_file(betag_tags_path)

    def set_seed(self):
        random.seed(config.SEED)
        np.random.seed(config.SEED)

    def sample_session(self, df, n = None):
        if n is None:
            n = config.SAMPLE_STEP
        idx = [random.randint(i, min(i + n - 1, len(df) - 1)) for i in range(0, len(df), n)]
        return df.iloc[idx]

    # Apply modified Sparrow-version（原樣保留）
    def format_session(self, session_row, include_tags: bool = False):
        domain = getattr(config, "DOMAIN", "food").lower()
        ecommerce_actions = []
        movie_actions = []
        amazon_asins = []
        yelp_business_ids = []
        
        for cell in session_row:
            if pd.isna(cell):
                continue
            cell_str = str(cell).strip()
            if not cell_str:
                continue
                
            # Amazon domain: plain ASIN strings
            if domain == "amazon":
                # ASIN format: alphanumeric, typically 10 chars (like B000FT9KTS)
                if cell_str and len(cell_str) >= 8:
                    amazon_asins.append(cell_str)
                continue
            
            # Yelp domain: plain business_id strings
            if domain == "yelp":
                # business_id format: 22-character alphanumeric with hyphens
                if cell_str and len(cell_str) >= 20:
                    yelp_business_ids.append(cell_str)
                continue
                
            # For movie/food: try parsing structured data
            try:
                parsed = eval(cell, config.SAFE_GLOBALS)
                if isinstance(parsed, list):
                    ecommerce_actions.append(parsed)
                elif isinstance(parsed, tuple):
                    movie_actions.append(parsed)
            except Exception:
                normalized = self._normalize_product_name(cell)
                if normalized:
                    session_id = getattr(session_row, "name", "session")
                    ecommerce_actions.append([session_id, None, "load", normalized])
                continue
                
        # Amazon domain rendering
        if domain == "amazon":
            if not amazon_asins:
                return "Empty session"
            session_id = getattr(session_row, "name", "session")
            lines = [f"Session {session_id}:"]
            for asin in amazon_asins:
                # Look up title from ASIN
                title = self._id_to_name.get(asin, asin)
                lines.append(f"- viewed '{title}'")
            if len(lines) == 1:
                return "Empty session"
            # Add tags if requested
            if include_tags:
                # For Amazon, we need to pass ASINs (not titles) to collect tags
                amazon_events = [(None, None, None, asin) for asin in amazon_asins]
                session_tags = self._collect_session_tags(amazon_events)
                if session_tags:
                    lines.append("")
                    lines.append(f"Relevant tags: {', '.join(session_tags)}")
            return "\n".join(lines)
            
        # Yelp domain rendering
        if domain == "yelp":
            if not yelp_business_ids:
                return "Empty session"
            session_id = getattr(session_row, "name", "session")
            lines = [f"Session {session_id}:"]
            for business_id in yelp_business_ids:
                # Look up internal_id from business_id, then get name
                internal_id = self._name_to_id.get(business_id)
                if internal_id is not None:
                    name = self._id_to_name.get(str(internal_id), business_id)
                else:
                    name = business_id
                lines.append(f"- visited '{name}'")
            if len(lines) == 1:
                return "Empty session"
            # Add tags if requested
            if include_tags:
                # For Yelp, we need to pass business_ids to collect tags
                yelp_events = [(None, None, None, bid) for bid in yelp_business_ids]
                session_tags = self._collect_session_tags(yelp_events)
                if session_tags:
                    lines.append("")
                    lines.append(f"Relevant tags: {', '.join(session_tags)}")
            return "\n".join(lines)
            
        # Movie domain rendering
        if domain == "movie":
            if not movie_actions and ecommerce_actions:
                # Fallback in case data already formatted as list
                movie_actions = ecommerce_actions
            if not movie_actions:
                return "Empty session"
            session_id = getattr(session_row, "name", None)
            if session_id is None:
                session_id = "movie_session"
            lines = [f"Session {session_id}:"]
            collected_actions = []
            for event in movie_actions:
                # Allow both tuple (datetime, rating, title) and list variants
                if isinstance(event, (tuple, list)) and len(event) >= 3:
                    ts_val, rating, title = event[0], event[1], event[2]
                else:
                    continue
                if hasattr(ts_val, "strftime"):
                    ts_str = ts_val.strftime("%H:%M:%S")
                else:
                    ts_str = str(ts_val)
                    if len(ts_str) >= 8 and ts_str[10:18].count(":") == 2:
                        ts_str = ts_str[11:19]
                lines.append(f"- {ts_str}: rated {rating}/5 '{title}'")
                collected_actions.append(event)
            if len(lines) == 1:
                return "Empty session"
            if include_tags:
                session_tags = self._collect_session_tags(collected_actions)
                if session_tags:
                    lines.append("")
                    lines.append(f"Relevant tags: {', '.join(session_tags)}")
            return "\n".join(lines)
            
        # Default e-commerce (food) rendering
        if not ecommerce_actions:
            return "Empty session"
        session_id_fallback = getattr(session_row, "name", "session")
        device_id = ecommerce_actions[0][0] if ecommerce_actions[0][0] else session_id_fallback
        lines = [f"Session {device_id}:"]
        for action in ecommerce_actions:
            if not isinstance(action, (list, tuple)) or len(action) < 4:
                continue
            _, _, act, prod = action
            act = act or "load"
            prod = self._normalize_product_name(prod)
            if not prod:
                continue
            # Drop timestamps to reduce prompt length; only keep action + product
            lines.append(f"- {act} '{prod}'")
        if include_tags:
            session_tags = self._collect_session_tags(ecommerce_actions)
            if session_tags:
                lines.append("")
                lines.append(f"相關標籤: {', '.join(session_tags)}")
        return "\n".join(lines)

    def _load_json_file(self, path: str):
        """
        安全地載入 JSON 檔案，如果不存在或解析失敗就回傳空 dict，避免中斷流程。
        """
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except FileNotFoundError:
            print(f"[WARN] JSON resource not found: {path}")
        except json.JSONDecodeError as exc:
            print(f"[WARN] Failed to parse JSON resource {path}: {exc}")
        return {}

    def _normalize_product_name(self, name):
        if name is None:
            return None
        normalized = str(name).strip()
        # 移除多餘的全形空白或不可見字元
        normalized = normalized.replace("\u3000", "").strip()
        return normalized if normalized else None

    def _get_tags_for_product(self, name: str):
        """
        取得單一商品/電影的標籤清單，依照設定來源選擇使用 base tags 或 native(keyword) tags。
        
        For Amazon: name is ASIN (from CSV) → get PID → get tags
        For Movie/Food: name is title/product name → get ID → get tags
        """
        if not name or self.tag_source == "none":
            return []

        domain = getattr(config, "DOMAIN", "food").lower()
        
        # For Amazon, special handling: ASIN → PID
        if domain == "amazon":
            asin = str(name).strip()
            # Get PID from ASIN
            pid = self._asin_to_pid.get(asin)
            if pid is None:
                return []
            key = str(pid)
        # For Yelp, special handling: business_id → internal_id
        elif domain == "yelp":
            business_id = str(name).strip()
            if not business_id:
                return []
            # Get internal_id from business_id
            internal_id = self._name_to_id.get(business_id)
            if internal_id is None:
                return []
            key = str(internal_id)
        # For Movie
        elif domain == "movie":
            normalized = str(name).strip()
            if not normalized:
                return []
            entity_id = self._name_to_id.get(normalized)
            if entity_id is None:
                return []
            key = str(entity_id)
        # For Food (default)
        else:
            normalized = self._normalize_product_name(name)
            if not normalized:
                return []
            entity_id = self._name_to_id.get(normalized)
            if entity_id is None:
                return []
            key = str(entity_id)

        if self.tag_source == "base":
            tags = self._base_tags.get(key, [])
        elif self.tag_source == "native":
            tags = self._keyword_tags.get(key, [])
        elif self.tag_source == "betag":
            tags = self._keyword_tags.get(key, [])
        else:
            tags = []
        return tags if isinstance(tags, list) else []

    def _collect_session_tags(self, actions):
        """
        聚合 session 內所有物件（商品、電影、餐廳）的標籤。
        食品 domain: 每個商品取前三個標籤；
        電影 domain: 取所有標籤（避免重複，保留順序）。
        Amazon/Yelp domain: 每個項目取前三個標籤。
        """
        if self.tag_source == "none":
            return []

        domain = getattr(config, "DOMAIN", "food").lower()
        if not actions:
            return []

        collected_tags = []
        seen_entities = set()
        ordered_entities = []

        for action in actions:
            if not isinstance(action, (list, tuple)):
                continue

            if domain == "movie":
                if len(action) < 3:
                    continue
                entity_name = action[-1]
            else:
                if len(action) < 4:
                    continue
                entity_name = action[3]

            normalized_name = (
                str(entity_name).strip() if domain == "movie"
                else self._normalize_product_name(entity_name)
            )
            if not normalized_name or normalized_name in seen_entities:
                continue
            seen_entities.add(normalized_name)
            ordered_entities.append(normalized_name)

        for name in ordered_entities:
            tags = self._get_tags_for_product(name)
            if not tags:
                continue
            limited = [tag for tag in tags if tag][:3]
            if not limited:
                continue
            if domain == "movie":
                for tag in limited:
                    if tag not in collected_tags:
                        collected_tags.append(tag)
            else:
                collected_tags.extend(limited)

        if domain == "movie" and self.tag_source != "none" and actions and not collected_tags:
            print("[WARN] No movie tags found for session despite tag source being enabled.")

        return collected_tags

    # -------------------------------
    # vLLM Chat 呼叫（共用私函式）
    # -------------------------------
    def _chat(self, system_text: str, user_text: str, temperature: float, seed=None):
        """
        封裝一次 chat 製作，回傳純文字 content。
        使用 vLLM OpenAI 相容 API，由 vLLM 代管 chat template。
        
        Args:
            seed: 如果为 None，则不设置 seed（用于 multi-beam 增加随机性）
                  如果未指定，默认使用 self.seed
        """
        if seed is None:
            actual_seed = None
        elif seed == "default":
            actual_seed = self.seed
        else:
            actual_seed = seed
            
        extra = {}
        if "qwen" in self.model.lower():
            extra["chat_template_kwargs"] = {"enable_thinking": False}

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": user_text},
            ],
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            seed=actual_seed,              # vLLM 支援
            extra_body=extra if extra else None,
            # 需要可以加入 stop=[] 做結尾約束
        )
        return resp.choices[0].message.content.strip()

    # -------------------------------
    # 生成/增強/分類 三個對 LLM 的呼叫
    # -------------------------------
    def augment_data(self, augmentation_prompt: str) -> str:
        """
        你原本用模型把 meta 做 paraphrase/augmentation。
        這裡改成呼叫 vLLM chat。
        """
        system = "You are a helpful assistant that rewrites prompts."
        user = augmentation_prompt
        return self._chat(system, user, temperature=self.temp_gen, seed="default")

    def generate_cluster(self, cluster_prompt: str) -> str:
        """
        你原本讓模型產生 clusters（文字/JSON 都可能）。
        這裡同樣用 chat；溫度保持 0.7。
        """
        domain = getattr(config, "DOMAIN", "food").lower()
        if domain == "movie":
            system = "You are an analyst who excels at clustering movie viewing and rating sessions."
        else:
            system = "You are an analyst excel in clustering shopping sessions."
        user = cluster_prompt
        return self._chat(system, user, temperature=self.temp_gen, seed="default")

    def predict_cluster(self, classification_prompt: str, temperature: float = None, use_seed: bool = True) -> str:
        """
        你原本會回 JSON list，這裡沿用你的擷取邏輯：
        先取大回覆中的第一個 '[' 到最後一個 ']'。
        
        Args:
            classification_prompt: 分类提示
            temperature: 可选的温度参数，默认使用 self.temp_cls
            use_seed: 是否使用固定 seed。False 时增加随机性（用于 multi-beam）
        """
        if temperature is None:
            temperature = self.temp_cls
            
        domain = getattr(config, "DOMAIN", "food").lower()
        if domain == "movie":
            system = """You are a helpful assistant that analyzes movie viewing sessions. Only output JSON. 
        IMPORTANT: Output must start with '[' and end with ']'. No other characters before or after."""
        else:
            system = """You are a helpful assistant that analyzes food e-commerce shopping sessions. Only output JSON. 
        IMPORTANT: Output must start with '[' and end with ']'. No other characters before or after."""
        user = classification_prompt.strip()

        # Multi-beam 模式下不使用固定 seed，增加随机性
        seed_param = "default" if use_seed else None
        content = self._chat(system, user, temperature=temperature, seed=seed_param)

        # 保留你原始的 JSON 擷取容錯
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            return content[start_idx:end_idx]
        else:
            print(f"Could not find valid JSON in response: {content}")
            return "Error: No valid JSON found"

    # -------------------------------
    # 評估指標（原樣保留）
    # -------------------------------
    def Recall_at_k(y_true, y_pred, k, agg="sum"):
        batch_size = y_pred.shape[0]
        topk_idxes = np.argpartition(-y_pred, k, axis=1)[:, :k]
        y_pred_bin = np.zeros_like(y_pred, dtype=bool)
        y_pred_bin[np.arange(batch_size)[:, None], topk_idxes] = True
        y_true_bin = (y_true > 0)
        hits = np.sum(np.logical_and(y_true_bin, y_pred_bin), axis=-1).astype(np.float32)
        recalls = hits/np.minimum(k, np.sum(y_true_bin, axis=1))
        if agg == "sum":
            recall = np.sum(recalls)
        elif agg == "mean":
            recall = np.mean(recalls)
        else:
            raise NotImplementedError(f"aggregation method {agg} not defined!")
        return recall

    def NDCG_at_k(y_true, y_pred, k, agg="sum"):
        batch_size = y_pred.shape[0]
        topk_idxes_unsort = np.argpartition(-y_pred, k, axis=1)[:, :k]
        topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
        topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
        topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
        y_true_topk = y_true[np.arange(batch_size)[:, None], topk_idxes]
        y_true_bin = (y_true > 0).astype(np.float32)
        weights = 1./np.log2(np.arange(2, k + 2))
        DCG = np.sum(y_true_topk*weights, axis=-1)
        normalizer = np.array([np.sum(weights[:int(n)]) for n in np.minimum(k, np.sum(y_true_bin, axis=-1))])
        if agg == "sum":
            NDCG = np.sum(DCG/normalizer)
        elif agg == "mean":
            NDCG = np.mean(DCG/normalizer)
        return NDCG
