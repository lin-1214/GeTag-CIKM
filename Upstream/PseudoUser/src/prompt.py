from config import Config
config = Config()

class Prompt:
    cluster_prompt = None
    classification_prompt = None
    augmentation_prompt = None

    def generate_cluster_prompt(self, t_clusters = Config.T_CLUSTERS, body = None, meta = None, previous_clusters = None, previous_tags = None, previous_high_tags = None, previous_low_tags = None):
        
        # This function is now responsible for separating the core instruction
        # from the output format, to prevent the format from being augmented.
        
        domain = getattr(config, "DOMAIN", "food").lower()

        output_format = f"""
        3. Output Requirements
        - Output the category labels as follows:
            {{
            "Categories": {{
                "Category 1": "<label_name>", "Category 2": "<label_name>", ...
            }},
            }}
        - Do not include any other text (no reasoning or explanation).
        """

        if meta is None:
            if domain == "movie":
                if config.T_VARIANCE == 0:
                    introduction = f"Below are user movie rating sessions. Divide them into exactly {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze each session's viewing and rating behavior.
                    - Based on genre preferences and rating patterns, group the sessions into exactly {t_clusters} persona categories and assign a label to each category.

                    2. Input Format
                    - You will be given multiple sessions, and each session begins with the word "Session".
                    - Each subsequent line represents an event that belongs to that session, containing the timestamp, rating score (1-5), and movie title.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in English.
                    - Use persona-style, behavior-centric labels that describe movie preferences, viewing habits, or rating tendencies.
                    - Prefer concise 2–4 word names (e.g., "Action Movie Enthusiast", "Classic Film Lover", "Family Entertainment Seeker", "Sci-Fi Fanatic", "Documentary Viewer").
                    - Avoid specific movie titles, actors, directors, or franchise names in the labels.
                    - Keep labels distinct and non-overlapping.
                    """
                else:
                    introduction = f"Below are user movie rating sessions. Divide them into approximately {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze each session's viewing and rating behavior.
                    - Based on genre preferences and rating patterns, group the sessions into {t_clusters - config.T_VARIANCE} to {t_clusters + config.T_VARIANCE} persona categories and assign a label to each category.

                    2. Input Format
                    - You will be given multiple sessions, and each session begins with the word "Session".
                    - Each subsequent line represents an event that belongs to that session, containing the timestamp, rating score (1-5), and movie title.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in English.
                    - Use persona-style, behavior-centric labels that describe movie preferences, viewing habits, or rating tendencies.
                    - Prefer concise 2–4 word names (e.g., "Action Movie Enthusiast", "Classic Film Lover", "Family Entertainment Seeker", "Sci-Fi Fanatic", "Documentary Viewer").
                    - Avoid specific movie titles, actors, directors, or franchise names in the labels.
                    - Keep labels distinct and non-overlapping.
                    """
            elif domain == "amazon":
                if config.T_VARIANCE == 0:
                    introduction = f"Below are Amazon product browsing sessions. Divide them into exactly {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze the product viewing patterns in each session.
                    - Based on product categories, browsing behavior, and shopping intent, group the sessions into exactly {t_clusters} user persona categories and assign a label to each category.

                    2. Input Format
                    - You will be given multiple sessions, and each session begins with the word "Session".
                    - Each subsequent line contains "- viewed '<product_title>'" representing a product the user browsed.
                    - Products are listed in chronological order of viewing.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in English.
                    - Use persona-style, behavior-centric labels that describe shopping interests, product preferences, or buyer types.
                    - Prefer concise 2–4 word names (e.g., "Gaming Enthusiast", "Home Improvement Shopper", "Tech Early Adopter", "Fitness Equipment Buyer", "Book Collector").
                    - Avoid specific product names, brands, or generic category names (e.g., avoid "Electronics Buyer", prefer "Smart Home Enthusiast").
                    - Keep labels distinct and non-overlapping.
                    """
                else:
                    introduction = f"Below are Amazon product browsing sessions. Divide them into approximately {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze the product viewing patterns in each session.
                    - Based on product categories, browsing behavior, and shopping intent, group the sessions into {t_clusters - config.T_VARIANCE} to {t_clusters + config.T_VARIANCE} user persona categories and assign a label to each category.

                    2. Input Format
                    - You will be given multiple sessions, and each session begins with the word "Session".
                    - Each subsequent line contains "- viewed '<product_title>'" representing a product the user browsed.
                    - Products are listed in chronological order of viewing.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in English.
                    - Use persona-style, behavior-centric labels that describe shopping interests, product preferences, or buyer types.
                    - Prefer concise 2–4 word names (e.g., "Gaming Enthusiast", "Home Improvement Shopper", "Tech Early Adopter", "Fitness Equipment Buyer", "Book Collector").
                    - Avoid specific product names, brands, or generic category names (e.g., avoid "Electronics Buyer", prefer "Smart Home Enthusiast").
                    - Keep labels distinct and non-overlapping.
                    """
            elif domain == "yelp":
                if config.T_VARIANCE == 0:
                    introduction = f"Below are Yelp restaurant browsing sessions. Divide them into exactly {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze the restaurant visiting patterns in each session.
                    - Based on cuisine types, dining preferences, and user behavior, group the sessions into exactly {t_clusters} user persona categories and assign a label to each category.

                    2. Input Format
                    - You will be given multiple sessions, and each session begins with the word "Session".
                    - Each subsequent line contains "- visited '<restaurant_name>'" representing a restaurant the user checked.
                    - Restaurants are listed in chronological order of visiting.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in English.
                    - Use persona-style, behavior-centric labels that describe dining preferences, cuisine interests, or diner types.
                    - Prefer concise 2–4 word names (e.g., "Fine Dining Enthusiast", "Fast Food Lover", "Ethnic Cuisine Explorer", "Health Conscious Diner", "Budget Eater", "Foodie Adventurer").
                    - Avoid specific restaurant names, chains, or generic category names (e.g., avoid "Restaurant Goer", prefer "Italian Food Lover").
                    - Keep labels distinct and non-overlapping.
                    """
                else:
                    introduction = f"Below are Yelp restaurant browsing sessions. Divide them into approximately {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze the restaurant visiting patterns in each session.
                    - Based on cuisine types, dining preferences, and user behavior, group the sessions into {t_clusters - config.T_VARIANCE} to {t_clusters + config.T_VARIANCE} user persona categories and assign a label to each category.

                    2. Input Format
                    - You will be given multiple sessions, and each session begins with the word "Session".
                    - Each subsequent line contains "- visited '<restaurant_name>'" representing a restaurant the user checked.
                    - Restaurants are listed in chronological order of visiting.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in English.
                    - Use persona-style, behavior-centric labels that describe dining preferences, cuisine interests, or diner types.
                    - Prefer concise 2–4 word names (e.g., "Fine Dining Enthusiast", "Fast Food Lover", "Ethnic Cuisine Explorer", "Health Conscious Diner", "Budget Eater", "Foodie Adventurer").
                    - Avoid specific restaurant names, chains, or generic category names (e.g., avoid "Restaurant Goer", prefer "Italian Food Lover").
                    - Keep labels distinct and non-overlapping.
                    """
            else:  # food domain (default)
                if config.T_VARIANCE == 0:
                    introduction = f"Below are user e-commerce sessions. Divide them into exactly {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze the interactions in each session.
                    - Based on this, put the sessions into exactly {t_clusters} categories and determine the label of each category.

                    2. Input Format
                    - You will be given multiple sessions, where each session starts with "Session" followed by a list of product names.
                    - Each product name represents an item the user interacted with during that shopping session.
                    - Products are listed in the order they were browsed or purchased.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in Chinese (請以中文書寫類別名稱)。
                    - Use persona-style, behavior-centric labels (user types), not product or cuisine names.
                    - Prefer concise 2–4 word names that reflect shopping behavior or intent.
                    - Avoid specific item/brand/department/cuisine names (e.g., "Dairy", "Italian Pasta").
                    - Keep labels distinct and non-overlapping.
                    - Good examples (中文): 「健身愛好者」、「精打細算的家庭主婦」、「宵夜族」、「品牌忠誠者」、「健康意識高」、「大量囤貨者」。

                    """
                else:
                    introduction = f"Below are user e-commerce sessions. Divide them into approximately {t_clusters} categories\n"
                    core_instruction = f"""
                    Instructions:

                    1. Task
                    - Analyze the interactions in each session.
                    - Based on this, put the sessions into {t_clusters - config.T_VARIANCE} to {t_clusters + config.T_VARIANCE} categories and determine the label of each category.

                    2. Input Format
                    - You will be given multiple sessions, where each session starts with "Session" followed by a list of product names.
                    - Each product name represents an item the user interacted with during that shopping session.
                    - Products are listed in the order they were browsed or purchased.

                    3. Labeling Guidelines (CRITICAL)
                    - All category names MUST be written in Chinese (請以中文書寫類別名稱)。
                    - Use persona-style, behavior-centric labels (user types), not product or cuisine names.
                    - Prefer concise 2–4 word names that reflect shopping behavior or intent.
                    - Avoid specific item/brand/department/cuisine names (e.g., "Dairy", "Italian Pasta").
                    - Keep labels distinct and non-overlapping.
                    - Good examples (中文): 「健身愛好者」、「精打細算的家庭主婦」、「宵夜族」、「品牌忠誠者」、「健康意識高」、「大量囤貨者」。
                    """
            
            meta = f"{introduction}\n\n{core_instruction}"
        # Build references block separately to avoid mutating `meta` (which is returned for augmentation)
        references_block = ""
        # Prefer explicit high/low tags if provided; otherwise fall back to previous_tags
        sec_idx = 4
        if previous_high_tags is not None:
            try:
                from json import dumps as _jd
                high_str = previous_high_tags if isinstance(previous_high_tags, str) else _jd(previous_high_tags, ensure_ascii=False)
            except Exception:
                high_str = str(previous_high_tags)
            references_block += f"""
            {sec_idx}. Reference High-Frequency Tags (from previous iteration)
            - Prioritize these tags; prefer keeping or adapting them unless the session evidence strongly contradicts.
            {high_str}
            """
            sec_idx += 1
        if previous_low_tags is not None and previous_low_tags != []:
            try:
                from json import dumps as _jd
                low_str = previous_low_tags if isinstance(previous_low_tags, str) else _jd(previous_low_tags, ensure_ascii=False)
            except Exception:
                low_str = str(previous_low_tags)
            references_block += f"""
            {sec_idx}. Low-Frequency Tags to Avoid
            - Do NOT reuse any of these tags. They performed poorly in the previous round; replace them with better alternatives even if they appear in the sessions.
            {low_str}
            """
            sec_idx += 1
        elif previous_tags:
            # Backward compatible single list
            try:
                from json import dumps as _jd
                tags_str = previous_tags if isinstance(previous_tags, str) else _jd(previous_tags, ensure_ascii=False)
            except Exception:
                tags_str = str(previous_tags)
            references_block += f"""
            {sec_idx}. Reference Tags (from previous iteration)
            - Use these tags as inspiration for category naming.
            {tags_str}
            """
        # The final prompt is constructed by combining the (potentially augmented) meta
        # with the static output format and the dynamic body.
        final_prompt_meta = meta + references_block + output_format
        self.cluster_prompt = f"{final_prompt_meta}\n\n{body}"
        
        # We return the original (unaugmented by refs) `meta` for augmentation, and the full prompt for execution.
        return self.cluster_prompt, meta
    
    def generate_augmentation_prompt(self, body = None):
        self.augmentation_prompt = f"""
        1. Task
        - Given a prompt instruction, generate an improved variation of it by applying effective prompting techniques.
        - Keep the main ideas and objectives intact, but enhance the prompt using various prompting skills to help the LLM complete the task more effectively.
        - You may apply techniques such as:
            * Chain-of-thought reasoning (asking the model to think step-by-step)
            * Few-shot examples (adding relevant examples)
            * Role-playing (assigning a specific expert role)
            * Constraint specification (being more explicit about requirements)
            * Output structuring (clarifying the expected format)
            * Decomposition (breaking complex tasks into sub-tasks)
            * Contextual grounding (providing relevant background information)
        - Feel free to rephrase, reorganize, prune, or elaborate on the details to make the instruction clearer and more effective.

        2. Input
        {body}

        3. Output Requirements
        - Do not include any other text (no reasoning, explanation, or even wrapper text like "Here is a rewritten version…"), only output the improved prompt instruction itself.
        """

        return self.augmentation_prompt
        
        
    
    def generate_classification_prompt(self, k_aug = Config.K_AUG, batch_size = Config.BATCH_SIZE, categories = None, body = None):

        domain = getattr(config, "DOMAIN", "food").lower()

        if config.MULTI_LABEL:
            if domain == "movie":
                output_format = f"""
                    3. Output Requirements
                    - Output must be a JSON array of length {batch_size}.
                    - Each element must contain the key "predicted_classes" with a JSON array of distinct category names.
                    - Choose between {config.MIN_LABELS_PER_SESSION} and {config.MAX_LABELS_PER_SESSION} labels per session.
                    - Only use category names from the provided list, exactly as given.
                    - All category names MUST be written in English.
                    - Do not include any other text (no reasoning or explanation).
                    """
                per_session_instructions = "for each session, assign one or more persona labels from the provided list that best describe the viewer's movie preferences (multi-label)."
                output_example = """
                    [
                    {"predicted_classes": ["Action Movie Enthusiast", "Sci-Fi Fanatic"]},
                    {"predicted_classes": ["Classic Film Lover"]}
                    ]
                    """
            elif domain == "amazon":
                output_format = f"""
                    3. Output Requirements
                    - Output must be a JSON array of length {batch_size}.
                    - Each element must contain the key "predicted_classes" with a JSON array of distinct category names.
                    - Choose between {config.MIN_LABELS_PER_SESSION} and {config.MAX_LABELS_PER_SESSION} labels per session.
                    - Only use category names from the provided list, exactly as given.
                    - All category names MUST be written in English.
                    - Do not include any other text (no reasoning or explanation).
                    """
                per_session_instructions = "for each session, assign one or more persona labels from the provided list that best describe the shopper's interests (multi-label)."
                output_example = """
                    [
                    {"predicted_classes": ["Gaming Enthusiast", "Tech Early Adopter"]},
                    {"predicted_classes": ["Home Improvement Shopper"]}
                    ]
                    """
            elif domain == "yelp":
                output_format = f"""
                    3. Output Requirements
                    - Output must be a JSON array of length {batch_size}.
                    - Each element must contain the key "predicted_classes" with a JSON array of distinct category names.
                    - Choose between {config.MIN_LABELS_PER_SESSION} and {config.MAX_LABELS_PER_SESSION} labels per session.
                    - Only use category names from the provided list, exactly as given.
                    - All category names MUST be written in English.
                    - Do not include any other text (no reasoning or explanation).
                    """
                per_session_instructions = "for each session, assign one or more persona labels from the provided list that best describe the diner's preferences (multi-label)."
                output_example = """
                    [
                    {"predicted_classes": ["Fine Dining Enthusiast", "Italian Food Lover"]},
                    {"predicted_classes": ["Fast Food Lover"]}
                    ]
                    """
            else:  # food domain
                output_format = f"""
                    3. Output Requirements
                    - Output must be a JSON array of length {batch_size}.
                    - Each element must be an object with key "predicted_classes" whose value is a JSON array of distinct category names.
                    - Choose between {config.MIN_LABELS_PER_SESSION} and {config.MAX_LABELS_PER_SESSION} labels per session.
                    - Only use category names from the provided list, exactly as given.
                    - All category names MUST be written in Chinese (請以中文書寫類別名稱)。
                    - Do not include any other text (no reasoning or explanation).
                    """
                per_session_instructions = "for each session, classify it into one or more categories from the given category names (multi-label)."
                output_example = """
                    [
                    {"predicted_classes": ["健身愛好者", "精打細算的家庭主婦"]},
                    {"predicted_classes": ["健康意識高"]}
                    ]
                    """
        else:  # single label
            if domain == "movie":
                output_format = f"""
                    3. Output Requirements
                    - Output must follow this JSON array format:
                    [
                    {{"predicted_class": "Category name"}},
                    {{"predicted_class": "Category name"}},
                    ...
                    {{"predicted_class": "Category name"}}
                    ]
                    - Do not include any other text (no reasoning or explanation).
                    - The number of outputs must match {batch_size}.
                    - Each output must use exactly one of the provided category names (case-sensitive).
                    - All category names MUST be written in English.
                    """
                per_session_instructions = "for each session, assign exactly one persona label from the provided list."
                output_example = ""
            elif domain == "amazon":
                output_format = f"""
                    3. Output Requirements
                    - Output must follow this JSON array format:
                    [
                    {{"predicted_class": "Category name"}},
                    {{"predicted_class": "Category name"}},
                    ...
                    {{"predicted_class": "Category name"}}
                    ]
                    - Do not include any other text (no reasoning or explanation).
                    - The number of outputs must match {batch_size}.
                    - Each output must use exactly one of the provided category names (case-sensitive).
                    - All category names MUST be written in English.
                    """
                per_session_instructions = "for each session, assign exactly one persona label from the provided list."
                output_example = ""
            elif domain == "yelp":
                output_format = f"""
                    3. Output Requirements
                    - Output must follow this JSON array format:
                    [
                    {{"predicted_class": "Category name"}},
                    {{"predicted_class": "Category name"}},
                    ...
                    {{"predicted_class": "Category name"}}
                    ]
                    - Do not include any other text (no reasoning or explanation).
                    - The number of outputs must match {batch_size}.
                    - Each output must use exactly one of the provided category names (case-sensitive).
                    - All category names MUST be written in English.
                    """
                per_session_instructions = "for each session, assign exactly one persona label from the provided list."
                output_example = ""
            else:  # food domain
                output_format = f"""
                    3. Output Requirements
                    - Output must be in the following format:
                    [
                    {{"predicted_class": "Category name"}},
                    {{"predicted_class": "Category name"}},
                    ...
                    {{"predicted_class": "Category name"}}
                    ]
                    - Do not include any other text (no reasoning or explanation).
                    - The number of outputs must match {batch_size}.
                    - Each output must use exactly one of the given category names. Be sure to use them exactly as provided and without any prefix.
                    - All category names MUST be written in Chinese (請以中文書寫類別名稱)。
                    - Before submitting the output, check carefully that the format and given category names are correct.
                    """
                per_session_instructions = "for each session, classify it into exactly one of the categories from the given category names."
                output_example = ""

        if domain == "movie":
            self.classification_prompt = f"""
                Instructions:

                1. Task
                - First, carefully review these category names that you must use exactly as provided:
                {categories}.
                - Second, {per_session_instructions}
                - To determine the correct category, analyze each session based on:
                    - Genres and themes reflected in the rated movies
                    - Rating scores (e.g., high vs. low ratings)
                    - Variety or consistency in viewing choices
                    - Frequency and timing of ratings
                    - Any other notable viewing behaviors or patterns

                2. Input Format
                - You will be given multiple sessions in the following structure:
                    ### SESSION <user_id>:
                    - <hh:mm:ss>: rated <score>/5 '<movie_title>'
                    - <hh:mm:ss>: rated <score>/5 '<movie_title>'
                    ...
                - <user_id> = anonymized identifier for the viewer.
                - <hh:mm:ss> = timestamp of the rating event.
                - <score>/5 = rating value (1–5, with 5 being the highest).
                - <movie_title> = title of the movie.

                3. Labeling Guidelines (CRITICAL)
                - Assign persona-style, behavior-centric labels that describe viewing or rating preferences.
                - Keep names concise (2–4 words) and ensure they are semantically distinct.
                - All category names MUST be written in English.
                - Do not use specific movie titles, actors, directors, or franchise names in the labels.

                {output_format}
                - Example output:
                {output_example}

                4. Sessions
                - Here are the sessions to classify:
                {body}
                """
        elif domain == "amazon":
            self.classification_prompt = f"""
                Instructions:

                1. Task
                - First, carefully review these category names that you must use exactly as provided:
                {categories}.
                - Second, {per_session_instructions}
                - To determine the correct category, analyze each session based on:
                    - Product categories and types being browsed
                    - Shopping intent (e.g., specific purchase vs. browsing)
                    - Consistency or variety in product choices
                    - Price ranges and product quality levels
                    - Any notable patterns in viewing behavior

                2. Input Format
                - You will be given multiple sessions in the following structure:
                    ### SESSION <session_id>:
                    - viewed '<product_title>'
                    - viewed '<product_title>'
                    ...
                - <session_id> = anonymized identifier for the shopping session.
                - <product_title> = title/name of the product viewed.
                - Products are listed in chronological order of viewing.

                3. Labeling Guidelines (CRITICAL)
                - Assign persona-style, behavior-centric labels that describe shopping interests and buyer types.
                - Keep names concise (2–4 words) and ensure they are semantically distinct.
                - All category names MUST be written in English.
                - Do not use specific product names or brands in the labels.

                {output_format}
                - Example output:
                {output_example}

                4. Sessions
                - Here are the sessions to classify:
                {body}
                """
        elif domain == "yelp":
            self.classification_prompt = f"""
                Instructions:

                1. Task
                - First, carefully review these category names that you must use exactly as provided:
                {categories}.
                - Second, {per_session_instructions}
                - To determine the correct category, analyze each session based on:
                    - Cuisine types and restaurant categories being browsed
                    - Dining preferences (e.g., fine dining vs. casual)
                    - Consistency or variety in restaurant choices
                    - Price ranges and quality levels
                    - Any notable patterns in visiting behavior

                2. Input Format
                - You will be given multiple sessions in the following structure:
                    ### SESSION <session_id>:
                    - visited '<restaurant_name>'
                    - visited '<restaurant_name>'
                    ...
                - <session_id> = anonymized identifier for the browsing session.
                - <restaurant_name> = name of the restaurant visited.
                - Restaurants are listed in chronological order of visiting.

                3. Labeling Guidelines (CRITICAL)
                - Assign persona-style, behavior-centric labels that describe dining preferences and diner types.
                - Keep names concise (2–4 words) and ensure they are semantically distinct.
                - All category names MUST be written in English.
                - Do not use specific restaurant names or chains in the labels.

                {output_format}
                - Example output:
                {output_example}

                4. Sessions
                - Here are the sessions to classify:
                {body}
                """
        else:  # food domain
            self.classification_prompt = f"""
                Instructions:

                1. Task
                - First, carefully review these category names that you must use, use the exact names not the numbers or prefixes like "Category 1":
                {categories}.
                - Second, {per_session_instructions}
                - To determine the correct category, analyze the session based on:
                    - Overall shopping behavior (e.g., focused vs. exploratory browsing)
                    - Types of products the user interacted with
                    - Variety and diversity of products in the session
                    - Product categories and themes present
                    - Any notable patterns or insights in product selection

                2. Input Format
                - You will be given multiple sessions in the following structure:
                    ### SESSION <session_id>:
                    <product_name_1>, <product_name_2>, <product_name_3>, ...
                - <session_id> = unique identifier for the shopping session.
                - Each product name represents an item the user interacted with during the session.
                - Products are listed in chronological order (the order they were browsed or purchased).

                3. Labeling Guidelines (CRITICAL)
                - Assign persona-style, behavior-centric labels; avoid product/cuisine/brand/department labels.
                - Keep names concise (2–4 words) and semantically distinct.
                - All category names MUST be written in Chinese (請以中文書寫類別名稱)。
                - Choose labels that best describe the user's shopping behavior or intent.

                {output_format}
                - Example output:
                {output_example}

                4. Sessions
                - Here are the sessions to classify:
                {body}
                """

        self.classification_prompt_shortened = self.classification_prompt

        return self.classification_prompt

