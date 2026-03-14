# Upstream: Session Labeling for GeTag Pipeline

> For detailed usage, please refer to [`PseudoUser/README.md`](./PseudoUser/README.md).

## Overview

This module is the **upstream** component of the GeTag pipeline. Its sole job is to run an LLM-based session labeling system that classifies user sessions into pseudo-user persona labels. The output feeds directly into the GeTag downstream recommendation system.

**In practice, you only need to run one script:**

```bash
cd PseudoUser/src
bash run_all_label_phase.sh
```

This script generates classified session CSV files for all dataset x tag combinations (food, games/amazon, yelp x native/base tags) and copies them into the GeTag classified data directory.

---

## Role in the Pipeline

```
Upstream (this module)
  raw session data + item tags
       |   run_all_label_phase.sh
       v
  label_phase.py  (LLM: Qwen3-4B via vLLM)
       |
       v
  classified_data_{domain}_{tag}_0.csv
       |   copied to GeTag
       v
Downstream (GeTag)
  data/classified/{dataset}_{tag}.csv
```

---

## Background (Original PULLRS System)

This module was originally built as **PULLRS** — a full pseudo-user recommendation system combining prompt-based tagging, UCB reinforcement learning for prompt selection, and masked next-item prediction. That full system has since been simplified: only the **label (classification) phase** is used in the current GeTag pipeline.

Historical experimental scripts (retrieval, fine-tuning, multi-beam, etc.) remain in the repo but are not part of the active workflow.
