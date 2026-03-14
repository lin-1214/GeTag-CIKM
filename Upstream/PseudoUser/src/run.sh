#!/bin/bash

# Configuration
NUM_ITERATIONS=20  # Number of times to run the full cycle
LABEL_GPU=0       # GPU for label phase
# Read TOP_K_PROMPTS from config.py to ensure consistency
# Use grep to extract the value without importing (avoids dependency issues in shell)
TOP_K_PROMPTS=$(grep -oP 'TOP_K_PROMPTS\s*=\s*\K\d+' config.py | head -1)
EVAL_SOURCE=user  # Selection source: user | item | auto

echo "Configuration loaded:"
echo "  NUM_ITERATIONS=$NUM_ITERATIONS"
echo "  TOP_K_PROMPTS=$TOP_K_PROMPTS (from config.py)"
echo "  EVAL_SOURCE=$EVAL_SOURCE"
echo ""

# --- Preparing for new run ---
echo "Cleaning up all artifacts from previous runs..."

# 1. Remove archive and checkpoint directories
for dir in ../label_data/archive ../cllm_data/archive json/archive checkpoints; do
    if [ -d "$dir" ]; then
        echo "Removing $dir"
        rm -rf "$dir"
    fi
done

# 2. Remove classified data from previous runs
rm -f ../label_data/classified_data_*.csv
rm -rf ../cllm_data/user_session_data_*

# 3. Remove json working files
echo "Cleaning json/ directory..."
rm -f json/best_prompts.json
rm -f json/best_prompts_test.json
rm -f json/iteration_meta.json
rm -f json/iteration_metrics.json
rm -f json/iteration_prompt_scores.csv
rm -f json/iteration_best_scores.png
rm -f json/iteration_prompts.json
rm -f json/selected_prompt_history.json  # Clear history at start of new run

# 4. Remove results and tags from previous runs
echo "Cleaning results and tags..."
rm -f ../results/retrieval_v2/summary_*.json
rm -f ../results/retrieval_v2/retrieval_results_*.csv
rm -f ../json/tags/getags_zscore_*.json
rm -f ../json/item_group_mapping.json
rm -f ../json/group_tag_frequency.json

# 5. Ensure the base directories exist for the new run
mkdir -p json
mkdir -p ../results/retrieval_v2
mkdir -p ../json/tags


for ((i=1; i<=$NUM_ITERATIONS; i++)); do
    echo "Starting iteration $i of $NUM_ITERATIONS"

    # Clean up checkpoints for variating user design
    if [ -d "checkpoints" ]; then
        echo "Removing existing checkpoints directory"
        rm -rf checkpoints
    fi
    
    # Run label phase
    # CUDA_VISIBLE_DEVICES=$LABEL_GPU python3 label_phase.py
    python3 label_phase.py

    # Clean previous iteration's temporary files (tags and summaries for this iteration only)
    echo "Cleaning temporary files from previous attempts of iteration $i..."
    rm -f ../results/retrieval_v2/summary_getags_zscore_iter${i}_*.json
    rm -f ../json/tags/getags_zscore_iter${i}_*.json
    rm -f ../json/item_group_mapping.json ../json/group_tag_frequency.json

    # Evaluate each classified CSV greedily
    echo "Evaluating each classified_data_*.csv with evaluation_phase.py (no UCB, greedy)"
    shopt -s nullglob
    csv_files=(../label_data/classified_data_*.csv)
    if [ ${#csv_files[@]} -eq 0 ]; then
        echo "Error: No 'classified_data_*.csv' files found after label_phase.py on iteration $i. Aborting."
        exit 1
    fi

    # Optionally clean caches once per iteration for fresh results
    CLEAN_FLAG=--clean_cache
    
    # Run evaluation for each CSV and collect summary scores
    summaries=()
    scores=()
    indices=()

    for csv_path in "${csv_files[@]}"; do
        filename=$(basename "$csv_path")
        # Extract index from classified_data_<idx>.csv
        if [[ $filename =~ classified_data_([0-9]+)\.csv ]]; then
            idx=${BASH_REMATCH[1]}
        else
            echo "Warning: Could not parse index from $filename, skipping."
            continue
        fi
        tag_name="getags_zscore_iter${i}_${idx}"
        echo "Evaluating $filename with tag_name=$tag_name (eval_source=$EVAL_SOURCE)"
        # Clean cache only once for the first file
        if [ ${#summaries[@]} -eq 0 ]; then
            python3 evaluation_phase.py --classified_csv "$csv_path" --tag_name "$tag_name" --eval_source "$EVAL_SOURCE" $CLEAN_FLAG
        else
            python3 evaluation_phase.py --classified_csv "$csv_path" --tag_name "$tag_name" --eval_source "$EVAL_SOURCE"
        fi
        summary_path="../results/retrieval_v2/summary_${tag_name}.json"
        if [ -f "$summary_path" ]; then
            summaries+=("$summary_path")
            indices+=("$idx")
            # Extract and log scores: user/item best to CSV; print best for sorting
            score=$(python3 - "$summary_path" "$i" "$idx" "$tag_name" <<'PY'
import json,sys,os,csv
p,iteration,pidx,tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(p,'r',encoding='utf-8') as f:
    d=json.load(f)
user = (d.get('results',{}) or {}).get('userbased',{})
item = (d.get('results',{}) or {}).get('itembased',{})
user_score = user.get('ndcg@10/100/test', 0.0)
item_score = item.get('ndcg@10/100/test', 0.0)
best = d.get('best',{})
best_score = best.get('score',0.0)
best_source = best.get('source', '')
path='json/iteration_prompt_scores.csv'
exists = os.path.exists(path)
with open(path,'a',newline='',encoding='utf-8') as f2:
    w=csv.writer(f2)
    if not exists:
        w.writerow(['iteration','prompt_index','tag_name','user_ndcg10_100_test','item_ndcg10_100_test','best_source','best_score'])
    w.writerow([iteration,pidx,tag,user_score,item_score,best_source,best_score])
print(best_score)
PY
)
            scores+=("$score")
            echo "  -> Best Score: $score"
        else
            echo "Warning: summary not found: $summary_path"
        fi
    done

    shopt -u nullglob

    # Select top-K indices greedily by score
    if [ ${#scores[@]} -eq 0 ]; then
        echo "Error: No scores collected in iteration $i. Aborting."
        exit 1
    fi

    # Build a sortable list "score idx"
    tmp_list=()
    for j in "${!scores[@]}"; do
        tmp_list+=("${scores[$j]} ${indices[$j]}")
    done
    # Sort descending by score and take top K
    # Fixed: Use 'sort -n -r' instead of 'sort -r -k1,1g' for correct numeric descending sort
    IFS=$'\n' sorted=( $(printf '%s\n' "${tmp_list[@]}" | sort -n -r) )
    top_indices=()
    count=0
    for entry in "${sorted[@]}"; do
        idx=$(echo "$entry" | awk '{print $2}')
        top_indices+=("$idx")
        count=$((count+1))
        if [ $count -ge $TOP_K_PROMPTS ]; then
            break
        fi
    done

    echo "Selected Top-$TOP_K_PROMPTS prompt indices: ${top_indices[*]}"

    # Map indices back to metas/clusters using iteration_meta.json
    if [ ! -f "json/iteration_meta.json" ]; then
        echo "Error: json/iteration_meta.json not found. Aborting."
        exit 1
    fi

    # Pass indices as args to Python (avoid bad substitution)
    # First arg is iteration number, rest are indices
python3 - $i ${top_indices[@]} <<'PY'
import json,sys,os,csv,ast,hashlib
from collections import Counter
from datetime import datetime

def parse_labels(v):
    # Robust parser for multi/single label in 'Class' column
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, (int,float)):
        return [str(v)]
    s = str(v).strip()
    if not s:
        return []
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if isinstance(x, str) and str(x).strip()]
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
    # Fallback: split by common separators
    for sep in ['|', ',', ';']:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]

with open('json/iteration_meta.json','r',encoding='utf-8') as f:
    data=json.load(f)
# First arg is iteration number
i=int(sys.argv[1])
# Rest are prompt indices
indices=[int(s) for s in sys.argv[2:]]
metas=[]
clusters=[]
tags_list=[]
high_tags_list=[]
low_tags_list=[]
source_steps=[]
source_is_original=[]
source_parent_meta=[]
source_tag_names=[]
source_meta_digest=[]
source_cluster_digest=[]
selected_entries=[]

for s in indices:
    if 0 <= s < len(data):
        entry=data[s]
        meta_val=entry.get('meta')
        cluster_val=entry.get('clusters')
        entry_step=entry.get('step', s)
        is_original=bool(entry.get('is_original', False))
        parent_meta=entry.get('parent_meta')
        
        # Build lineage by tracing back through previous best_prompts.json
        lineage_chain=[]
        if parent_meta:
            lineage_chain.append(parent_meta)
            # Try to trace further back through previous iterations
            try:
                prev_best_path='json/best_prompts.json'
                if os.path.exists(prev_best_path):
                    with open(prev_best_path,'r',encoding='utf-8') as bf:
                        prev_best=json.load(bf)
                        prev_metas=prev_best.get('metas',[])
                        prev_parent_lineage=prev_best.get('source_parent_lineage',[])
                        # Fallback to old field name
                        if not prev_parent_lineage:
                            prev_parent_lineage=prev_best.get('source_parent_meta',[])
                        # Find if parent_meta matches any previous meta
                        for idx,pm in enumerate(prev_metas):
                            if pm==parent_meta and idx<len(prev_parent_lineage):
                                # Add the previous lineage chain
                                prev_chain=prev_parent_lineage[idx]
                                if isinstance(prev_chain,list):
                                    lineage_chain.extend(prev_chain)
                                elif prev_chain:  # String (old format)
                                    lineage_chain.append(prev_chain)
                                break
            except Exception as e:
                pass  # If can't trace back, just keep what we have

        metas.append(meta_val)
        clusters.append(cluster_val)
        source_steps.append(entry_step)
        source_is_original.append(is_original)
        # Store the lineage chain (always a list, may be empty)
        source_parent_meta.append(lineage_chain)
        
        # NEW APPROACH: Read strong_tags and sparse_tags from gen_getag.py output
        tag_name=f"getags_zscore_iter{i}_{s}"
        strong_tags_path=f"../json/tags/{tag_name}_strong_tags.json"
        sparse_tags_path=f"../json/tags/{tag_name}_sparse_tags.json"
        
        high_tags=[]
        low_tags=[]
        all_tags=[]
        source_tag_names.append(tag_name)
        
        # Try to read strong tags (high-performance tags)
        if os.path.exists(strong_tags_path):
            try:
                with open(strong_tags_path,'r',encoding='utf-8') as f:
                    strong_data=json.load(f)
                    # Extract just the tag names from (tag, freq) tuples
                    high_tags=[tag for tag,freq in strong_data.get('tags',[])]
                    all_tags.extend(high_tags)
                    print(f"Loaded {len(high_tags)} strong tags from {strong_tags_path}")
            except Exception as e:
                print(f"Warning: Could not load strong tags: {e}")
        
        # Try to read sparse tags (low-performance tags to avoid)
        if os.path.exists(sparse_tags_path):
            try:
                with open(sparse_tags_path,'r',encoding='utf-8') as f:
                    sparse_data=json.load(f)
                    # Extract just the tag names from (tag, freq) tuples
                    low_tags=[tag for tag,freq in sparse_data.get('tags',[])]
                    print(f"Loaded {len(low_tags)} sparse tags from {sparse_tags_path}")
            except Exception as e:
                print(f"Warning: Could not load sparse tags: {e}")
        
        # FALLBACK: If strong/sparse tags don't exist, use old CSV-based approach
        if not high_tags and not low_tags:
            print(f"Fallback: Using CSV-based tag extraction for index {s}")
            csv_path=f"../label_data/classified_data_{s}.csv"
            tag_counts=Counter()
            if os.path.exists(csv_path):
                try:
                    with open(csv_path,'r',encoding='utf-8') as fcsv:
                        rdr=csv.DictReader(fcsv)
                        for row in rdr:
                            labs=parse_labels(row.get('Class'))
                            tag_counts.update(labs)
                except Exception:
                    pass
            # Sorted tags by frequency (desc)
            sorted_tags=[t for t,_ in tag_counts.most_common()]
            all_tags=sorted_tags
            # Split by ratio (default 0.5)
            try:
                ratio=float(os.getenv('HIGH_TAG_RATIO','0.5'))
            except Exception:
                ratio=0.5
            ratio=max(0.0,min(1.0,ratio))
            if sorted_tags:
                split_idx=max(1, int(len(sorted_tags)*ratio)) if ratio>0 else 0
                high_tags=sorted_tags[:split_idx]
                low_tags=sorted_tags[split_idx:]
        
        tags_list.append(all_tags if all_tags else high_tags+low_tags)
        high_tags_list.append(high_tags)
        low_tags_list.append(low_tags)

        meta_digest=None
        cluster_digest=None
        if isinstance(meta_val,str):
            meta_digest=hashlib.sha1(meta_val.encode('utf-8')).hexdigest()[:12]
        if isinstance(cluster_val,str):
            cluster_digest=hashlib.sha1(cluster_val.encode('utf-8')).hexdigest()[:12]
        source_meta_digest.append(meta_digest)
        source_cluster_digest.append(cluster_digest)
        selected_entries.append({
            'index': s,
            'step': entry_step,
            'is_original': is_original,
            'tag_name': tag_name,
            'parent_meta_digest': hashlib.sha1(parent_meta.encode('utf-8')).hexdigest()[:12] if isinstance(parent_meta, str) else None,
            'meta_digest': meta_digest,
            'clusters_digest': cluster_digest,
            'strong_tags_count': len(high_tags),
            'sparse_tags_count': len(low_tags)
        })

out={
    'metas':metas,
    'clusters':clusters,
    'tags':tags_list,
    'high_tags':high_tags_list,
    'low_tags':low_tags_list,
    'source_steps':source_steps,
    'source_is_original':source_is_original,
    'source_parent_meta':source_parent_meta,  # Now contains lineage chains (list of lists)
    'source_parent_lineage':source_parent_meta,  # Alias for clarity
    'source_tag_names':source_tag_names,
    'source_meta_digest':source_meta_digest,
    'source_cluster_digest':source_cluster_digest
}
with open('json/best_prompts.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print('Wrote json/best_prompts.json with', len(metas), 'prompts')
print(f"  High-frequency (strong) tags: {sum(len(h) for h in high_tags_list)} total")
print(f"  Low-frequency (sparse) tags: {sum(len(l) for l in low_tags_list)} total")

history_path='json/selected_prompt_history.json'
if selected_entries:
    record={
        'iteration': i,
        'timestamp': datetime.utcnow().isoformat()+'Z',
        'selected': selected_entries
    }
    # Append to history (accumulate across iterations within one run)
    # History is cleared at the start of each new run (see cleanup section)
    try:
        with open(history_path,'r',encoding='utf-8') as hf:
            history=json.load(hf)
            if not isinstance(history,list):
                history=[]
    except FileNotFoundError:
        history=[]
    except Exception as e:
        print(f"Warning: Failed to read {history_path}: {e}")
        history=[]
    history.append(record)
    try:
        with open(history_path,'w',encoding='utf-8') as hf:
            json.dump(history,hf,ensure_ascii=False,indent=2)
        print(f"Appended iteration {i} to {history_path} (total: {len(history)} iterations)")
    except Exception as e:
        print(f"Warning: Failed to update {history_path}: {e}")
else:
    print("Warning: No valid selections; history file not updated.")
PY

    echo "Archiving data for iteration $i..."
    # Create archive directories for this iteration
    ARCHIVE_LABEL_PATH="../label_data/archive/iter_$i"
    ARCHIVE_CLLM_PATH="../cllm_data/archive/iter_$i"
    ARCHIVE_JSON_PATH="json/archive/iter_$i"
    ARCHIVE_RESULTS_PATH="../results/retrieval_v2/archive/iter_$i"
    mkdir -p "$ARCHIVE_LABEL_PATH"
    mkdir -p "$ARCHIVE_CLLM_PATH"
    mkdir -p "$ARCHIVE_JSON_PATH"
    mkdir -p "$ARCHIVE_RESULTS_PATH"

    # Archive label CSVs
    shopt -s nullglob
    csv_files=(../label_data/classified_data_*.csv)
    if [ ${#csv_files[@]} -gt 0 ]; then
        cp "${csv_files[@]}" "$ARCHIVE_LABEL_PATH/"
    else
        echo "Error: No 'classified_data_*.csv' files found after label_phase.py on iteration $i. Aborting."
        exit 1
    fi
    shopt -u nullglob

    # Archive iteration-specific json files, ensuring they exist
    if [ -f "json/best_prompts.json" ]; then
        cp "json/best_prompts.json" "$ARCHIVE_JSON_PATH/best_prompts.json"
    else
        echo "Error: 'json/best_prompts.json' not found after evaluation on iteration $i. Aborting."
        exit 1
    fi
    if [ -f "json/iteration_meta.json" ]; then
        cp "json/iteration_meta.json" "$ARCHIVE_JSON_PATH/iteration_meta.json"
    else
        echo "Error: 'json/iteration_meta.json' not found after label_phase.py on iteration $i. Aborting."
        exit 1
    fi
    if [ -f "json/iteration_prompts.json" ]; then
        cp "json/iteration_prompts.json" "$ARCHIVE_JSON_PATH/iteration_prompts.json"
    fi

    # Archive retrieval results for this iteration
    shopt -s nullglob
    cp ../results/retrieval_v2/*_getags_zscore_iter${i}_*.csv "$ARCHIVE_RESULTS_PATH/" 2>/dev/null || true
    cp ../results/retrieval_v2/summary_getags_zscore_iter${i}_*.json "$ARCHIVE_RESULTS_PATH/" 2>/dev/null || true
    shopt -u nullglob

    # Archive strong/sparse tags files
    shopt -s nullglob
    cp ../json/tags/getags_zscore_iter${i}_*_strong_tags.json "$ARCHIVE_JSON_PATH/" 2>/dev/null || true
    cp ../json/tags/getags_zscore_iter${i}_*_sparse_tags.json "$ARCHIVE_JSON_PATH/" 2>/dev/null || true
    shopt -u nullglob

    echo "Completed iteration $i"
done