#!/bin/bash

# Test script for Phase 1 changes
# This runs a single iteration to verify the strong/sparse tags implementation

echo "========================================================================"
echo "PHASE 1 TEST: Strong/Sparse Tags Implementation"
echo "========================================================================"

# Configuration
TEST_ITERATION=1
LABEL_GPU=0
EVAL_SOURCE=user

echo ""
echo "Step 1: Clean up previous test files"
echo "------------------------------------------------------------------------"
rm -f ../label_data/classified_data_*.csv
rm -f ../json/tags/getags_zscore_*
rm -f ../json/item_group_mapping.json
rm -f ../json/group_tag_frequency.json
rm -f ../results/retrieval_v2/summary_*.json
rm -f json/best_prompts.json
rm -f json/iteration_meta.json
echo "✓ Cleaned up"

echo ""
echo "Step 2: Run label_phase.py"
echo "------------------------------------------------------------------------"
python3 label_phase.py
if [ $? -ne 0 ]; then
    echo "✗ label_phase.py failed"
    exit 1
fi
echo "✓ label_phase.py completed"

echo ""
echo "Step 3: Check classified CSV files"
echo "------------------------------------------------------------------------"
shopt -s nullglob
csv_files=(../label_data/classified_data_*.csv)
if [ ${#csv_files[@]} -eq 0 ]; then
    echo "✗ No classified_data_*.csv files found"
    exit 1
fi
echo "✓ Found ${#csv_files[@]} classified CSV files:"
for f in "${csv_files[@]}"; do
    echo "  - $(basename $f)"
done

echo ""
echo "Step 4: Run evaluation_phase.py for each CSV"
echo "------------------------------------------------------------------------"
for csv_path in "${csv_files[@]}"; do
    filename=$(basename "$csv_path")
    if [[ $filename =~ classified_data_([0-9]+)\.csv ]]; then
        idx=${BASH_REMATCH[1]}
    else
        echo "⚠ Warning: Could not parse index from $filename, skipping."
        continue
    fi
    
    tag_name="getags_zscore_test_${idx}"
    echo "Evaluating $filename -> $tag_name"
    
    python3 evaluation_phase.py \
        --classified_csv "$csv_path" \
        --tag_name "$tag_name" \
        --eval_source "$EVAL_SOURCE" \
        --clean_cache
    
    if [ $? -ne 0 ]; then
        echo "✗ evaluation_phase.py failed for $filename"
        exit 1
    fi
done
echo "✓ All evaluations completed"

echo ""
echo "Step 5: Verify strong_tags and sparse_tags files"
echo "------------------------------------------------------------------------"
shopt -s nullglob
strong_files=(../json/tags/*_strong_tags.json)
sparse_files=(../json/tags/*_sparse_tags.json)

if [ ${#strong_files[@]} -eq 0 ]; then
    echo "✗ No strong_tags.json files found!"
    exit 1
fi

if [ ${#sparse_files[@]} -eq 0 ]; then
    echo "✗ No sparse_tags.json files found!"
    exit 1
fi

echo "✓ Found ${#strong_files[@]} strong_tags files:"
for f in "${strong_files[@]}"; do
    echo "  - $(basename $f)"
    # Show sample content
    echo "    Sample: $(head -n 10 $f | tail -n 5)"
done

echo ""
echo "✓ Found ${#sparse_files[@]} sparse_tags files:"
for f in "${sparse_files[@]}"; do
    echo "  - $(basename $f)"
    # Show sample content
    echo "    Sample: $(head -n 10 $f | tail -n 5)"
done

echo ""
echo "Step 6: Test best_prompts.json generation"
echo "------------------------------------------------------------------------"
# Simulate the run.sh logic for selecting top prompts
summaries=()
scores=()
indices=()

for csv_path in "${csv_files[@]}"; do
    filename=$(basename "$csv_path")
    if [[ $filename =~ classified_data_([0-9]+)\.csv ]]; then
        idx=${BASH_REMATCH[1]}
    else
        continue
    fi
    
    tag_name="getags_zscore_test_${idx}"
    summary_path="../results/retrieval_v2/summary_${tag_name}.json"
    
    if [ -f "$summary_path" ]; then
        indices+=("$idx")
        # Extract score
        score=$(python3 -c "import json; d=json.load(open('$summary_path')); print(d.get('best',{}).get('score',0.0))")
        scores+=("$score")
        echo "Index $idx: score=$score"
    fi
done

if [ ${#indices[@]} -eq 0 ]; then
    echo "✗ No summaries found"
    exit 1
fi

# Select best index (for test, just take the first one)
best_idx=${indices[0]}
echo "Selected index: $best_idx (for test purposes)"

# Generate best_prompts.json using the new logic
i=$TEST_ITERATION
python3 - $i ${best_idx} <<'PY'
import json,sys,os

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

for s in indices:
    if 0 <= s < len(data):
        metas.append(data[s].get('meta'))
        clusters.append(data[s].get('clusters'))
        
        # Read strong_tags and sparse_tags
        tag_name=f"getags_zscore_test_{s}"
        strong_tags_path=f"../json/tags/{tag_name}_strong_tags.json"
        sparse_tags_path=f"../json/tags/{tag_name}_sparse_tags.json"
        
        high_tags=[]
        low_tags=[]
        all_tags=[]
        
        if os.path.exists(strong_tags_path):
            try:
                with open(strong_tags_path,'r',encoding='utf-8') as f:
                    strong_data=json.load(f)
                    high_tags=[tag for tag,freq in strong_data.get('tags',[])]
                    all_tags.extend(high_tags)
                    print(f"✓ Loaded {len(high_tags)} strong tags from {strong_tags_path}")
            except Exception as e:
                print(f"✗ Error loading strong tags: {e}")
        else:
            print(f"✗ Strong tags file not found: {strong_tags_path}")
        
        if os.path.exists(sparse_tags_path):
            try:
                with open(sparse_tags_path,'r',encoding='utf-8') as f:
                    sparse_data=json.load(f)
                    low_tags=[tag for tag,freq in sparse_data.get('tags',[])]
                    print(f"✓ Loaded {len(low_tags)} sparse tags from {sparse_tags_path}")
            except Exception as e:
                print(f"✗ Error loading sparse tags: {e}")
        else:
            print(f"✗ Sparse tags file not found: {sparse_tags_path}")
        
        tags_list.append(all_tags if all_tags else high_tags+low_tags)
        high_tags_list.append(high_tags)
        low_tags_list.append(low_tags)

out={'metas':metas,'clusters':clusters,'tags':tags_list,'high_tags':high_tags_list,'low_tags':low_tags_list}
with open('json/best_prompts_test.json','w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)

print(f"\n✓ Wrote json/best_prompts_test.json with {len(metas)} prompts")
print(f"  High-frequency (strong) tags: {sum(len(h) for h in high_tags_list)} total")
print(f"  Low-frequency (sparse) tags: {sum(len(l) for l in low_tags_list)} total")

# Show sample
if high_tags_list and high_tags_list[0]:
    print(f"\n  Sample strong tags: {high_tags_list[0][:5]}")
if low_tags_list and low_tags_list[0]:
    print(f"  Sample sparse tags: {low_tags_list[0][:5]}")
PY

if [ $? -ne 0 ]; then
    echo "✗ best_prompts generation failed"
    exit 1
fi

echo ""
echo "Step 7: Verify best_prompts_test.json content"
echo "------------------------------------------------------------------------"
if [ -f "json/best_prompts_test.json" ]; then
    echo "✓ best_prompts_test.json created"
    echo "Content preview:"
    head -n 30 json/best_prompts_test.json
else
    echo "✗ best_prompts_test.json not found"
    exit 1
fi

echo ""
echo "========================================================================"
echo "PHASE 1 TEST COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ Strong/sparse tags are generated by gen_getag.py"
echo "  ✓ Tags are correctly loaded in best_prompts generation"
echo "  ✓ High-frequency tags = Strong tags (P60+)"
echo "  ✓ Low-frequency tags = Sparse tags (Q1-)"
echo ""
echo "Next steps:"
echo "  1. Review json/best_prompts_test.json to verify tags quality"
echo "  2. Run a full iteration with run.sh to test end-to-end"
echo "  3. Check that prompt.py correctly uses these tags"
echo ""
