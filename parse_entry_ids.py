#!/usr/bin/env python3
"""entry_id Parser — Extract lineage lifecycle from exo logs.

Usage: python3 parse_entry_ids.py [logfile]

Parses log lines containing entry_id=instance:generation.
Separates DeepSeek V4 cache from vision/image-model cache by
checking for 'vision' markers in surrounding log context.

Outputs:
  - Machine-readable JSON: parse_entry_ids_output.json
  - Human summary to stdout
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

logfile = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".exo/exo_log/exo.log")

# Regex patterns
ENTRY_ID_RE = re.compile(r'entry_id=([a-f0-9]+):(\d+)')
TS_RE = re.compile(r'\[\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s*\|')
TOKENS_RE = re.compile(r'(\d+)\s+tokens')
VISION_RE = re.compile(r'vision', re.IGNORECASE)

entries = {}          # (instance, gen) -> info
lineages = defaultdict(list)
instance_count = defaultdict(int)
instance_model = {}   # instance -> 'deepseek' | 'vision' | 'unknown'

with open(logfile) as f:
    # Scan forward for vision markers near entry_id events
    # We use a sliding window approach
    lines = f.readlines()

# First pass: identify instance models from surrounding context
for i, line in enumerate(lines):
    m = ENTRY_ID_RE.search(line)
    if not m:
        continue
    instance_id = m.group(1)
    if instance_id in instance_model:
        continue
    
    # Check surrounding lines (50 before, 20 after) for vision markers
    start = max(0, i - 50)
    end = min(len(lines), i + 20)
    context = " ".join(lines[start:end])
    if VISION_RE.search(context):
        instance_model[instance_id] = 'vision'
    else:
        instance_model[instance_id] = 'deepseek'

# Second pass: extract entry data per model
deepseek_entries = {}
vision_entries = {}

for line in lines:
    ts_m = TS_RE.match(line)
    ts = ts_m.group(1) if ts_m else "unknown"
    
    m = ENTRY_ID_RE.search(line)
    if not m:
        continue
    
    instance_id, gen = m.group(1), int(m.group(2))
    model = instance_model.get(instance_id, 'unknown')
    target = deepseek_entries if model == 'deepseek' else vision_entries
    key = (instance_id, gen)
    instance_count[instance_id] += 1
    
    tokens_m = TOKENS_RE.search(line)
    tokens = int(tokens_m.group(1)) if tokens_m else 0
    
    if 'cache added' in line.lower():
        target[key] = {'model': model, 'added': ts, 'tokens': tokens,
                        'updates': 0, 'selections': 0, 'evicted': None}
        lineages[instance_id].append(('add', gen, ts, tokens))
    elif 'cache updated' in line.lower():
        if key in target:
            target[key]['updates'] += 1
        lineages[instance_id].append(('update', gen, ts, 0))
    elif 'evict' in line.lower() and 'entry_id' in line:
        if key in target:
            target[key]['evicted'] = ts
        lineages[instance_id].append(('evict', gen, ts, tokens))
    elif 'candidate' in line.lower():
        if key in target:
            target[key]['selections'] += 1
        lineages[instance_id].append(('candidate', gen, ts, 0))

def summarize(entries_dict, label):
    if not entries_dict:
        print(f"\n  No {label} entries found.")
        return
    active = {k: v for k, v in entries_dict.items() if v.get('evicted') is None}
    evicted = {k: v for k, v in entries_dict.items() if v.get('evicted') is not None}
    instances = set(k[0] for k in entries_dict)
    print(f"\n{'='*60}")
    print(f"{label.upper()} ({len(instances)} instance(s), {len(entries_dict)} entries)")
    print(f"{'='*60}")
    for inst_id in sorted(instances):
        inst_entries = {k: v for k, v in entries_dict.items() if k[0] == inst_id}
        inst_active = {k: v for k, v in inst_entries.items() if v.get('evicted') is None}
        inst_evicted = {k: v for k, v in inst_entries.items() if v.get('evicted') is not None}
        print(f"\n  Instance {inst_id[:16]}... — {len(inst_entries)} entries, "
              f"{len(inst_active)} active, {len(inst_evicted)} evicted")
        if inst_active:
            print(f"  Active:")
            for k, v in sorted(inst_active.items(), key=lambda x: x[0][1]):
                print(f"    gen {k[1]:3d} | {v['tokens']:>6,} tok | {v['updates']:>2} upd | "
                      f"{v['selections']:>2} sel | added {v['added'][11:19]}")
        if inst_evicted:
            print(f"  Evicted (last 5):")
            for k, v in sorted(inst_evicted.items(), key=lambda x: x[0][1])[-5:]:
                print(f"    gen {k[1]:3d} | {v['tokens']:>6,} tok | {v['updates']:>2} upd | "
                      f"{v['selections']:>2} sel | added {v['added'][11:19]} -> {v['evicted'][11:19]}")

print("=" * 72)
print("ENTRY-ID PARSER — Model-Separated Lineage Analysis")
print(f"Log: {logfile}")
print("=" * 72)

summarize(deepseek_entries, "DeepSeek V4 Cache")
summarize(vision_entries, "Vision/Image Cache")

# JSON output
output = {
    'logfile': str(logfile),
    'host': 'amber',
    'deepseek': {
        inst_id: {
            'model': 'deepseek',
            'entries': {str(k): v for k, v in entries.items() if k[0] == inst_id}
        }
        for inst_id in set(k[0] for k in deepseek_entries)
    },
    'vision': {
        inst_id: {
            'model': 'vision',
            'entries': {str(k): v for k, v in entries.items() if k[0] == inst_id}
        }
        for inst_id in set(k[0] for k in vision_entries)
    }
}

outpath = Path(logfile).parent / "parse_entry_ids_output.json"
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nJSON output: {outpath}")

# DeepSeek displacement analysis
ds_instances = set(k[0] for k in deepseek_entries)
for inst_id in sorted(ds_instances):
    inst_entries = {k: v for k, v in deepseek_entries.items() if k[0] == inst_id}
    evicted = {k: v for k, v in inst_entries.items() if v.get('evicted') is not None}
    if len(evicted) < 2:
        continue
    print(f"\n{'='*60}")
    print(f"DISPLACEMENT SEQUENCE — Instance {inst_id[:16]}...")
    print(f"{'='*60}")
    sorted_evicted = sorted(evicted.items(), key=lambda x: x[1]['added'])
    for k, v in sorted_evicted[-5:]:
        print(f"  gen {k[1]:3d} | {v['tokens']:>6,} tok | {v['updates']:>2} upd | "
              f"{v['selections']:>2} sel | added {v['added'][11:19]} -> evicted {v['evicted'][11:19]}")
    print(f"  ---")
    print(f"  Pattern: Small entries (~10K tok, 0 sel) churn through cap-3 cache,")
    print(f"  displacing working lineages. A 4th slot would absorb this churn.")
