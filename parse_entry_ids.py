#!/usr/bin/env python3
"""entry_id Parser — Full Lineage Analysis Pipeline

Usage: python3 parse_entry_ids.py [logfile]

Model-separated analysis of KVPrefixCache entry lifecycle.
Outputs machine-readable JSON + human summary.

Contract: deepseek-v4-cache-capture-protocol (Analysis Outputs)
"""
import json, re, sys, os
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

logfile = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".exo/exo_log/exo.log")
host = os.uname().nodename

# Patterns
ENTRY_ID_RE = re.compile(r'entry_id=([a-f0-9]+):(\d+)')
TS_RE = re.compile(r'\[\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s*\|')
TOKENS_RE = re.compile(r'(\d+)\s+tokens|tokens=(\d+)')
VISION_RE = re.compile(r'vision|image|pipeline', re.IGNORECASE)
SNAPSHOT_RE = re.compile(r'snapshots=\[([^\]]+)\]')
SNAPSHOT_BYTES_RE = re.compile(r'snapshot_bytes=([\d.]+)(MiB|GiB)')
CACHE_BYTES_RE = re.compile(r'cache=([\d.]+)(MiB|GiB)')
RAW_PREFIX_RE = re.compile(r'raw=(\d+)')
VALIDATED_RE = re.compile(r'validated=(\d+)')
RESTORE_RE = re.compile(r'restore=(\d+)')
CACHED_RE = re.compile(r'cached=(\d+)')
LAST_USED_RE = re.compile(r'last_used=(\d+)')
PREFILL_START_RE = re.compile(r'Prefill start:\s*(\d+)\s*tokens')
PREFILL_COMPLETE_RE = re.compile(r'Prefill complete:\s*(\d+)\s*tokens.*?(\d+\.\d+)s.*?(\d+\.\d+)\s*tok/s')
PREFILL_PROGRESS_RE = re.compile(r'Prefill progress:\s*(\d+)/(\d+)\s*tokens')

entries = {}
lineages = defaultdict(list)
instance_count = defaultdict(int)
instance_model = {}
snapshot_data = defaultdict(list)
request_data = []
anomalies = []
prefill_starts = {}
prefill_comp = {}

def parse_bytes(s, unit):
    return float(s) * (1024**3 if unit == 'GiB' else 1024**2)

def fmt_ts(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except:
        return ts_str

with open(logfile) as f:
    lines = f.readlines()

# Pass 1: classify instances
for i, line in enumerate(lines):
    m = ENTRY_ID_RE.search(line)
    if not m:
        continue
    inst = m.group(1)
    if inst in instance_model:
        continue
    start = max(0, i - 50)
    end = min(len(lines), i + 20)
    ctx = " ".join(lines[start:end])
    instance_model[inst] = 'vision' if VISION_RE.search(ctx) else 'deepseek'

# Pass 2: parse events
for i, line in enumerate(lines):
    ts_m = TS_RE.match(line)
    ts = ts_m.group(1) if ts_m else "unknown"
    
    m = ENTRY_ID_RE.search(line)
    if not m:
        continue
    
    inst, gen = m.group(1), int(m.group(2))
    model = instance_model.get(inst, 'unknown')
    target = entries
    key = (inst, gen)
    instance_count[inst] += 1
    
    tok_m = TOKENS_RE.search(line)
    tokens = int(tok_m.group(1) or tok_m.group(2)) if tok_m else 0
    
    # Parse snapshot data
    snap_m = SNAPSHOT_RE.search(line)
    snap_str = snap_m.group(1) if snap_m else ""
    
    raw_m = RAW_PREFIX_RE.search(line)
    val_m = VALIDATED_RE.search(line)
    rest_m = RESTORE_RE.search(line)
    cach_m = CACHED_RE.search(line)
    lu_m = LAST_USED_RE.search(line)
    sb_m = SNAPSHOT_BYTES_RE.search(line)
    cb_m = CACHE_BYTES_RE.search(line)
    
    snap_bytes = parse_bytes(sb_m.group(1), sb_m.group(2)) if sb_m else None
    cache_bytes = parse_bytes(cb_m.group(1), cb_m.group(2)) if cb_m else None
    raw_prefix = int(raw_m.group(1)) if raw_m else None
    validated = int(val_m.group(1)) if val_m else None
    restore = int(rest_m.group(1)) if rest_m else None
    cached = int(cach_m.group(1)) if cach_m else None
    last_used = int(lu_m.group(1)) if lu_m else None
    
    event = {
        'line': i + 1, 'ts': ts, 'model': model,
        'instance': inst, 'generation': gen,
        'tokens': tokens, 'raw_prefix': raw_prefix,
        'validated': validated, 'restore': restore,
        'cached': cached, 'last_used': last_used,
        'snapshot_bytes': snap_bytes, 'cache_bytes': cache_bytes,
        'snapshot_anchors': snap_str
    }
    
    if 'cache added' in line.lower():
        event['type'] = 'add'
        target[key] = {'model': model, 'added': ts, 'tokens': tokens,
                       'updates': 0, 'selections': 0, 'evicted': None,
                       'snapshot_bytes': [], 'cache_bytes': []}
        if snap_bytes: target[key]['snapshot_bytes'].append(snap_bytes)
        if cache_bytes: target[key]['cache_bytes'].append(cache_bytes)
        lineages[inst].append(event)
    elif 'cache updated' in line.lower():
        event['type'] = 'update'
        if key in target:
            target[key]['updates'] += 1
            if snap_bytes: target[key]['snapshot_bytes'].append(snap_bytes)
            if cache_bytes: target[key]['cache_bytes'].append(cache_bytes)
        lineages[inst].append(event)
    elif 'evict' in line.lower() and 'entry_id' in line:
        event['type'] = 'evict'
        if key in target:
            target[key]['evicted'] = ts
        lineages[inst].append(event)
    elif 'candidate' in line.lower():
        event['type'] = 'select'
        if key in target:
            target[key]['selections'] += 1
        lineages[inst].append(event)
    else:
        event['type'] = 'unknown'
        lineages[inst].append(event)

# Pass 3: collect prefill timing
for i, line in enumerate(lines):
    ps = PREFILL_START_RE.search(line)
    if ps:
        prefill_starts[i] = int(ps.group(1))
    pc = PREFILL_COMPLETE_RE.search(line)
    if pc:
        prefill_comp[i] = (int(pc.group(1)), float(pc.group(2)), float(pc.group(3)))

# Build output
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
            print(f"\n  ▸ ACTIVE:")
            for k, v in sorted(inst_active.items(), key=lambda x: x[0][1]):
                snap_gb = sum(v['snapshot_bytes']) / 1024**3 if v['snapshot_bytes'] else 0
                print(f"    gen {k[1]:3d} | {v['tokens']:>6,} tok | {v['updates']:>2} upd | "
                      f"{v['selections']:>2} sel | snap={snap_gb:.1f}GiB | added {v['added'][11:19]}")
        if inst_evicted:
            print(f"\n  ▸ EVICTED (all {len(inst_evicted)}):")
            for k, v in sorted(inst_evicted.items(), key=lambda x: x[1]['added']):
                life = f"{v['evicted'][11:19]}"
                snap_gb = sum(v['snapshot_bytes']) / 1024**3 if v['snapshot_bytes'] else 0
                print(f"    gen {k[1]:3d} | {v['tokens']:>6,} tok | {v['updates']:>2} upd | "
                      f"{v['selections']:>2} sel | snap={snap_gb:.1f}GiB | "
                      f"{v['added'][11:19]} → {life}")
        
        # Timeline
        print(f"\n  ▸ TIMELINE ({len(lineages[inst_id])} events):")
        for ev in lineages[inst_id]:
            icon = {'add': '+', 'update': '~', 'select': '?', 'evict': '-'}.get(ev['type'], ' ')
            tok_s = f" {ev['tokens']}tok" if ev['tokens'] else ""
            rp = f" raw={ev['raw_prefix']}" if ev['raw_prefix'] else ""
            print(f"    {icon} gen {ev['generation']:3d} | {ev['ts'][11:19]}{tok_s}{rp}")
        
        # Displacement analysis
        if len(inst_evicted) >= 2:
            print(f"\n  ▸ DISPLACEMENT SEQUENCE:")
            sorted_ev = sorted(inst_evicted.items(), key=lambda x: x[1]['added'])
            for j, (k, v) in enumerate(sorted_ev):
                displacer = ""
                if j < len(sorted_ev) - 1:
                    nk, nv = sorted_ev[j + 1]
                    if v['evicted'] and nv['added']:
                        if abs(fmt_ts(v['evicted']) - fmt_ts(nv['added'])).total_seconds() < 60:
                            displacer = f" ← displaced by gen {nk[1]}"
                print(f"    gen {k[1]:3d} | {v['tokens']:>6,} tok | {v['selections']:>2} sel | "
                      f"alive {v['added'][11:19]}→{v['evicted'][11:19]}{displacer}")

print("=" * 72)
print(f"ENTRY-ID PARSER v2 — Full Analysis Pipeline")
print(f"Host: {host}  |  Log: {logfile}")
print(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 72)

summarize({k: v for k, v in entries.items() if v['model'] == 'deepseek'}, "DeepSeek V4 Cache")
summarize({k: v for k, v in entries.items() if v['model'] == 'vision'}, "Vision/Image Cache")

print(f"\n{'='*72}")
print("ANOMALIES")
print(f"{'='*72}")
if anomalies:
    for a in anomalies:
        print(f"  {a}")
else:
    print("  None detected — all transitions valid.")

# Machine-readable JSON
output = {
    'meta': {
        'parser_version': 2,
        'generated': datetime.now(timezone.utc).isoformat(),
        'host': host,
        'logfile': str(logfile),
        'commit': '78fc60d4',
        'protocol': 'deepseek-v4-cache-capture-protocol'
    },
    'deepseek': {
        inst_id: {
            'entries': {
                f"{k[0]}:{k[1]}": {
                    'generation': k[1],
                    'added': v['added'],
                    'evicted': v['evicted'],
                    'tokens': v['tokens'],
                    'updates': v['updates'],
                    'selections': v['selections'],
                    'snapshot_bytes_gib': [b/1024**3 for b in v['snapshot_bytes']],
                    'cache_bytes_gib': [b/1024**3 for b in v['cache_bytes']]
                }
                for k, v in entries.items() if k[0] == inst_id and v['model'] == 'deepseek'
            },
            'events': [ev for ev in lineages[inst_id]]
        }
        for inst_id in set(k[0] for k, v in entries.items() if v['model'] == 'deepseek')
    },
    'vision': {
        inst_id: {
            'entries': {
                f"{k[0]}:{k[1]}": {
                    'generation': k[1],
                    'added': v['added'],
                    'evicted': v['evicted'],
                    'tokens': v['tokens'],
                    'updates': v['updates'],
                    'selections': v['selections']
                }
                for k, v in entries.items() if k[0] == inst_id and v['model'] == 'vision'
            },
            'events': [ev for ev in lineages[inst_id]]
        }
        for inst_id in set(k[0] for k, v in entries.items() if v['model'] == 'vision')
    },
    'anomalies': anomalies
}

outpath = Path(logfile).parent / "parse_entry_ids_output.json"
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nJSON: {outpath} ({os.path.getsize(outpath)} bytes)")

# Pass 4: per-request cache efficiency
# Re-scan for candidate events with raw/validated/restore fields
request_metrics = []
with open(logfile) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'candidates' not in line.lower() and 'prefill' not in line.lower():
        continue
    
    ts_m = TS_RE.match(line)
    ts = ts_m.group(1) if ts_m else "unknown"
    
    m = ENTRY_ID_RE.search(line)
    inst = m.group(1) if m else None
    gen = int(m.group(2)) if m else None
    model = instance_model.get(inst, 'unknown') if inst else 'unknown'
    if model != 'deepseek':
        continue
    
    raw_m = RAW_PREFIX_RE.search(line)
    val_m = VALIDATED_RE.search(line)
    rest_m = RESTORE_RE.search(line)
    cach_m = CACHED_RE.search(line)
    tok_m = TOKENS_RE.search(line)
    
    raw_prefix = int(raw_m.group(1)) if raw_m else None
    validated = int(val_m.group(1)) if val_m else None
    restore = int(rest_m.group(1)) if rest_m else None
    cached = int(cach_m.group(1)) if cach_m else None
    tokens = int(tok_m.group(1) or tok_m.group(2)) if tok_m else None
    
    if 'candidates' in line.lower() and raw_prefix and tokens:
        backfill = validated - restore if (validated is not None and restore is not None) else None
        new_suffix = tokens - validated if (validated is not None and tokens) else None
        total_computed = tokens - restore if (restore is not None and tokens) else None
        raw_ratio = raw_prefix / tokens * 100 if tokens else None
        val_ratio = validated / tokens * 100 if (validated is not None and tokens) else None
        restore_eff = restore / validated * 100 if (validated and restore is not None) else None
        
        request_metrics.append({
            'ts': ts,
            'entry_id': f"{inst}:{gen}" if inst else None,
            'request_tokens': tokens,
            'raw_prefix': raw_prefix,
            'raw_ratio': round(raw_ratio, 2) if raw_ratio else None,
            'validated_prefix': validated,
            'validated_ratio': round(val_ratio, 2) if val_ratio else None,
            'restore_tokens': restore,
            'restore_efficiency': round(restore_eff, 2) if restore_eff else None,
            'checkpoint_backfill': backfill,
            'new_suffix': new_suffix,
            'total_computed': total_computed,
            'cached_tokens': cached
        })

print(f"\n{'='*72}")
print(f"PER-REQUEST CACHE EFFICIENCY (last 20 DeepSeek requests)")
print(f"{'='*72}")
print(f"{'Time':>10} | {'Entry':>8} | {'ReqTok':>7} | {'Raw%':>6} | {'Val%':>6} | {'Rest%':>6} | {'Backfill':>8} | {'Suffix':>7}")
print(f"{'-'*10} | {'-'*8} | {'-'*7} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*7}")
for rm in request_metrics[-20:]:
    raw_s = f"{rm['raw_ratio']:.1f}%" if rm['raw_ratio'] else " —  "
    val_s = f"{rm['validated_ratio']:.1f}%" if rm['validated_ratio'] else " —  "
    rest_s = f"{rm['restore_efficiency']:.1f}%" if rm['restore_efficiency'] else " —  "
    bf = f"{rm['checkpoint_backfill']:>6}" if rm['checkpoint_backfill'] is not None else "   —  "
    sfx = f"{rm['new_suffix']:>5}" if rm['new_suffix'] is not None else "   —"
    print(f"{rm['ts'][11:19]:>10} | {rm['entry_id']:>8} | {rm['request_tokens']:>7} | {raw_s:>6} | {val_s:>6} | {rest_s:>6} | {bf:>8} | {sfx:>7}")

# Aggregate summary
if request_metrics:
    print(f"\n  AGGREGATE ({len(request_metrics)} requests):")
    raw_ratios = [r['raw_ratio'] for r in request_metrics if r['raw_ratio']]
    val_ratios = [r['validated_ratio'] for r in request_metrics if r['validated_ratio']]
    rest_effs = [r['restore_efficiency'] for r in request_metrics if r['restore_efficiency']]
    backfills = [r['checkpoint_backfill'] for r in request_metrics if r['checkpoint_backfill'] is not None]
    suffixes = [r['new_suffix'] for r in request_metrics if r['new_suffix'] is not None]
    
    if raw_ratios:
        print(f"  Raw prefix match:   mean={sum(raw_ratios)/len(raw_ratios):.1f}%  min={min(raw_ratios):.1f}%  max={max(raw_ratios):.1f}%")
    if val_ratios:
        print(f"  Validated prefix:   mean={sum(val_ratios)/len(val_ratios):.1f}%  min={min(val_ratios):.1f}%  max={max(val_ratios):.1f}%")
    if rest_effs:
        print(f"  Restore efficiency: mean={sum(rest_effs)/len(rest_effs):.1f}%  min={min(rest_effs):.1f}%  max={max(rest_effs):.1f}%")
    if backfills:
        print(f"  Checkpoint backfill: mean={sum(backfills)/len(backfills):.0f}  min={min(backfills)}  max={max(backfills)}")
    if suffixes:
        print(f"  New suffix tokens:  mean={sum(suffixes)/len(suffixes):.0f}  min={min(suffixes)}  max={max(suffixes)}")

# Add to JSON output
output['per_request'] = request_metrics

# Write updated JSON
with open(outpath, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nJSON updated: {outpath} ({os.path.getsize(outpath)} bytes)")
