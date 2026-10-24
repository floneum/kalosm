"""Compare matched profile_forward logs; never launch a model or drop slow samples."""
import argparse
import json
import math
from pathlib import Path
import re


def read_log(path):
    config = None
    records = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('llama_forward_config '):
            if config is not None:
                raise ValueError('multiple configurations in one log')
            config = line
        elif line.startswith('llama_forward_profile '):
            # A prerun precedes the configuration and is not a measured round.
            if config is None:
                continue
            values = dict(re.findall(r'(\w+)=([^ ]+)', line))
            unit = 'tokens' if 'tokens' in values else 'chunks'
            count = int(values[unit])
            if count <= 0 or count != int(values['requested']):
                raise ValueError('incomplete decode sample')
            boundary = 'warmup' if unit == 'tokens' else 'warmup_bytes'
            warmup = int(values[boundary])
            if warmup < (1 if unit == 'tokens' else 0):
                raise ValueError('invalid warmup boundary')
            records.append(dict(unit=unit, **{unit: count, boundary: warmup}))
        elif config is not None and line.startswith('llama_forward_gaps_ms '):
            if not records or 'gaps_ms' in records[-1]:
                raise ValueError('unexpected latency row')
            gaps = json.loads(line.split(' ', 1)[1])
            if not isinstance(gaps, list) or len(gaps) != records[-1][records[-1]['unit']] or not all(
                type(x) in (int, float) and math.isfinite(x) and x > 0 for x in gaps
            ):
                raise ValueError('invalid latency sample')
            records[-1]['gaps_ms'] = gaps
        elif config is not None and line.startswith('llama_forward_token_ids '):
            if not records or records[-1]['unit'] != 'tokens' or 'token_ids' in records[-1]:
                raise ValueError('unexpected token row')
            ids = json.loads(line.split(' ', 1)[1])
            if not isinstance(ids, list) or len(ids) != records[-1]['warmup'] + records[-1]['tokens'] or not all(
                type(token) is int and 0 <= token < 2**32 for token in ids
            ):
                raise ValueError('invalid token IDs')
            records[-1]['token_ids'] = ids
        elif config is not None and line.startswith('llama_forward_output '):
            if not records or 'output' in records[-1]:
                raise ValueError('unexpected output row')
            records[-1]['output'] = line.split(' ', 1)[1]
    if config is None or not records or any(
        'gaps_ms' not in row or 'output' not in row
        or (row['unit'] == 'tokens' and 'token_ids' not in row) for row in records
    ):
        raise ValueError('missing configuration or incomplete decode log')
    if len({row['unit'] for row in records}) != 1:
        raise ValueError('mixed token and chunk measurements')
    repeats = re.search(r'\brepeats=(\d+)\b', config)
    if repeats is None or len(records) != int(repeats[1]):
        raise ValueError('missing benchmark rounds')
    return config, records


def summarize(records, first_round=0):
    """Keep every gap in the statistics; locate the worst five in the source.

    Round and token/chunk positions are zero-based in the original log, including
    any excluded warmup rounds in the round numbering.
    """
    gaps = [gap for row in records for gap in row['gaps_ms']]
    unit = records[0]['unit']
    position = 'token' if unit == 'tokens' else 'chunk'
    ordered = sorted(gaps)
    slowest = sorted((dict(round=round_index, **{position: chunk}, ms=gap)
                      for round_index, row in enumerate(records, first_round)
                      for chunk, gap in enumerate(row['gaps_ms'])),
                     key=lambda point: -point['ms'])[:5]
    result = dict(rounds=len(records), **{unit: len(gaps)}, total_ms=sum(gaps),
                mean_ms=sum(gaps) / len(gaps), p50_ms=ordered[(len(gaps)-1)//2],
                p95_ms=ordered[math.ceil(len(gaps)*.95)-1], max_ms=ordered[-1],
                **{'slowest_' + unit: slowest})
    if unit == 'tokens':
        result['tokens_per_second'] = 1000 * len(gaps) / result['total_ms']
    return result


def compare(baseline, candidate, warmup_rounds=1):
    a_config, a = read_log(baseline)
    b_config, b = read_log(candidate)
    if a_config != b_config or len(a) != len(b):
        raise ValueError('benchmark configurations differ')
    if not 0 <= warmup_rounds < len(a):
        raise ValueError('warmup exclusion leaves no measured rounds')
    for i, (left, right) in enumerate(zip(a, b)):
        if {k: v for k, v in left.items() if k != 'gaps_ms'} != {
            k: v for k, v in right.items() if k != 'gaps_ms'
        }:
            raise ValueError(f'generated output or measurement boundary differs in round {i}')
    metric = ('matched-token decode latency' if a[0]['unit'] == 'tokens'
              else 'matched-output stream-chunk latency')
    a, b = (summarize(rows[warmup_rounds:], warmup_rounds) for rows in [a, b])
    return dict(metric=metric,
                excluded_warmup_rounds=warmup_rounds, baseline=a, candidate=b,
                elapsed_speedup=a['total_ms']/b['total_ms'],
                p95_speedup=a['p95_ms']/b['p95_ms'], max_speedup=a['max_ms']/b['max_ms'])


def compare_bracketed(before, candidate, after, warmup_rounds=1):
    """Compare a candidate with baseline runs on both sides; expose drift.

    Report the smaller improvement for each metric, never the more favorable
    baseline or an average that could hide changing background load. This does
    not by itself establish that the GPU was idle during all three runs.
    """
    first = compare(before, candidate, warmup_rounds)
    last = compare(after, candidate, warmup_rounds)
    durations = [run['baseline']['total_ms'] for run in [first, last]]
    return dict(metric=first['metric'],
                baseline_elapsed_ratio=max(durations)/min(durations),
                elapsed_speedup_min=min(first['elapsed_speedup'], last['elapsed_speedup']),
                p95_speedup_min=min(first['p95_speedup'], last['p95_speedup']),
                max_speedup_min=min(first['max_speedup'], last['max_speedup']),
                before=first, after=last)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--baseline-after', type=Path,
                        help='baseline rerun after the candidate; reports drift and conservative ratios')
    parser.add_argument('--warmup-rounds', type=int, default=1)
    parser.add_argument('--minimum-speedup', type=float,
                        help='fail unless elapsed speedup reaches this target against every baseline')
    parser.add_argument('--minimum-tps', type=float,
                        help='fail unless candidate actual-token throughput reaches this target')
    args = parser.parse_args()
    for name, target in [('minimum-speedup', args.minimum_speedup), ('minimum-tps', args.minimum_tps)]:
        if target is not None and not (math.isfinite(target) and target > 0):
            parser.error(f'--{name} must be finite and positive')
    try:
        if args.baseline_after is None:
            result = compare(args.baseline, args.candidate, args.warmup_rounds)
        else:
            result = compare_bracketed(args.baseline, args.candidate,
                                       args.baseline_after, args.warmup_rounds)
        print(json.dumps(result, indent=2))
        if args.minimum_tps is not None:
            candidate = result.get('before', result)['candidate']
            if 'tokens_per_second' not in candidate:
                raise ValueError('--minimum-tps requires actual token measurements')
            tps = candidate['tokens_per_second']
            if tps < args.minimum_tps:
                parser.exit(1, f'FAIL: {tps:.3f} tokens/sec < {args.minimum_tps:.3f} target\n')
        if args.minimum_speedup is not None:
            measured = result.get('elapsed_speedup_min', result.get('elapsed_speedup'))
            if measured < args.minimum_speedup:
                parser.exit(1, f'FAIL: {measured:.3f}x speedup < {args.minimum_speedup:.3f}x target\n')
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'REFUSED: {error}\n')
