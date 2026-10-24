#!/usr/bin/env python3
"""Show named temporary lifetimes in a generated single-function WGSL shader.

This is source-level diagnostic evidence, not hardware register allocation.
Inline expressions, variables, compiler scheduling and spills are not modeled.
Branches are conservatively treated as one linear instruction sequence.
"""
import argparse
import json
from pathlib import Path
import re

NAME = re.compile(r'\b[A-Za-z_][A-Za-z_0-9]*\b')
LET = re.compile(r'^\s*let\s+(\w+)\s*:\s*([^=]+?)\s*=')
TYPE = re.compile(r'(?:(vec[234])\s*<\s*)?(f32|f16|u32|i32|bool)\s*>?')


def analyze(source):
    # Generated WGSL comments cannot contribute uses of an identifier.
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.S)
    lines = [line.split('//', 1)[0] for line in source.splitlines()]
    functions = re.findall(r'\bfn\s+(\w+)\s*\(', '\n'.join(lines))
    if functions != ['main']:
        raise ValueError('expected generated WGSL with exactly one main function')
    definitions, last_use = {}, {}
    for index, line in enumerate(lines, 1):
        declaration = LET.match(line)
        rhs = line[declaration.end():] if declaration else line
        for token in NAME.findall(rhs):
            if token in definitions:
                last_use[token] = index
        if declaration:
            name, spelling = declaration.groups()
            if name in definitions:
                raise ValueError(f'duplicate temporary name {name}')
            scalar = TYPE.fullmatch(spelling.strip())
            if not scalar:
                raise ValueError(f'unsupported temporary type {spelling}')
            vector, kind = scalar.groups()
            definitions[name] = dict(line=index, words=int(vector[-1]) if vector else 1,
                                     kind=kind, type=spelling.strip())
    if not definitions:
        raise ValueError('no typed temporaries found')
    events = []
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        live = [name for name, item in definitions.items()
                if name in last_use and item['line'] <= index <= last_use[name]]
        if not live:
            continue
        events.append(dict(line=index, named_words=sum(definitions[n]['words'] for n in live),
                           float_words=sum(definitions[n]['words'] for n in live
                                           if definitions[n]['kind'].startswith('f')),
                           live=live, source=line.strip()))
    peak = max(events, key=lambda e: e['named_words'], default=dict(line=0, named_words=0, live=[]))
    return dict(temporaries=len(definitions), unused_temporaries=len(definitions)-len(last_use),
                peak_named_words=peak['named_words'], peak_line=peak['line'], peak_names=peak['live'],
                peak_float_words=max((e['float_words'] for e in events), default=0),
                note='Named WGSL values only; not hardware registers or measured spills. Branches use conservative linear lifetimes.',
                steps=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shader', type=Path, nargs='+')
    parser.add_argument('--steps', action='store_true', help='include every source step and live names')
    args = parser.parse_args()
    for path in args.shader:
        result = analyze(path.read_text())
        if not args.steps:
            result.pop('steps')
        print(json.dumps(dict(shader=str(path), **result)))


if __name__ == '__main__':
    main()
