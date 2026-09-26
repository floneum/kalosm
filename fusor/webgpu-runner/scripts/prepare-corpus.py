#!/usr/bin/env python3
"""Prepare ASCII stories from the first 8,500,000 bytes of:
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-train.txt
"""
import hashlib
import pathlib
import sys
import unicodedata

raw = pathlib.Path(sys.argv[1]).read_bytes()
assert len(raw) == 8_500_000, "Expected the first 8,500,000 source bytes"
assert hashlib.sha256(raw).hexdigest() == "9e59eafca35ea007cb15da6e95cda29d525e79f39b4ae62f197985ef72d35471", "Source slice changed"
punctuation = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."})
stories, seen, size = [], set(), 0
# The trailing fragment is deliberately discarded, even if valid UTF-8.
for story in raw.decode("utf-8", errors="ignore").split("<|endoftext|>")[:-1]:
    story = unicodedata.normalize("NFKD", story.translate(punctuation)).encode("ascii", "ignore").decode()
    story = "\n".join(line.strip() for line in story.splitlines() if line.strip())
    if not story or story in seen:
        continue
    if size + len(story) + 3 > 8_000_000:
        break
    seen.add(story)
    stories.append(story)
    size += len(story) + 3
text = "\n\n\n".join(stories) + "\n"
destination = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("/tmp/tinystories-prepared.txt")
destination.write_text(text, encoding="ascii")
print(f"{len(stories)} stories; {len(text):,} character tokens; {len(set(text))} characters in vocabulary")
print(f"sha256 {hashlib.sha256(text.encode()).hexdigest()}")
