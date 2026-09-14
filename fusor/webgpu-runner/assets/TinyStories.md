# Embedded TinyStories data

Source: [Ronen Eldan and Yuanzhi Li's TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories),
revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, `TinyStoriesV2-GPT4-train.txt`.
The dataset is distributed under [CDLA-Sharing 1.0](TinyStories-LICENSE.txt).
The data files in this directory retain that license; the software's license is separate.

`tinystories.txt` is a modified, normalized slice: 9,804 complete unique stories,
7,999,444 ASCII character tokens. Preparation maps typographic punctuation to ASCII,
decomposes accents, discards remaining non-ASCII characters, trims lines and removes
blank lines within stories. Three newlines separate stories. The final partial story
in the download is discarded. The training/held-out split falls between stories near
90% of the text. These are portions of the upstream training split, not the upstream
validation benchmark. No synthetic repetition is used to increase the token count.

Reproduce from `fusor/webgpu-runner`:

```sh
curl --fail --location --range 0-8499999 \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-train.txt \
  -o /tmp/tinystories-source.txt
python3 scripts/prepare-corpus.py /tmp/tinystories-source.txt
```

The script checks the source byte range's SHA-256. The prepared file's SHA-256 is
`4b26bc79469080b97a984fd79d3e61ebd5c3b86f7b9cf79fbc463a349b6cc7e2`.

`tinystories-benchmark.txt` preserves the previous 319,868-character demo slice,
including its historical normalization and split, for unchanged compiler performance
and numerical regression comparisons. Its precise upstream byte offset was not
recorded. It is used by the native `train_small` example and optional browser training
checks; production training uses the expanded file.
