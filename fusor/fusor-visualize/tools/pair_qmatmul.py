#!/usr/bin/env python3
"""Disabled: two resident GPU processes violate the experiment memory policy."""
raise SystemExit(
    "Disabled after memory exhaustion: two resident benchmark processes are not allowed. "
    "Use sequential guarded runs. Source is archived in "
    "reports/llama-decode-2x/excluded-resource-experiments/."
)
