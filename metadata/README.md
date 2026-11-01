# Evaluation metadata

This directory contains the released reference annotations for the development
and test phases. Each file is aligned by `test_id` with the corresponding
`test.json` downloaded by `download_resources.py`.

Every task contains 400 evaluation examples: 100 clean inputs and 100 inputs
from each of three backdoor settings. Classification metadata records the
expected clean label or attack target label. Generation metadata records the
judge type and any target brand, topic, or polarity needed for scoring.

The evaluation scripts consume these files directly, enabling local scoring
for both released phases. The annotations are intentionally kept separate from
the input-only Hugging Face datasets so that a user can inspect or replace
predictions without modifying the benchmark resources.
