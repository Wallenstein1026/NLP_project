# Processed Data

Full data files are not included in this GitHub package.

After downloading the public datasets listed in the root `README.md`, place processed JSONL files at:

```text
data/processed/sciq/merged_fb.json
data/processed/simple_questions_wiki/merged_fb.json
data/processed/nq/merged_fb.json
data/processed/truthfulQA/merged_fb.json
```

Each line should be a JSON object with at least:

```json
{"question": "...", "correct_answer": "..."}
```

The project used filenames ending in `.json`, but the pipeline reads these files as JSONL.
