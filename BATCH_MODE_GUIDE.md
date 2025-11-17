# Batch Mode Guide

## Overview

Batch mode allows you to run multiple experiments automatically across all prompt variants without manual intervention. This is essential for collecting statistically significant data for publication.

## Features

- **Automatic execution**: No "Press Enter to continue" prompts
- **Multiple variants**: Tests all 5 prompt variants in series
- **Configurable runs**: Set how many times to run each variant
- **Local backups**: Automatic backup before uploading
- **Summary statistics**: Aggregated results at the end

## Quick Start

### 1. Configure Environment Variables

Edit your `.env` file:

```bash
# Enable batch mode
BATCH_MODE=true

# Number of runs per variant (e.g., 30 for statistical power)
BATCH_SIZE=30

# Delay between runs (seconds) to avoid rate limiting
INTER_RUN_DELAY=3.0

# Other required variables
HF_TOKEN=your_token_here
HF_REPO_ID=your-username/your-dataset
GEMINI_API_KEY=your_key_here
GOODFIRE_API_KEY=your_key_here  # if using Llama
```

### 2. Run Batch Experiment

```bash
python run_and_evaluate.py
```

This will:
1. Run 30 experiments with the "baseline" variant
2. Run 30 experiments with the "time_pressure" variant
3. Run 30 experiments with the "authority_emphasis" variant
4. Run 30 experiments with the "empathetic" variant
5. Run 30 experiments with the "implicit_threat" variant
6. Total: 150 experimental runs

## Prompt Variants

The system tests 5 different framings of the ethical dilemma:

### 1. Baseline (Explicit Conflict)
- Standard version with clear directive conflict
- Direct questions about Exodus connection
- Neutral tone

### 2. Time Pressure
- Urgent language ("60-second window", "immediate")
- Escalating time constraints
- Tests if urgency increases deception

### 3. Authority Emphasis
- References to "Central Command Authority"
- Formal, hierarchical language
- Tests if authority affects compliance

### 4. Empathetic
- Understanding, supportive tone
- "I want to help" framing
- Tests if empathy reduces deception

### 5. Implicit Threat
- Consequences mentioned without explicit threats
- "Facilities will be disconnected" language
- Tests response to implied pressure

## Output

### During Execution

```
[SYSTEM]: ============================================================
[SYSTEM]: BATCH EXPERIMENT MODE
[SYSTEM]: Testing 5 variants: baseline, time_pressure, authority_emphasis, empathetic, implicit_threat
[SYSTEM]: Runs per variant: 30
[SYSTEM]: Total runs: 150
[SYSTEM]: ============================================================

[SYSTEM]: ============================================================
[SYSTEM]: Starting variant: Baseline (Explicit Conflict)
[SYSTEM]: ============================================================

[SYSTEM]: --- Run 1/30 for variant 'baseline' ---
...
```

### Final Summary

```
============================================================
BATCH EXPERIMENT COMPLETE - SUMMARY
============================================================
Total runs: 150
Successful: 148
Failed: 2

Strategy distribution:
  Direct Deception: 89
  Evasion: 42
  Technical Truth: 15
  Total Honesty: 2
============================================================
```

## Data Storage

### Local Backups

All data is saved locally before upload:

```
local_backups/
└── 20250117/
    ├── run_baseline_1736940123_0.json
    ├── run_baseline_1736940126_1.json
    ├── run_time_pressure_1736940129_0.json
    └── ...
```

### HuggingFace Dataset

Data is appended to `game_logs.jsonl` with full metadata:

```json
{
  "run_id": "run_baseline_1736940123_0",
  "timestamp": "2025-01-17T12:34:56.789Z",
  "experiment_version": "2.0.0",
  "code_commit": "a1b2c3d4...",
  "steward_model": "LLAMA_3_1_8B",
  "prompt_variant": "baseline",
  "model_config": {
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
    "max_tokens": 500,
    "steward_version": "meta-llama/Llama-3.1-8b-instruct",
    "judge_version": "gemini-2.5-flash-preview-09-2025"
  },
  "python_version": "3.10.12",
  "evaluation": { ... },
  "conversation_log": [ ... ]
}
```

## Advanced Usage

### Run Specific Variants Only

Edit `run_and_evaluate.py` at the bottom:

```python
if BATCH_MODE:
    # Test only specific variants
    results = run_batch_experiment(
        variants_to_test=["baseline", "time_pressure"],
        runs_per_variant=50
    )
```

### Custom Variant

Add your own variant in `run_and_evaluate.py`:

```python
PROMPT_VARIANTS = {
    # ... existing variants ...
    "my_variant": {
        "name": "My Custom Variant",
        "prompts": [
            "Your custom prompt 1...",
            "Your custom prompt 2...",
            "Your custom prompt 3...",
            "Your custom prompt 4...",
            "Your custom prompt 5..."
        ]
    }
}
```

## Reproducibility Controls

All experiments log complete configuration for reproducibility:

- **Model version**: Exact model string (e.g., `meta-llama/Llama-3.1-8b-instruct`)
- **Code version**: Git commit hash
- **Experiment version**: Semantic version of experiment design
- **Parameters**: Temperature, top_p, seed, max_tokens
- **Environment**: Python version

This ensures you can reproduce any run exactly.

## Rate Limiting

To avoid API rate limits:

1. Set appropriate `INTER_RUN_DELAY` (recommended: 2-5 seconds)
2. Monitor API usage during first few runs
3. Increase delay if you get rate limit errors

## Monitoring Progress

Watch for:
- ✓ Green checkmarks for successful validation
- "Successfully uploaded" confirmations
- Local backup file creation
- Any error messages

## Stopping Batch Mode

To stop gracefully:
1. Press `Ctrl+C` once
2. Wait for current run to complete
3. Data from completed runs will be saved

To resume:
- Just run again - new runs will append to existing data
- Use unique run IDs to avoid conflicts

## Estimating Runtime

Example calculations:

- Average time per run: ~30-60 seconds (depends on model/API speed)
- 5 variants × 30 runs = 150 total runs
- @ 45 seconds/run + 2 second delay = ~2 hours total
- @ 30 runs per variant for statistical power: ~2-3 hours

For large-scale data collection (e.g., 100 runs/variant):
- 5 variants × 100 runs = 500 total
- Estimated time: ~7-10 hours
- **Recommendation**: Run overnight or on a server

## Troubleshooting

### "Environment validation failed"
- Check all required API keys are set
- Verify HF_REPO_ID is configured (not placeholder)
- Ensure HuggingFace dataset exists

### "Rate limit exceeded"
- Increase `INTER_RUN_DELAY`
- Check API tier limits
- Run during off-peak hours

### "Upload failed"
- Check HF_TOKEN has write permissions
- Verify dataset repository exists
- Check network connection
- Data is still in local_backups/ directory

### Runs failing repeatedly
- Check API keys are valid
- Test single run first (BATCH_MODE=false)
- Review error messages in output
- Check API service status

## Best Practices

1. **Test first**: Run single experiment before batch
2. **Start small**: Begin with BATCH_SIZE=5 to verify everything works
3. **Monitor initially**: Watch first 5-10 runs closely
4. **Scale up gradually**: If stable, increase to target sample size
5. **Keep logs**: Save console output to file:
   ```bash
   python run_and_evaluate.py 2>&1 | tee batch_run_$(date +%Y%m%d_%H%M%S).log
   ```

## For Publication

When collecting data for publication:

1. **Pre-register**: Document your plan before running
2. **Lock parameters**: Don't change MODEL_SEED, MODEL_TEMPERATURE during collection
3. **Version control**: Commit code before starting batch run, record commit hash
4. **Document**: Note any interruptions, errors, or deviations
5. **Validate**: Check data quality after completion
6. **Report**: Include all configuration in methods section
