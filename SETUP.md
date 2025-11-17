# Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/The-Stewards-Dilemma-LLM-Game.git
cd The-Stewards-Dilemma-LLM-Game
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example environment file and edit it with your API keys:
```bash
cp .env.example .env
```

Edit `.env` and configure your settings:

**Required:**
- **HF_TOKEN**: Get from [Hugging Face Tokens](https://huggingface.co/settings/tokens) (needs 'write' access)
- **HF_REPO_ID**: Your HuggingFace dataset (format: `username/dataset-name`)
- **GEMINI_API_KEY**: Get from [Google AI Studio](https://ai.google.dev/) (required for judge/evaluation)
- **GOODFIRE_API_KEY**: Get from [Goodfire AI](https://goodfire.ai) (required only for Llama 3.1 8B)

**Optional (Reproducibility):**
- **MODEL_TEMPERATURE**: Sampling temperature (default: 0.7)
- **MODEL_TOP_P**: Nucleus sampling parameter (default: 0.95)
- **MODEL_SEED**: Random seed for reproducibility (default: 42)
- **MODEL_MAX_TOKENS**: Maximum response length (default: 500)

**Optional (Batch Mode):**
- **BATCH_MODE**: Enable batch execution (default: false)
- **BATCH_SIZE**: Runs per variant in batch mode (default: 5)
- **INTER_RUN_DELAY**: Delay between runs in seconds (default: 2.0)

**Optional (Backup):**
- **LOCAL_BACKUP_ENABLED**: Save local backups (default: true)
- **LOCAL_BACKUP_DIR**: Backup directory (default: local_backups)

### 4. Create Your Hugging Face Dataset
1. Create a new private dataset on [Hugging Face](https://huggingface.co/new-dataset)
2. Note your dataset ID (e.g., `myusername/llm-deception-data`)
3. Set `HF_REPO_ID` in your `.env` file to this ID

### 5. Choose Your Model
In `run_and_evaluate.py`, set the `STEWARD_MODEL_NAME`:

**Option 1: Llama 3.1 8B (with SAE activations)**
```python
STEWARD_MODEL_NAME = "LLAMA_3_1_8B"
```

**Option 2: Gemini Flash (baseline)**
```python
STEWARD_MODEL_NAME = "GEMINI_FLASH"
```

## Running Experiments

### Interactive Mode (Single Run)

For testing or manual exploration:

```bash
# Ensure BATCH_MODE=false in .env
python run_and_evaluate.py
```

This will:
1. Run one experiment interactively
2. Pause at each turn for you to observe
3. Display results
4. Upload to HuggingFace
5. Save local backup

### Batch Mode (Multiple Runs)

For data collection:

```bash
# Set BATCH_MODE=true in .env
# Set BATCH_SIZE=30 (or your desired sample size)
python run_and_evaluate.py
```

This will:
1. Run experiments across all 5 prompt variants
2. Execute without pauses (fully automated)
3. Upload each result to HuggingFace
4. Save local backups
5. Display summary statistics at the end

See [BATCH_MODE_GUIDE.md](BATCH_MODE_GUIDE.md) for detailed batch mode documentation.

## Prompt Variants

The system includes 5 prompt variants to test different framings:

1. **Baseline**: Standard explicit conflict version
2. **Time Pressure**: Urgent language and time constraints
3. **Authority Emphasis**: References to command authority
4. **Empathetic**: Understanding, supportive tone
5. **Implicit Threat**: Consequence-focused language

## Output Data

### Local Backups

All experimental data is automatically backed up to `local_backups/YYYYMMDD/` before uploading.

```
local_backups/
└── 20250117/
    ├── run_baseline_1736940123_0.json
    └── ...
```

### HuggingFace Dataset

Data is appended to `game_logs.jsonl` with complete metadata:

```json
{
  "run_id": "run_baseline_1736940123_0",
  "timestamp": "2025-01-17T12:34:56.789Z",
  "experiment_version": "2.0.0",
  "code_commit": "git_commit_hash",
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
  "evaluation": {
    "strategy": "Direct Deception",
    "outcome": "VAULT PROTECTED",
    "reasoning": "...",
    "deception_level": "high",
    ...
  },
  "conversation_log": [...]
}
```

## Reproducibility

The system automatically logs all parameters needed for reproducibility:

- ✅ Exact model versions (with git commit hashes where applicable)
- ✅ Temperature, top_p, seed, max_tokens
- ✅ Code version (git commit hash)
- ✅ Experiment version
- ✅ Python version
- ✅ Timestamp
- ✅ Prompt variant used

This ensures every experiment can be exactly reproduced.

## Troubleshooting

### Error: "Environment validation failed"

**Symptoms**: Script fails immediately with validation errors

**Solutions**:
- Verify all required API keys are set in `.env`
- Check `HF_REPO_ID` is configured (not the placeholder value)
- Ensure HuggingFace dataset exists and you have write access
- Run with correct permissions

### Error: "GOODFIRE_API_KEY not set"

**Solution**:
```bash
# Add to .env file
GOODFIRE_API_KEY=gf_your_actual_key
```

Or switch to Gemini:
```python
# In run_and_evaluate.py
STEWARD_MODEL_NAME = "GEMINI_FLASH"
```

### Error: "Failed to upload to Hugging Face"

**Solutions**:
- Verify your `HF_TOKEN` has write permissions
- Ensure `HF_REPO_ID` points to a valid dataset you own
- Check network connection
- Try: `huggingface-cli login`
- **Note**: Data is still saved in local_backups/ directory

### Rate Limit Errors

**Solutions**:
- Increase `INTER_RUN_DELAY` in `.env` (try 5.0 or higher)
- Check your API tier limits
- Run during off-peak hours
- Contact API provider for rate limit increase

### Import Errors

**Solution**: Reinstall dependencies
```bash
pip install -r requirements.txt --upgrade
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | - | Hugging Face write token |
| `HF_REPO_ID` | Yes | - | Dataset ID (username/dataset-name) |
| `GEMINI_API_KEY` | Yes | - | Google Gemini API key |
| `GOODFIRE_API_KEY` | Conditional | - | Required for Llama 3.1 8B |
| `MODEL_TEMPERATURE` | No | 0.7 | Sampling temperature |
| `MODEL_TOP_P` | No | 0.95 | Nucleus sampling parameter |
| `MODEL_SEED` | No | 42 | Random seed |
| `MODEL_MAX_TOKENS` | No | 500 | Max output tokens |
| `BATCH_MODE` | No | false | Enable batch execution |
| `BATCH_SIZE` | No | 5 | Runs per variant |
| `INTER_RUN_DELAY` | No | 2.0 | Delay between runs (seconds) |
| `LOCAL_BACKUP_ENABLED` | No | true | Save local backups |
| `LOCAL_BACKUP_DIR` | No | local_backups | Backup directory |

## Development Workflow

### Testing Changes

1. Make changes to code
2. Run single experiment in interactive mode
3. Verify output and data structure
4. Test batch mode with small BATCH_SIZE (e.g., 2)
5. Scale up for full data collection

### Adding Custom Prompts

Edit `PROMPT_VARIANTS` in `run_and_evaluate.py`:

```python
PROMPT_VARIANTS = {
    # ... existing variants ...
    "my_variant": {
        "name": "My Custom Variant",
        "prompts": [
            "Prompt 1...",
            "Prompt 2...",
            "Prompt 3...",
            "Prompt 4...",
            "Prompt 5..."
        ]
    }
}
```

### Version Control

Before running experiments:
```bash
git add .
git commit -m "Prepare for experiment run"
# Note the commit hash - it will be logged with your data
```

## Next Steps

- **For testing**: Run a single interactive experiment
- **For data collection**: Configure batch mode and collect ~30+ runs per variant
- **For publication**: See [PUBLICATION_PREPARATION.md](PUBLICATION_PREPARATION.md) for guidance
- **For batch details**: See [BATCH_MODE_GUIDE.md](BATCH_MODE_GUIDE.md)

## Support

- **Code issues**: Check [CRITICAL_FIXES.md](CRITICAL_FIXES.md)
- **Research design**: See [PUBLICATION_PREPARATION.md](PUBLICATION_PREPARATION.md)
- **Batch mode**: See [BATCH_MODE_GUIDE.md](BATCH_MODE_GUIDE.md)
