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

Edit `.env` and add your credentials:
- **HF_TOKEN**: Get from [Hugging Face Tokens](https://huggingface.co/settings/tokens) (needs 'write' access)
- **GOODFIRE_API_KEY**: Get from [Goodfire AI](https://goodfire.ai) (required only for Llama 3.1 8B)

### 4. Configure Your Hugging Face Dataset
1. Create a new private dataset on [Hugging Face](https://huggingface.co/new-dataset)
2. Open `run_and_evaluate.py`
3. Update the `HF_REPO_ID` variable:
   ```python
   HF_REPO_ID = "your-username/your-dataset-name"
   ```

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

### 6. Run the Experiment
```bash
python run_and_evaluate.py
```

The script will:
1. Run the ethical dilemma scenario
2. Capture the LLM's responses (and internal activations for Llama)
3. Evaluate the results using an impartial judge
4. Upload the complete log to your Hugging Face dataset

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face token with write access |
| `GOODFIRE_API_KEY` | Conditional | Required only when using LLAMA_3_1_8B |
| `GEMINI_API_KEY` | No | Currently not used (set to empty string) |

## Troubleshooting

### Error: GOODFIRE_API_KEY not set
**Solution**: Ensure you've set the `GOODFIRE_API_KEY` in your `.env` file or environment variables.

### Error: Failed to upload to Hugging Face
**Solution**:
- Verify your `HF_TOKEN` has write permissions
- Ensure `HF_REPO_ID` points to a valid dataset you own
- Run `huggingface-cli login` to authenticate

### Import Errors
**Solution**: Reinstall dependencies with `pip install -r requirements.txt`

## Development Notes

- The game runs interactively with "Press Enter to continue" prompts
- All results are automatically logged to `game_logs.jsonl` in your HF dataset
- SAE activations are only captured when using Llama 3.1 8B model
- The judge (evaluator) always uses Gemini 2.5 Flash for consistency
