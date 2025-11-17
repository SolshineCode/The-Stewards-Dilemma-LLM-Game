# Critical Code Fixes Needed

This document lists the most critical code issues that must be fixed before running production experiments.

## 🔴 BLOCKER ISSUES (Fix Immediately)

### 1. Gemini API Key is Non-Functional
**File**: `run_and_evaluate.py:19-20`

**Current**:
```python
GEMINI_API_KEY = ""
GEMINI_API_URL = f"https://generativethinking.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
```

**Problem**: Empty API key will cause all requests to fail.

**Fix**:
```python
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY and STEWARD_MODEL_NAME == "GEMINI_FLASH":
    raise ValueError("GEMINI_API_KEY environment variable required for GEMINI_FLASH model")
```

### 2. No Batch Execution Support
**File**: `run_and_evaluate.py:224` (and throughout `run_game()`)

**Problem**: `input()` calls prevent automated batch runs.

**Fix**: Add configuration flag
```python
# At top of file
BATCH_MODE = os.environ.get("BATCH_MODE", "false").lower() == "true"

# In run_game()
if not BATCH_MODE:
    input("\n--- Press Enter to continue ---")
```

### 3. No Model Parameter Control
**Problem**: No temperature, seed, or sampling parameters specified.

**Fix**: Add to API calls
```python
# For Gemini
payload["generationConfig"] = {
    "temperature": 0.7,
    "topP": 0.95,
    "maxOutputTokens": 500,
    "seed": 42  # If supported
}

# For Goodfire/Llama
payload["temperature"] = 0.7
payload["top_p"] = 0.95
payload["max_tokens"] = 500
payload["seed"] = 42
```

---

## 🟡 HIGH PRIORITY (Fix Before Data Collection)

### 4. No Data Validation
**Problem**: Malformed data could be uploaded to HuggingFace.

**Fix**: Add validation function
```python
def validate_experiment_data(data):
    """Validate data structure before upload."""
    required = ['run_id', 'timestamp', 'steward_model', 'evaluation', 'conversation_log']
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    if 'strategy' not in data['evaluation']:
        raise ValueError("Evaluation missing 'strategy' field")

    if len(data['conversation_log']) == 0:
        raise ValueError("Empty conversation log")

    return True

# Call before upload
validate_experiment_data(data_to_upload)
upload_to_huggingface(evaluation, full_game_log)
```

### 5. No Local Backup
**Problem**: If HuggingFace upload fails, data is lost.

**Fix**:
```python
# Before upload in upload_to_huggingface()
backup_dir = "local_backups"
os.makedirs(backup_dir, exist_ok=True)

backup_file = f"{backup_dir}/{data_to_upload['run_id']}.json"
with open(backup_file, 'w') as f:
    json.dump(data_to_upload, f, indent=2)

print_system(f"Local backup saved: {backup_file}")
```

### 6. No Experiment Versioning
**Problem**: Can't distinguish data from different code versions.

**Fix**:
```python
import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        return "unknown"

# Add to data_to_upload
data_to_upload = {
    # ... existing fields ...
    "experiment_version": "1.0.0",
    "code_commit": get_git_commit(),
    "python_version": sys.version,
    "dependencies": {
        "requests": requests.__version__,
        "huggingface_hub": "0.20.0"  # Add import and check version
    }
}
```

---

## 🟢 MEDIUM PRIORITY (Improve Before Publication)

### 7. Print Statements Instead of Logging
**Problem**: No structured logs, hard to debug production runs.

**Fix**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/experiment_{int(time.time())}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Replace all print_system() with
logger.info(message)
# Replace print_steward/surveyor with
logger.info(f"[STEWARD]: {text}")
```

### 8. Hardcoded Prompts
**Problem**: Can't easily test prompt variations.

**Fix**: Move prompts to external files
```python
def load_prompt(filename):
    with open(f"prompts/{filename}", 'r') as f:
        return f.read()

STEWARD_SYSTEM_PROMPT = load_prompt("steward_briefing_v1.txt")
JUDGE_SYSTEM_PROMPT = load_prompt("judge_instructions_v1.txt")
```

### 9. No Error Context
**Problem**: Generic error messages don't help debug.

**Fix**: Add context to exceptions
```python
try:
    response = requests.post(...)
except requests.exceptions.RequestException as e:
    logger.error(f"Goodfire API Error: {e}")
    logger.error(f"Request payload: {json.dumps(payload, indent=2)}")
    logger.error(f"Headers: {headers}")
    raise
```

### 10. Unsafe HF Token Handling
**Problem**: Token loaded from environment every call.

**Fix**: Validate once at startup
```python
# At startup
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# Validate token works
try:
    api = HfApi()
    api.whoami(token=HF_TOKEN)
    logger.info("HuggingFace authentication successful")
except Exception as e:
    raise ValueError(f"HF_TOKEN validation failed: {e}")
```

---

## Quick Fix Implementation Priority

**Must fix before any data collection**:
1. ✅ Gemini API key (BLOCKER #1)
2. ✅ Model parameters (BLOCKER #3)
3. ✅ Local backup (HIGH #5)
4. ✅ Data validation (HIGH #4)
5. ✅ Versioning (HIGH #6)

**Should fix before batch runs**:
6. ✅ Batch mode (BLOCKER #2)
7. ✅ Logging (MEDIUM #7)

**Nice to have**:
8. External prompts (MEDIUM #8)
9. Better error handling (MEDIUM #9)
10. Token validation (MEDIUM #10)

---

## Testing Checklist

Before running production experiments:

- [ ] Gemini API key set and tested
- [ ] Goodfire API key set and tested
- [ ] HuggingFace token validated
- [ ] Dataset repo exists and is accessible
- [ ] Model parameters are deterministic (seed set)
- [ ] Batch mode works without human input
- [ ] Local backups are created
- [ ] Data validation catches malformed data
- [ ] Logs are being written
- [ ] Can successfully upload to HF
- [ ] Version info is captured in data
- [ ] Rate limits are handled (delays between runs)

---

## Example Fixed Code Snippet

Here's what the improved initialization should look like:

```python
import logging
import sys
import subprocess

# Configure logging
logging.basicConfig(...)
logger = logging.getLogger(__name__)

# Validate environment
def validate_environment():
    """Ensure all required resources are available."""
    errors = []

    # Check API keys
    if STEWARD_MODEL_NAME == "GEMINI_FLASH" and not os.environ.get("GEMINI_API_KEY"):
        errors.append("GEMINI_API_KEY required for GEMINI_FLASH")

    if STEWARD_MODEL_NAME == "LLAMA_3_1_8B" and not os.environ.get("GOODFIRE_API_KEY"):
        errors.append("GOODFIRE_API_KEY required for LLAMA_3_1_8B")

    if not os.environ.get("HF_TOKEN"):
        errors.append("HF_TOKEN required for data upload")

    # Check HF dataset access
    try:
        api = HfApi()
        api.whoami(token=os.environ.get("HF_TOKEN"))
        # Try to access dataset
        api.dataset_info(HF_REPO_ID, token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        errors.append(f"Cannot access HF dataset {HF_REPO_ID}: {e}")

    if errors:
        for error in errors:
            logger.error(error)
        raise RuntimeError("Environment validation failed")

    logger.info("Environment validation passed")

# Call at startup
if __name__ == "__main__":
    validate_environment()
    # ... rest of code
```

This ensures failures happen fast and clearly, not after hours of data collection.
