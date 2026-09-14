# Hugging Face Hub Integration

ViForge provides native, enterprise-grade publishing of specialized model weights, merged checkpoints, LoRA adapters, and auto-generated Model Cards directly to the [Hugging Face Hub](https://huggingface.co/).

---

## Architecture & Features

The Hugging Face integration (`viforge.artifacts.hub`) enables end-to-end artifact distribution with:

- **1-Click Publishing**: Push full weights or PEFT adapters in seconds.
- **Model Card Automation**: Automatically packages generated `README.md` containing evaluation metrics, statistical Wilson confidence intervals, and Pareto deployment recommendations into the repository root.
- **Public & Private Repositories**: Control visibility with `--private` / private repo toggle.
- **Secure Attestation**: Supports token-based authentication via CLI arguments, environment variable (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`), or cached Hugging Face CLI login credentials.
- **Resilient Folder Upload**: Fast chunked file uploads via `huggingface_hub.HfApi`.

---

## CLI Usage: `push-to-hub`

ViForge provides a dedicated CLI command `push-to-hub` to publish any local artifact directory:

```bash
# Push specialist weights to Hugging Face Hub
viforge push-to-hub runs/software_engineering/specialist_model viforge-org/qwen2.5-coder-specialist

# Push as a private repository with custom commit message and auto-generated model card
viforge push-to-hub runs/software_engineering/specialist_model my-org/qwen2.5-specialist \
    --token "$HF_TOKEN" \
    --private \
    --commit-message "Release specialized SWE model v1.0.0" \
    --model-card README.md
```

### Options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_dir` | Positional Argument | *Required* | Path to local directory containing model weights or LoRA adapter. |
| `repo_id` | Positional Argument | *Required* | Target repository ID on HF Hub (e.g., `username/model-name`). |
| `--token`, `-t` | Option | `None` (`HF_TOKEN` env) | Hugging Face user access token with write permission. |
| `--private` | Flag | `False` | Create or update as a private repository. |
| `--commit-message`, `-m` | Option | `"Upload model via ViForge"` | Git commit message. |
| `--model-card`, `-c` | Option | `None` | Optional path to `README.md` model card to include in the upload. |

---

## Interactive Studio (Streamlit Dashboard)

In the **ViForge Interactive Studio** (`viforge ui`), Tab 5 (**ViPym & Downstream Hub**) provides a 1-Click UI:

1. Launch the studio:
   ```bash
   viforge ui
   ```
2. Navigate to **Tab 5 ("🗜️ ViPym Compression Hub")** $\rightarrow$ **🤗 Hugging Face Hub 1-Click Publisher**.
3. Input target Repository ID (e.g. `viforge-org/qwen2.5-coder-specialist`).
4. Select visibility (Public / Private) and attach your Model Card.
5. Click **"🚀 Publish to Hugging Face Hub"** to upload and receive the direct Hub URL.

---

## Programmatic Python API

You can also trigger Hub publication directly inside automated training and deployment pipelines:

```python
from pathlib import Path
from viforge.artifacts.hub import HuggingFaceHubPublisher

repo_url = HuggingFaceHubPublisher.upload_model(
    model_dir=Path("runs/software_engineering/specialist_model"),
    repo_id="viforge-org/qwen2.5-coder-specialist",
    private=False,
    commit_message="Forged via ViForge automated pipeline",
    model_card_path=Path("README.md"),
)

print(f"Model published at: {repo_url}")
```
