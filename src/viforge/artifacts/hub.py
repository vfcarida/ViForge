"""
Direct Hugging Face Hub Integration for ViForge Models, LoRA Adapters, and Model Cards.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Union

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # pragma: no cover
    HfApi = None  # type: ignore[misc, assignment]
    HfHubHTTPError = Exception  # type: ignore[misc, assignment]


def validate_repo_id(repo_id: str) -> bool:
    """
    Validate that repo_id conforms to Hugging Face Hub format (namespace/repo_name or repo_name).
    """
    if not repo_id or not isinstance(repo_id, str):
        return False
    # Allowed: alphanumeric, hyphens, underscores, dots, and at most one slash
    pattern = r"^[a-zA-Z0-9_\.\-]+(/[a-zA-Z0-9_\.\-]+)?$"
    return bool(re.match(pattern, repo_id))


class HuggingFaceHubPublisher:
    """
    Publisher to upload fine-tuned models, LoRA adapters, and Model Cards directly to Hugging Face Hub.
    """

    @classmethod
    def upload_model(
        cls,
        model_dir: Union[str, Path],
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload model via ViForge",
        model_card_path: Optional[Union[str, Path]] = None,
        tags: Optional[List[str]] = None,
        exist_ok: bool = True,
    ) -> str:
        """
        Upload model artifacts and model card to Hugging Face Hub.

        Args:
            model_dir: Path to directory containing model weights or LoRA adapters.
            repo_id: Hugging Face Hub target repository ID (e.g. 'username/model-specialist').
            token: Optional Hugging Face API token. If omitted, reads from HF_TOKEN env var or CLI login.
            private: Whether to create the repository as private.
            commit_message: Git commit message for the upload.
            model_card_path: Optional path to a README.md model card to include in the upload.
            tags: Optional tags (for documentation / metadata).
            exist_ok: If True, do not fail if repository already exists.

        Returns:
            URL to the repository on Hugging Face Hub.
        """
        path = Path(model_dir)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Model directory not found or is not a directory: {model_dir}")

        if not validate_repo_id(repo_id):
            raise ValueError(
                f"Invalid Hugging Face repo_id format: '{repo_id}'. "
                f"Expected 'username/model-name' or 'model-name' with alphanumeric, hyphens, or underscores."
            )

        resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if HfApi is None:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is not installed. Install with 'pip install huggingface_hub'."
            )

        api = HfApi(token=resolved_token)

        # 1. Ensure repository exists or create it
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=exist_ok,
        )

        # 2. If an external model card path was provided, copy it into model_dir if needed or upload it
        if model_card_path is not None:
            mc_path = Path(model_card_path)
            if mc_path.exists() and mc_path.is_file():
                dest_readme = path / "README.md"
                if mc_path.resolve() != dest_readme.resolve():
                    shutil.copyfile(mc_path, dest_readme)

        # 3. Upload all model folder contents
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

        return f"https://huggingface.co/{repo_id}"
