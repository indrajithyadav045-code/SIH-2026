"""
VoiceGuard: Push to Hugging Face Hub (indrajit4533/voiceguard)
Problem Statement SIH26104: AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

Uploads:
- voiceguard_wav2vec2.py (Core Model Architecture & Forensics)
- dataset.py (Dual-Track Indic/English Data Pipeline)
- train.py (Multi-Task AM-Softmax Training Pipeline)
- inference.py (Production Threat Inference & Kill-Switch)
- README.md (Comprehensive Documentation & Model Card)
- Optional checkpoints in D:/voiceguard/checkpoints
"""

import os
import sys
import argparse
import logging
from huggingface_hub import HfApi, login

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("VoiceGuard.HF")

REPO_ID = "indrajit4533/voiceguard"
FILES_TO_UPLOAD = [
    "voiceguard_wav2vec2.py",
    "dataset.py",
    "train.py",
    "inference.py",
    "README.md"
]

def push_to_hub(token: str = None, commit_message: str = "Add production Wav2Vec2 training & inference pipeline for SIH26104"):
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("\n" + "=" * 70)
        print("ERROR: Hugging Face authentication token is required.")
        print("Please provide it via one of the following methods:")
        print("1. Pass as an argument:  python push_to_hf.py --token <YOUR_HF_WRITE_TOKEN>")
        print("2. Set environment var:  $env:HF_TOKEN = '<YOUR_HF_WRITE_TOKEN>'")
        print("3. Login via CLI:        huggingface-cli login")
        print("Generate a WRITE token at: https://huggingface.co/settings/tokens")
        print("=" * 70 + "\n")
        return False

    logger.info(f"Authenticating with Hugging Face Hub...")
    login(token=token, add_to_git_credential=True)
    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        logger.info(f"Authenticated as: {user_info.get('name', user_info.get('fullname'))}")
    except Exception as e:
        logger.error(f"Authentication verification failed: {e}")
        return False

    base_dir = "D:/voiceguard"
    logger.info(f"Target repository: https://huggingface.co/{REPO_ID}")

    for filename in FILES_TO_UPLOAD:
        local_path = os.path.join(base_dir, filename)
        if os.path.exists(local_path):
            logger.info(f"Uploading {filename} -> {REPO_ID}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"{commit_message} ({filename})"
            )
            logger.info(f"Uploaded {filename} successfully.")
        else:
            logger.warning(f"File not found: {local_path}, skipping.")

    # Check for best checkpoint if present
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_voiceguard_wav2vec2.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"Found trained checkpoint {ckpt_path}. Uploading...")
        api.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo="checkpoints/best_voiceguard_wav2vec2.pt",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add best model checkpoint weights"
        )
        logger.info("Uploaded checkpoint weights successfully.")

    print("\n" + "=" * 70)
    print("SUCCESS: VoiceGuard files successfully pushed to Hugging Face!")
    print(f"Repository URL: https://huggingface.co/{REPO_ID}")
    print("=" * 70 + "\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push VoiceGuard pipeline to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token (hf_...)")
    parser.add_argument("--message", type=str, default="Add production Wav2Vec2 training & inference pipeline for SIH26104", help="Commit message")
    args = parser.parse_args()

    success = push_to_hub(token=args.token, commit_message=args.message)
    sys.exit(0 if success else 1)