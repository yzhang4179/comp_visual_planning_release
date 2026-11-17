import os

from huggingface_hub import hf_hub_download


def resolve_file(path: str, repo_id: str, local_dir: str = "ckpts") -> str:
    """
    Return `path` if it exists, otherwise download it from the Hugging Face repo `repo_id`,
    where it is stored under its path relative to `local_dir`.
    """
    if os.path.exists(path):
        return path
    filename = os.path.relpath(path, local_dir)
    if filename.startswith(".."):
        raise FileNotFoundError(f"'{path}' does not exist and is not under '{local_dir}', so it cannot be downloaded")
    print(f"'{path}' not found, downloading '{filename}' from https://huggingface.co/{repo_id}")
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
