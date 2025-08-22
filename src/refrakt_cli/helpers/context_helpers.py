from typing import Dict, List, Any

def extract_metadata_context(metadata: Dict[str, Any]) -> str:
    """
    Extract context from metadata.
    """
    return "\n".join([f"{key}: {value}" for key, value in metadata.items()])

def extract_file_context(npy_files: List[str], config_files: List[str]) -> str:
    """
    Extract context from file lists.
    """
    npy_context = f"Numpy files: {', '.join(npy_files)}"
    config_context = f"Config files: {', '.join(config_files)}"
    return f"{npy_context}\n{config_context}"
