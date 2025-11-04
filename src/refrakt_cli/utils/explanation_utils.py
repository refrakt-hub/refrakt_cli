import logging
from typing import Any, Dict, List, Optional


def format_header(metadata: Dict[str, Any], methods: List[str]) -> List[str]:
    model_info = metadata.get("model_info", {})
    model_name = model_info.get("name", "unknown")
    model_type = model_info.get("type", "unknown")
    return [
        "# Comprehensive XAI Analysis",
        "This report provides explanations for all XAI methods used in "
        "this experiment.\n",
        "## Experiment Summary",
        f"- **Model**: {model_name} ({model_type})",
        f"- **Dataset**: {metadata.get('dataset_info', {}).get('name', 'unknown')}",
        f"- **XAI Methods**: {', '.join(methods)}",
        "",
    ]


def format_method_section(method_name: str, explanation: str) -> List[str]:
    return [f"## {method_name.upper()} Analysis", explanation, ""]


def format_summary() -> List[str]:
    return [
        "## Summary",
        "This analysis provides insights from multiple XAI methods, each "
        "offering a different perspective on the model's decision-making "
        "process.",
        "Individual explanations for each method have been saved as separate "
        "files for detailed review.",
    ]


def combine_method_explanations(
    method_explanations: Dict[str, str],
    metadata: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> str:
    if not method_explanations:
        return "No XAI explanations were generated."
    parts = format_header(metadata, list(method_explanations.keys()))
    for name, text in method_explanations.items():
        parts.extend(format_method_section(name, text))
    parts.extend(format_summary())
    return "\n".join(parts)


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
