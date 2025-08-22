import os
from typing import List, Dict, Any
from vertexai.generative_models import GenerativeModel
from refrakt_cli.llm_explainer import call_gemini_with_retry

# Shared helper functions to avoid circular imports

def organize_xai_files_by_method(file_paths, root_dirs, logger=None):
    """
    Organize files by method name based on directory structure.

    Args:
        file_paths: List of file paths (e.g., NPY or PNG files)
        root_dirs: List of root directories to search for methods
        logger: Optional logger

    Returns:
        Dictionary mapping method names to their associated files
    """
    organized_files = {}
    for file_path in file_paths:
        path_parts = file_path.split(os.sep)
        for root_dir in root_dirs:
            if root_dir in path_parts:
                try:
                    idx = path_parts.index(root_dir)
                    if idx + 2 < len(path_parts):
                        method_name = path_parts[idx + 2]
                        if method_name not in organized_files:
                            organized_files[method_name] = {"files": []}
                        organized_files[method_name]["files"].append(file_path)
                except (ValueError, IndexError):
                    if logger:
                        logger.warning(f"Error processing file path: {file_path}")
    return organized_files

def build_method_content(
    method_name: str, 
    method_files: Dict[str, List[str]], 
    metadata: Dict[str, Any], 
    config_files: List[str], 
    system_prompt: str, 
    logger=None
) -> List[str]:
    """
    Build structured content for a specific XAI method using Gemini API format.
    """
    content_parts = []
    context = f"Method: {method_name}, Metadata: {metadata}, Config: {config_files}"
    content_parts.append(context)
    return content_parts

def generate_method_explanation(
    method_name: str, 
    method_files: Dict[str, List[str]], 
    metadata: Dict[str, Any], 
    config_files: List[str], 
    system_prompt: str, 
    model_name: str, 
    logger=None
) -> str:
    """
    Generate explanation for a specific XAI method.
    """
    try:
        content_parts = [
            f"Method: {method_name}",
            f"Metadata: {metadata}",
            f"Config: {config_files}"
        ]
        model = GenerativeModel(model_name)
        explanation = call_gemini_with_retry(model, content_parts, logger=logger)
        if logger:
            logger.info(f"Generated explanation for {method_name}")
        return explanation
    except Exception as e:
        err_msg = f"Error generating explanation for {method_name}: {e}"
        if logger:
            logger.error(err_msg)
        return f"[ERROR] {e}"

def combine_method_explanations(method_explanations: Dict[str, str], metadata: Dict[str, Any], logger=None) -> str:
    """
    Combine individual method explanations into a comprehensive explanation.
    """
    if not method_explanations:
        return "No XAI explanations were generated."
    combined_parts = [
        "# Comprehensive XAI Analysis",
        "This report provides explanations for all XAI methods used in this experiment.\n",
        "## Experiment Summary",
        f"- **Model**: {metadata.get('model_info', {}).get('name', 'unknown')} ({metadata.get('model_info', {}).get('type', 'unknown')})",
        f"- **Dataset**: {metadata.get('dataset_info', {}).get('name', 'unknown')}",
        f"- **XAI Methods**: {', '.join(method_explanations.keys())}",
        ""
    ]
    for method_name, explanation in method_explanations.items():
        combined_parts.append(f"## {method_name.upper()} Analysis")
        combined_parts.append(explanation)
        combined_parts.append("")
    combined_parts.append("## Summary")
    combined_parts.append("This analysis provides insights from multiple XAI methods, each offering a different perspective on the model's decision-making process.")
    combined_parts.append("Individual explanations for each method have been saved as separate files for detailed review.")
    combined_explanation = "\n".join(combined_parts)
    if logger:
        logger.info(f"Combined explanations for {len(method_explanations)} XAI methods")
    return combined_explanation
