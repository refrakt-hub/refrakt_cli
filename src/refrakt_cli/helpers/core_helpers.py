# Core helper functions to avoid circular imports

from typing import List, Dict, Any, Optional, Tuple
from vertexai.generative_models import GenerativeModel, Part
import os
import json
from refrakt_cli.helpers.shared_core import extract_comprehensive_metadata

def call_gemini_with_retry(model: GenerativeModel, content_parts: Any, max_retries: int = 3, logger=None) -> str:
    """
    Call Gemini with exponential backoff retry logic for rate limiting.
    """
    for attempt in range(max_retries + 1):
        try:
            if logger:
                logger.info(f"Calling Gemini API (attempt {attempt + 1}/{max_retries + 1})")
            response = model.generate_content(content_parts)
            if logger:
                logger.info(f"Successfully received response from Gemini API")
            return response.text
        except Exception as e:
            if logger:
                logger.error(f"Error calling Gemini: {e}")
            if attempt == max_retries:
                return f"[ERROR] Failed to generate explanation: {e}"
    return "[ERROR] Unexpected error in retry logic"


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
        # Build comprehensive content for the method
        content_parts = []
        
        # Add system prompt
        content_parts.append(Part.from_text(system_prompt))
        
        # Add method-specific context
        method_context = f"""
## XAI Method Analysis: {method_name.upper()}

### Method Information
- **Method Name**: {method_name}
- **Files Available**: {len(method_files.get('files', []))} files

### Experiment Metadata
{json.dumps(metadata, indent=2)}

### Configuration Files
The following configuration files were used in this experiment:
{chr(10).join([f"- {config_file}" for config_file in config_files])}

### Available Files for {method_name}
The following files are available for analysis:
{chr(10).join([f"- {file_path}" for file_path in method_files.get('files', [])])}

Please provide a comprehensive analysis of the {method_name} XAI method based on the available data and metadata. Focus on:
1. What this method reveals about the model's behavior
2. Key patterns and insights from the available files
3. How this method contributes to understanding the model's decision-making process
4. Any limitations or considerations specific to this method
"""
        
        content_parts.append(Part.from_text(method_context))
        
        # Create the model and generate explanation
        model = GenerativeModel(model_name)
        explanation = call_gemini_with_retry(model, content_parts, logger=logger)
        
        if logger:
            logger.info(f"Generated explanation for {method_name}")
        return explanation
    except Exception as e:
        if logger:
            logger.error(f"Error generating explanation for {method_name}: {e}")
        return f"[ERROR] {e}"

def combine_method_explanations(method_explanations: Dict[str, str], metadata: Dict[str, Any], logger=None) -> str:
    """
    Combine individual method explanations into a comprehensive explanation.
    """
    if not method_explanations:
        return "No XAI explanations were generated."
    
    # Extract better metadata information
    model_info = metadata.get('model_info', {})
    dataset_info = metadata.get('dataset_info', {})
    experiment_info = metadata.get('experiment_info', {})
    performance_metrics = metadata.get('performance_metrics', {})
    
    combined_parts = [
        "# Comprehensive XAI Analysis",
        "This report provides explanations for all XAI methods used in this experiment.\n",
        "## Experiment Summary",
        f"- **Experiment ID**: {experiment_info.get('experiment_id', 'unknown')}",
        f"- **Model**: {model_info.get('name', 'unknown')} ({model_info.get('type', 'unknown')})",
        f"- **Dataset**: {dataset_info.get('name', 'unknown')}",
        f"- **Training Phase**: {'Yes' if experiment_info.get('has_train', False) else 'No'}",
        f"- **Inference Phase**: {'Yes' if experiment_info.get('has_inference', False) else 'No'}",
        f"- **XAI Methods**: {', '.join(method_explanations.keys())}",
        ""
    ]
    
    # Add performance metrics if available
    if performance_metrics:
        combined_parts.append("## Performance Metrics")
        for metric, value in performance_metrics.items():
            if value not in [None, '', 'N/A']:
                combined_parts.append(f"- **{metric}**: {value}")
        combined_parts.append("")
    
    # Add individual method explanations
    for method_name, explanation in method_explanations.items():
        combined_parts.append(f"## {method_name.upper()} Analysis")
        combined_parts.append(explanation)
        combined_parts.append("")
    
    combined_parts.append("## Summary")
    combined_parts.append("This analysis provides insights from multiple XAI methods, each offering a different perspective on the model's decision-making process.")
    combined_parts.append("Individual explanations for each method have been saved as separate files for detailed review.")
    
    return "\n".join(combined_parts)

def organize_xai_files_by_method(file_paths: List[str], root_dirs: Optional[List[str]] = None, logger=None) -> Dict[str, Dict[str, List[str]]]:
    """
    Organize files by method name based on directory structure.

    Args:
        file_paths: List of file paths (e.g., NPY or PNG files)
        root_dirs: List of root directories to search for methods (defaults to ['train', 'inference'])
        logger: Optional logger

    Returns:
        Dictionary mapping method names to their associated files
    """
    if root_dirs is None:
        root_dirs = ['train', 'inference']
    
    organized_files = {}
    for file_path in file_paths:
        path_parts = file_path.split(os.sep)
        for root_dir in root_dirs:
            if root_dir in path_parts:
                try:
                    idx = path_parts.index(root_dir)
                    if idx + 1 < len(path_parts):
                        method_name = path_parts[idx + 1]
                        if method_name not in organized_files:
                            organized_files[method_name] = {"files": []}
                        organized_files[method_name]["files"].append(file_path)
                except (ValueError, IndexError):
                    if logger:
                        logger.warning(f"Error processing file path: {file_path}")
    return organized_files

def extract_comprehensive_metadata(config_files: List[str], base_dir: str, checkpoints_dir: str, logger=None, training_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from config files, logs, and training results.
    """
    from refrakt_cli.helpers.metadata_helpers import (
        normalize_experiment_info,
        robust_merge_performance_metrics,
        collect_explanations_run_metadata,
        extract_experiment_metadata_helper,
        determine_train_inference,
        initialize_metadata_structure,
        load_config_files,
        extract_metadata_from_config
    )
    if logger:
        logger.info(f"extract_comprehensive_metadata called with training_results: {training_results}")

    # Extract experiment metadata
    experiment_metadata = extract_experiment_metadata_helper(checkpoints_dir, logger)

    # Determine has_train and has_inference
    has_train, has_inference = determine_train_inference(config_files)

    # Initialize metadata structure
    metadata = initialize_metadata_structure(experiment_metadata, has_train, has_inference)

    # Normalize and attach experiment_info
    metadata['experiment_info'] = normalize_experiment_info(experiment_metadata, has_train, has_inference)
    experiment_id = metadata['experiment_info']['experiment_id']

    # Load and parse configuration files
    config_data = load_config_files(config_files, logger)

    # Robust performance_metrics merging
    metadata["performance_metrics"] = robust_merge_performance_metrics(base_dir, training_results, logger)

    # Extract additional metadata from config data
    if config_data:
        config_metadata = extract_metadata_from_config(config_data)
        metadata.update(config_metadata)
        if logger:
            logger.info(f"Extracted metadata from config files: {config_metadata}")

    # Add run metadata
    metadata['run_metadata'] = collect_explanations_run_metadata(checkpoints_dir, experiment_id)

    if logger:
        logger.debug(f"[DEBUG] run_metadata.npy_files: {metadata['run_metadata']['npy_files']}")
        logger.debug(f"[DEBUG] run_metadata.png_files: {metadata['run_metadata']['png_files']}")

    if 'config_files' in metadata['run_metadata']:
        del metadata['run_metadata']['config_files']

    if logger:
        logger.info(f"[DEBUG] About to write summary: experiment_id={metadata['experiment_info']['experiment_id']}, metadata={json.dumps(metadata, indent=2)}")

    return metadata
