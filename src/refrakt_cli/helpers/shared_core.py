# Shared core functions to resolve circular imports

from typing import List, Dict, Any, Optional
from vertexai.generative_models import GenerativeModel
import os
import json
from refrakt_cli.helpers.metadata_helpers import (
    extract_experiment_metadata_helper,
    determine_train_inference,
    initialize_metadata_structure,
    load_config_files,
    extract_performance_metrics,
    extract_metadata_from_config
)
from refrakt_cli.utils.metadata_utils import create_experiment_info, merge_performance_metrics, collect_run_metadata
from refrakt_cli.utils.explanation_utils import combine_method_explanations

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
        if logger:
            logger.error(f"Error generating explanation for {method_name}: {e}")
        return f"[ERROR] {e}"

def organize_xai_files_by_method(file_paths: List[str], root_dirs: List[str], logger=None) -> Dict[str, Dict[str, List[str]]]:
    """
    Organize files by method name based on directory structure.
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

def extract_comprehensive_metadata(config_files: List[str], base_dir: str, checkpoints_dir: str, logger=None, training_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from config files, logs, and training results.
    Implements smart merging to preserve good metrics from previous phases.
    """
    if logger:
        logger.info(f"extract_comprehensive_metadata called with training_results: {training_results}")

    # Check for existing summary_metrics.json to preserve good values
    summary_metrics_path = os.path.join(checkpoints_dir, "explanations", "summary_metrics.json")
    existing_metadata = None
    if os.path.exists(summary_metrics_path):
        try:
            with open(summary_metrics_path, 'r') as f:
                existing_metadata = json.load(f)
                if logger:
                    logger.info(f"[DEBUG] Found existing metadata, preserving good values")
        except Exception as e:
            if logger:
                logger.warning(f"Could not read existing summary_metrics.json: {e}")

    # Extract experiment metadata
    experiment_metadata = extract_experiment_metadata_helper(checkpoints_dir, logger)

    # Determine has_train and has_inference
    has_train, has_inference = determine_train_inference(config_files)

    # Initialize metadata structure
    metadata = initialize_metadata_structure(experiment_metadata, has_train, has_inference)

    # normalize and attach experiment_info
    metadata['experiment_info'] = create_experiment_info(experiment_metadata, has_train, has_inference)
    experiment_id = metadata['experiment_info']['experiment_id']

    # Load and parse configuration files
    config_data = load_config_files(config_files, logger)

    # Smart merge performance metrics from logs + training_results + existing data
    new_performance_metrics = merge_performance_metrics(base_dir, training_results, logger)
    
    # If we have existing metadata, merge performance metrics intelligently
    if existing_metadata and existing_metadata.get('performance_metrics'):
        existing_perf = existing_metadata['performance_metrics']
        merged_perf = {}
        
        # Preserve good values from existing metadata
        for key, value in existing_perf.items():
            if value not in ["N/A", "", None]:
                merged_perf[key] = value
        
        # Add new values only if they're better (non-N/A) or if key doesn't exist
        for key, value in new_performance_metrics.items():
            if value not in ["N/A", "", None]:
                merged_perf[key] = value
            elif key not in merged_perf:
                merged_perf[key] = value
                
        metadata["performance_metrics"] = merged_perf
        if logger:
            logger.info(f"[DEBUG] Merged performance metrics: {merged_perf}")
    else:
        metadata["performance_metrics"] = new_performance_metrics

    # Extract additional metadata from config data
    if config_data:
        config_metadata = extract_metadata_from_config(config_data)
        # Merge config metadata, preserving existing good values
        if existing_metadata:
            for key in ['model_info', 'training_info', 'dataset_info', 'xai_info', 'viz_info']:
                if key in config_metadata:
                    if not existing_metadata.get(key) or existing_metadata[key] == {}:
                        metadata[key] = config_metadata[key]
                    else:
                        metadata[key] = existing_metadata[key]  # Keep existing if it has data
                elif existing_metadata.get(key):
                    metadata[key] = existing_metadata[key]  # Keep existing
        else:
            metadata.update(config_metadata)

    # collect run_metadata files under explanations/
    metadata['run_metadata'] = collect_run_metadata(checkpoints_dir, experiment_id, logger)
    if logger:
        logger.debug(f"[DEBUG] run_metadata.npy_files: {metadata['run_metadata']['npy_files']}")
        logger.debug(f"[DEBUG] run_metadata.png_files: {metadata['run_metadata']['png_files']}")

    # Collect runtime XAI information (method parameters, layer info, etc.)
    runtime_xai_info = collect_runtime_xai_info(checkpoints_dir, logger)
    if runtime_xai_info and 'xai_info' in metadata:
        # Merge runtime info with config-based XAI info
        if 'runtime_info' not in metadata['xai_info']:
            metadata['xai_info']['runtime_info'] = runtime_xai_info
        else:
            metadata['xai_info']['runtime_info'].update(runtime_xai_info)
    elif runtime_xai_info:
        # Create xai_info if it doesn't exist
        if 'xai_info' not in metadata:
            metadata['xai_info'] = {}
        metadata['xai_info']['runtime_info'] = runtime_xai_info

    if 'config_files' in metadata['run_metadata']:
        del metadata['run_metadata']['config_files']

    if logger:
        logger.info(f"[DEBUG] About to write summary: experiment_id={metadata['experiment_info']['experiment_id']}, metadata={json.dumps(metadata, indent=2)}")

    return metadata

def collect_runtime_xai_info(checkpoints_dir: str, logger=None) -> Dict[str, Any]:
    """
    Collect runtime XAI method information including layer details.
    This function looks for XAI instances that may have been saved or can be introspected.
    """
    runtime_xai_info = {}
    
    try:
        # Look for XAI hook configurations in config files or runtime data
        # This is a placeholder for future enhancement where we can introspect
        # actual XAI method instances and extract their runtime parameters
        
        # For now, we'll return an empty dict but structure is ready for enhancement
        # When XAI methods are executed, they could save their configuration info
        # to a runtime_xai_info.json file that we could read here
        
        runtime_info_path = os.path.join(checkpoints_dir, "explanations", "runtime_xai_info.json")
        if os.path.exists(runtime_info_path):
            with open(runtime_info_path, 'r') as f:
                runtime_xai_info = json.load(f)
                if logger:
                    logger.info(f"Loaded runtime XAI info from {runtime_info_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Error collecting runtime XAI info: {e}")
    
    return runtime_xai_info

def save_runtime_xai_info(xai_instance, method_name: str, params: Dict[str, Any], base_dir: str, logger=None):
    """
    Save runtime XAI method information to a JSON file.
    This captures method parameters, layer information, and other runtime details.
    """
    try:
        runtime_info = {
            'method': method_name,
            'config_params': params,
        }
        
        # Collect method-specific runtime information
        if hasattr(xai_instance, 'get_target_layer_info'):
            try:
                layer_info = xai_instance.get_target_layer_info()
                if layer_info:
                    runtime_info['layer_info'] = layer_info
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to get layer info for {method_name}: {e}")
        
        # Add other method-specific information
        if hasattr(xai_instance, 'layer') and xai_instance.layer is not None:
            runtime_info['resolved_layer_type'] = type(xai_instance.layer).__name__
            
        # Save to runtime_xai_info.json in explanations directory
        explanations_dir = os.path.join(base_dir, "explanations")
        os.makedirs(explanations_dir, exist_ok=True)
        
        runtime_file = os.path.join(explanations_dir, "runtime_xai_info.json")
        
        # Load existing data if file exists
        existing_info = {}
        if os.path.exists(runtime_file):
            try:
                with open(runtime_file, 'r') as f:
                    existing_info = json.load(f)
            except:
                existing_info = {}
        
        # Add new method info
        if 'methods' not in existing_info:
            existing_info['methods'] = {}
        existing_info['methods'][method_name] = runtime_info
        
        # Save updated info
        with open(runtime_file, 'w') as f:
            json.dump(existing_info, f, indent=2)
            
        if logger:
            logger.debug(f"Saved runtime XAI info for {method_name} to {runtime_file}")
            
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save runtime XAI info for {method_name}: {e}")
