"""
Core LLM and XAI utilities for Refrakt CLI.

This module contains the authoritative implementations of core functions
for LLM operations and XAI processing that were previously duplicated
across multiple helper files.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from vertexai.generative_models import GenerativeModel, Part


def call_gemini_with_retry(
    model: GenerativeModel,
    content_parts: Any,
    max_retries: int = 3,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Call Gemini with exponential backoff retry logic for rate limiting.

    Args:
        model: Gemini GenerativeModel instance
        content_parts: Content parts to send to the model
        max_retries: Maximum number of retry attempts
        logger: Optional logger instance

    Returns:
        Generated text response or error message
    """
    for attempt in range(max_retries + 1):
        try:
            if logger:
                logger.info(
                    f"Calling Gemini API (attempt {attempt + 1}/{max_retries + 1})"
                )
            response = model.generate_content(content_parts)
            if logger:
                logger.info("Successfully received response from Gemini API")
            return response.text
        except Exception as e:
            if logger:
                logger.error(f"Error calling Gemini: {e}")
            if attempt == max_retries:
                return f"[ERROR] Failed to generate explanation: {e}"
    return "[ERROR] Unexpected error in retry logic"


def organize_xai_files(
    file_paths: List[str],
    root_dirs: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Organize files by XAI method name based on directory structure.

    Args:
        file_paths: List of file paths (e.g., NPY or PNG files)
        root_dirs: List of root directories to search for methods
        logger: Optional logger instance

    Returns:
        Dictionary mapping method names to their associated files
    """
    if root_dirs is None:
        root_dirs = ["train", "inference"]

    organized_files: Dict[str, Dict[str, List[str]]] = {}
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
                        break  # Only process once per file
                except (ValueError, IndexError):
                    if logger:
                        logger.warning(f"Error processing file path: {file_path}")
    return organized_files


def _analyze_npy_file(file_path: str) -> str:
    """
    Analyze a NumPy array file and return formatted analysis string.

    Args:
        file_path: Path to the .npy file

    Returns:
        Formatted string with file analysis
    """
    import numpy as np

    data = np.load(file_path)
    non_zero_count = np.count_nonzero(data)
    non_zero_pct = 100 * non_zero_count / data.size if data.size > 0 else 0

    return f"""
### {os.path.basename(file_path)}
- **File Type**: NumPy Array (.npy)
- **Shape**: {data.shape}
- **Data Type**: {data.dtype}
- **Min Value**: {np.min(data):.6f}
- **Max Value**: {np.max(data):.6f}
- **Mean**: {np.mean(data):.6f}
- **Standard Deviation**: {np.std(data):.6f}
- **File Path**: {file_path}

**Data Sample (first few values):**
{str(data.flatten()[:10]) if data.size > 0 else 'Empty array'}

**Data Statistics:**
- Non-zero elements: {non_zero_count} / {data.size} ({non_zero_pct:.2f}%)
- Unique values: {len(np.unique(data))}
"""


def _analyze_png_file(file_path: str) -> str:
    """
    Analyze a PNG image file and return formatted analysis string.

    Args:
        file_path: Path to the .png file

    Returns:
        Formatted string with file analysis
    """
    try:
        from PIL import Image

        image = Image.open(file_path)
        return f"""
### {os.path.basename(file_path)}
- **File Type**: PNG Image
- **Size**: {image.size}
- **Mode**: {image.mode}
- **File Path**: {file_path}
- **Note**: Image content analysis available but not included in this API call
"""
    except Exception as img_error:
        return f"""
### {os.path.basename(file_path)}
- **File Type**: PNG Image (Error loading)
- **Error**: {str(img_error)}
- **File Path**: {file_path}
"""


def _process_file(
    file_path: str, logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Process a single file and return analysis string.

    Args:
        file_path: Path to the file to process
        logger: Optional logger instance

    Returns:
        Formatted analysis string or None if processing fails
    """
    try:
        if file_path.endswith(".npy"):
            return _analyze_npy_file(file_path)
        if file_path.endswith(".png"):
            return _analyze_png_file(file_path)
        return None
    except Exception as file_error:
        return f"""
### {os.path.basename(file_path)}
- **Error loading file**: {str(file_error)}
- **File Path**: {file_path}
"""


def _build_file_analysis_content(
    method_file_list: List[str], logger: Optional[logging.Logger] = None
) -> List[str]:
    """
    Build file analysis content for all files in the method.

    Args:
        method_file_list: List of file paths to analyze
        logger: Optional logger instance

    Returns:
        List of formatted file analysis strings
    """
    file_analysis = []
    for file_path in method_file_list:
        analysis = _process_file(file_path, logger)
        if analysis:
            file_analysis.append(analysis)
    return file_analysis


def _build_method_context(
    method_name: str,
    metadata: Dict[str, Any],
    config_files: List[str],
    file_analysis: List[str],
    method_file_list: List[str],
) -> str:
    """
    Build the method context string for LLM prompt.

    Args:
        method_name: Name of the XAI method
        metadata: Experiment metadata
        config_files: List of configuration files
        file_analysis: List of file analysis strings
        method_file_list: List of file paths

    Returns:
        Formatted method context string
    """
    return f"""
## XAI Method Analysis: {method_name.upper()}

### Method Information
- **Method Name**: {method_name}
- **Files Available**: {len(method_file_list)} files

### Experiment Metadata
```json
{json.dumps(metadata, indent=2)}
```

### Configuration Files
The following configuration files were used in this experiment:
{chr(10).join([f"- {config_file}" for config_file in config_files])}

### Detailed File Analysis for {method_name}
{chr(10).join(file_analysis)}

## Analysis Request

Please provide a comprehensive analysis of the **{method_name}** XAI method
based on the available data above. Your analysis should include:

1. **Method Overview**: What does the {method_name} method reveal about the
   model's behavior?

2. **Data Analysis**: What patterns do you observe in the numerical data
   (statistics, distributions, activation patterns)?

3. **Key Insights**: What unique insights does this {method_name} method
   provide about the model's decision-making process?

4. **Pattern Recognition**: What patterns are evident in the data that
   indicate how the model processes inputs?

5. **Quality Assessment**: How does the data quality and coverage look for
   this method?

6. **Limitations**: What limitations should be considered when interpreting
   these {method_name} results?

7. **Actionable Recommendations**: Based on this {method_name} analysis, what
   specific recommendations can you provide?

Please structure your response as a detailed markdown report focused
specifically on the {method_name} method.
"""


def generate_method_explanation(
    method_name: str,
    method_files: Dict[str, List[str]],
    metadata: Dict[str, Any],
    config_files: List[str],
    system_prompt: str,
    model_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Generate explanation for a specific XAI method with actual file content.

    Args:
        method_name: Name of the XAI method
        method_files: Dictionary containing files for the method
        metadata: Experiment metadata
        config_files: List of configuration files
        system_prompt: System prompt for the LLM
        model_name: Name of the model to use
        logger: Optional logger instance

    Returns:
        Generated explanation text
    """
    try:
        method_file_list = method_files.get("files", [])

        # Build comprehensive content for the method
        content_parts = [Part.from_text(system_prompt)]

        # Process files and build analysis
        file_analysis = _build_file_analysis_content(method_file_list, logger)

        # Build method context
        method_context = _build_method_context(
            method_name, metadata, config_files, file_analysis, method_file_list
        )
        content_parts.append(Part.from_text(method_context))

        if logger:
            logger.info(
                f"Generating explanation for {method_name} with "
                f"{len(method_file_list)} files"
            )

        # Create the model and generate explanation
        model = GenerativeModel(model_name)
        explanation = call_gemini_with_retry(model, content_parts, logger=logger)

        if logger:
            logger.info(f"Generated explanation for {method_name}")

        return explanation

    except Exception as e:
        error_msg = f"[ERROR] Failed to generate explanation for {method_name}: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg
