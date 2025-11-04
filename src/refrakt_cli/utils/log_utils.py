from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from datetime import datetime


def sort_log_files_by_time(log_files: List[str]) -> List[str]:
    """
    Sort log files by their modification time in descending order.

    Args:
        log_files: List of log file paths

    Returns:
        Sorted list of log file paths
    """
    log_files_with_time = []
    for log_file in log_files:
        try:
            mtime = os.path.getmtime(log_file)
            log_files_with_time.append((log_file, mtime))
        except Exception:
            continue

    log_files_with_time.sort(key=lambda x: x[1], reverse=True)
    return [log_file for log_file, _ in log_files_with_time]


def _extract_accuracy(line: str) -> Optional[float]:
    """
    Extract accuracy value from a log line.

    Args:
        line: A single line from the log file

    Returns:
        Extracted accuracy value or None if not found
    """
    try:
        # Check for various accuracy patterns
        patterns = [
            "Model accuracy: ",
            "model accuracy: ",
            "Validation Accuracy: ",
            "validation accuracy: ",
            "accuracy: ",
            "Accuracy: ",
        ]

        for pattern in patterns:
            if pattern in line:
                # Extract the value after the pattern
                value_part = line.split(pattern)[-1].strip()
                # Handle percentage format (e.g., "97.30%")
                if "%" in value_part:
                    acc_val = float(value_part.split("%")[0]) / 100.0
                else:
                    # Handle decimal format (e.g., "0.9730")
                    acc_val = float(value_part.split()[0])

                if 0.0 <= acc_val <= 1.0:
                    return acc_val
    except Exception:
        pass
    return None


def _extract_loss(line: str) -> Optional[float]:
    """
    Extract loss value from a log line.

    Args:
        line: A single line from the log file

    Returns:
        Extracted loss value or None if not found
    """
    try:
        # Check for various loss patterns
        patterns = [
            "Avg Loss: ",
            "avg loss: ",
            "Average Loss: ",
            "average loss: ",
            "Loss: ",
            "loss: ",
            "Validation Loss: ",
            "validation loss: ",
            "final_loss: ",
            "Final Loss: ",
        ]

        for pattern in patterns:
            if pattern in line:
                value_part = line.split(pattern)[-1].strip()
                return float(value_part.split()[0])
    except Exception:
        pass
    return None


def _extract_epoch(line: str) -> Optional[int]:
    """
    Extract epoch number from a log line.

    Args:
        line: A single line from the log file

    Returns:
        Extracted epoch number or None if not found
    """
    try:
        # Check for various epoch patterns
        import re

        # Pattern 1: "Epoch [1/5]" or "Epoch 1/5"
        match = re.search(r"Epoch\s*\[?(\d+)/(\d+)", line, re.IGNORECASE)
        if match:
            _ = int(match.group(1))  # current_epoch (not used)
            total_epochs = int(match.group(2))
            return total_epochs  # Return total epochs completed

        # Pattern 2: "epoch: 5" or "epochs_completed: 5"
        patterns = [
            r"epochs_completed:\s*(\d+)",
            r"epoch:\s*(\d+)",
            r"Epoch\s+(\d+)\s+complete",
        ]

        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return int(match.group(1))

    except Exception:
        pass
    return None


def _extract_timestamp_from_line(line: str) -> Optional[datetime]:
    """
    Extract timestamp from a log line if present.

    Args:
        line: Log line to parse

    Returns:
        datetime object if timestamp found, None otherwise
    """
    import re
    from datetime import datetime

    timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    if timestamp_match:
        try:
            return datetime.strptime(
                timestamp_match.group(1).split(",")[0], "%Y-%m-%d %H:%M:%S"
            )
        except (ValueError, AttributeError):
            return None
    return None


def _is_training_start_line(line: str) -> bool:
    """
    Check if a log line indicates training start.

    Args:
        line: Log line to check

    Returns:
        True if line indicates training start, False otherwise
    """
    return "Training phase started" in line or "Training started" in line


def _is_training_end_line(line: str) -> bool:
    """
    Check if a log line indicates training completion.

    Args:
        line: Log line to check

    Returns:
        True if line indicates training completion, False otherwise
    """
    return "Training completed successfully" in line or "Training results:" in line


def _extract_training_time(lines: List[str]) -> Optional[float]:
    """
    Extract training time by calculating time difference between start and end.

    Args:
        lines: List of log lines

    Returns:
        Training time in seconds or None if not found
    """
    try:
        start_time = None
        end_time = None

        for line in lines:
            if _is_training_start_line(line):
                timestamp = _extract_timestamp_from_line(line)
                if timestamp:
                    start_time = timestamp
            elif _is_training_end_line(line):
                timestamp = _extract_timestamp_from_line(line)
                if timestamp:
                    end_time = timestamp

        if start_time and end_time:
            return (end_time - start_time).total_seconds()

    except Exception:
        pass
    return None


def extract_key_metrics_from_logs(
    log_files: List[str], logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract key performance metrics from log files.

    Args:
        log_files: List of log file paths
        logger: Optional logger

    Returns:
        Dictionary containing extracted metrics
    """
    metrics = {
        "best_accuracy": "N/A",
        "final_loss": "N/A",
        "epochs_completed": "N/A",
        "training_time": "N/A",
    }

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Extract accuracy from individual lines
            for line in lines:
                acc_val = _extract_accuracy(line)
                if acc_val is not None:
                    metrics["best_accuracy"] = str(acc_val)

                loss_val = _extract_loss(line)
                if loss_val is not None:
                    metrics["final_loss"] = str(loss_val)

                epoch_val = _extract_epoch(line)
                if epoch_val is not None:
                    metrics["epochs_completed"] = str(epoch_val)

            # Extract training time from all lines
            training_time = _extract_training_time(lines)
            if training_time is not None:
                metrics["training_time"] = str(round(training_time, 2))

        except Exception as e:
            if logger:
                logger.warning(f"Error reading log file {log_file}: {e}")

    return metrics


def filter_recent_logs(
    log_files: List[str],
    max_age_seconds: int = 86400,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Filter log files to include only those modified within a certain time frame.

    Args:
        log_files: List of log file paths
        max_age_seconds: Maximum age of log files in seconds (default: 24 hours)
        logger: Optional logger

    Returns:
        List of recent log file paths
    """
    current_time = time.time()
    recent_logs = []
    for log_file in log_files:
        try:
            log_time = os.path.getmtime(log_file)
            if current_time - log_time <= max_age_seconds:
                recent_logs.append(log_file)
        except Exception as e:
            if logger:
                logger.warning(f"Error accessing log file {log_file}: {e}")
    return recent_logs
