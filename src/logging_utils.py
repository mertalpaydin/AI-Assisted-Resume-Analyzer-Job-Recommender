"""
Centralized Logging Utilities for AI-Assisted Resume Analyzer & Job Recommender

This module provides:
- Centralized logging configuration for all project modules
- Helper functions to log experiments, decisions, and findings to markdown files
- Easy-to-use wrappers for documentation capture
- Checkpoint functions for recording progress

Author: Mert Alp Aydin
Date: 2025-11-13
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        name: Name of the logger (typically __name__ from calling module)
        log_file: Optional log file path. If None, uses 'logs/debug.log'
        level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing resume...")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler
    if log_file is None:
        log_file = LOGS_DIR / "debug.log"
    else:
        log_file = Path(log_file)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    return logger


def log_experiment(
    experiment_number: int,
    name: str,
    phase: str,
    objective: str,
    hypothesis: str,
    options_tested: List[Dict[str, Any]],
    metrics_used: Dict[str, str],
    winner: str,
    rationale: str,
    key_learning: str,
    impact: str,
    next_steps: str = ""
) -> None:
    """
    Log an experiment to the experiment_log.md file.

    Args:
        experiment_number: Unique experiment number
        name: Name of the experiment
        phase: Phase number and name
        objective: What we're trying to learn/optimize
        hypothesis: What we expect to happen
        options_tested: List of dicts with 'name', 'configuration', 'results', 'observations'
        metrics_used: Dict of metric names and their definitions
        winner: Selected option
        rationale: Why this option was chosen
        key_learning: What we learned
        impact: How this affects the project
        next_steps: Follow-up actions

    Example:
        >>> log_experiment(
        ...     experiment_number=1,
        ...     name="LLM Model Comparison",
        ...     phase="Phase 2 - Resume Parsing",
        ...     objective="Compare local LLM models for resume parsing accuracy",
        ...     hypothesis="Larger models will have better extraction accuracy",
        ...     options_tested=[
        ...         {
        ...             'name': 'granite4:micro',
        ...             'configuration': 'Default settings',
        ...             'results': 'Accuracy: 75%, Latency: 0.5s',
        ...             'observations': 'Fast but misses some fields'
        ...         }
        ...     ],
        ...     metrics_used={'Accuracy': 'Percentage of correctly extracted fields'},
        ...     winner='llama3.2:3b',
        ...     rationale='Best balance of accuracy and speed',
        ...     key_learning='Smaller models need more prompt engineering',
        ...     impact='Will use llama3.2:3b for all resume parsing'
        ... )
    """
    exp_log_path = LOGS_DIR / "experiment_log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Build the experiment entry
    entry = f"\n---\n\n"
    entry += f"### Experiment #{experiment_number}: {name}\n"
    entry += f"**Date:** {timestamp}\n"
    entry += f"**Phase:** {phase}\n"
    entry += f"**Objective:** {objective}\n\n"
    entry += f"**Hypothesis:** {hypothesis}\n\n"

    entry += "**Options Tested:**\n"
    for i, option in enumerate(options_tested, 1):
        entry += f"{i}. **{option['name']}**\n"
        entry += f"   - Configuration: {option['configuration']}\n"
        entry += f"   - Results: {option['results']}\n"
        entry += f"   - Observations: {option['observations']}\n\n"

    entry += "**Metrics Used:**\n"
    for metric_name, metric_def in metrics_used.items():
        entry += f"- {metric_name}: {metric_def}\n"
    entry += "\n"

    entry += f"**Winner:** {winner}\n\n"
    entry += f"**Rationale:** {rationale}\n\n"
    entry += f"**Key Learning:** {key_learning}\n\n"
    entry += f"**Impact on Project:** {impact}\n"

    if next_steps:
        entry += f"\n\n**Next Steps:** {next_steps}\n"

    # Append to experiment log
    with open(exp_log_path, 'a', encoding='utf-8') as f:
        f.write(entry)

    logger = logging.getLogger(__name__)
    logger.info(f"Logged Experiment #{experiment_number}: {name}")


def log_decision(
    decision_number: int,
    title: str,
    phase: str,
    context: str,
    decision: str,
    alternatives: List[Dict[str, Any]],
    rationale: str,
    tradeoffs: List[str],
    consequences: Dict[str, str],
    status: str = "Accepted"
) -> None:
    """
    Log a technical decision to the decisions.md file.

    Args:
        decision_number: Unique decision number
        title: Title of the decision
        phase: Phase number
        context: Situation that led to this decision
        decision: What was decided
        alternatives: List of dicts with 'name', 'pros', 'cons'
        rationale: Why this option was chosen
        tradeoffs: List of trade-off descriptions
        consequences: Dict with 'positive', 'negative', 'risks'
        status: Decision status (default: "Accepted")

    Example:
        >>> log_decision(
        ...     decision_number=4,
        ...     title="Use PyPDF2 for PDF Parsing",
        ...     phase="Phase 2",
        ...     context="Need reliable PDF parsing for various resume formats",
        ...     decision="Use PyPDF2 as primary parser with fallback to pdfplumber",
        ...     alternatives=[
        ...         {
        ...             'name': 'PyPDF2',
        ...             'pros': 'Fast, reliable, simple API',
        ...             'cons': 'Struggles with complex layouts'
        ...         }
        ...     ],
        ...     rationale="Best balance of speed and reliability",
        ...     tradeoffs=["Speed vs accuracy for complex layouts"],
        ...     consequences={
        ...         'positive': 'Fast parsing for most resumes',
        ...         'negative': 'May need manual review for complex formats',
        ...         'risks': 'Some resumes may not parse correctly'
        ...     }
        ... )
    """
    decisions_path = LOGS_DIR / "decisions.md"
    timestamp = datetime.now().strftime("%Y-%m-%d")

    entry = f"\n---\n\n"
    entry += f"### Decision #{decision_number}: {title}\n"
    entry += f"**Date:** {timestamp}\n"
    entry += f"**Phase:** {phase}\n"
    entry += f"**Status:** {status}\n\n"

    entry += "**Context:**\n"
    entry += f"{context}\n\n"

    entry += "**Decision:**\n"
    entry += f"{decision}\n\n"

    entry += "**Alternatives Considered:**\n"
    for i, alt in enumerate(alternatives, 1):
        entry += f"{i}. **{alt['name']}**\n"
        entry += f"   - Pros: {alt['pros']}\n"
        entry += f"   - Cons: {alt['cons']}\n\n"

    entry += "**Rationale:**\n"
    entry += f"{rationale}\n\n"

    entry += "**Trade-offs:**\n"
    for tradeoff in tradeoffs:
        entry += f"- {tradeoff}\n"
    entry += "\n"

    entry += "**Consequences:**\n"
    entry += f"- Positive: {consequences.get('positive', 'N/A')}\n"
    entry += f"- Negative: {consequences.get('negative', 'N/A')}\n"
    entry += f"- Risks: {consequences.get('risks', 'N/A')}\n"

    # Append to decisions log
    with open(decisions_path, 'a', encoding='utf-8') as f:
        f.write(entry)

    logger = logging.getLogger(__name__)
    logger.info(f"Logged Decision #{decision_number}: {title}")


def log_challenge(
    challenge_number: int,
    title: str,
    phase: str,
    severity: str,
    problem: str,
    context: str,
    impact: str,
    attempts: List[Dict[str, str]],
    solution: str,
    prevention: str,
    lessons: str,
    status: str = "Resolved"
) -> None:
    """
    Log a challenge/problem to the challenges.md file.

    Args:
        challenge_number: Unique challenge number
        title: Title of the challenge
        phase: Phase number
        severity: Low / Medium / High / Critical
        problem: Description of the problem
        context: What were you trying to do
        impact: How this affected the project
        attempts: List of dicts with 'approach', 'result', 'outcome'
        solution: Final solution that worked
        prevention: How to avoid this in the future
        lessons: Key takeaways
        status: Challenge status (default: "Resolved")

    Example:
        >>> log_challenge(
        ...     challenge_number=1,
        ...     title="LLM Returns Invalid JSON",
        ...     phase="Phase 2",
        ...     severity="High",
        ...     problem="LLM sometimes returns malformed JSON",
        ...     context="Extracting structured data from resumes",
        ...     impact="Pipeline fails on ~20% of resumes",
        ...     attempts=[
        ...         {
        ...             'approach': 'Stricter prompt',
        ...             'result': 'Improved to 90% success',
        ...             'outcome': 'partial'
        ...         }
        ...     ],
        ...     solution="Use Pydantic output parser with retry logic",
        ...     prevention="Always use output parsers, not raw LLM output",
        ...     lessons="Structured output requires robust parsing"
        ... )
    """
    challenges_path = LOGS_DIR / "challenges.md"
    timestamp = datetime.now().strftime("%Y-%m-%d")

    entry = f"\n---\n\n"
    entry += f"### Challenge #{challenge_number}: {title}\n"
    entry += f"**Date Encountered:** {timestamp}\n"
    entry += f"**Phase:** {phase}\n"
    entry += f"**Severity:** {severity}\n"
    entry += f"**Status:** {status}\n\n"

    entry += "**Problem Description:**\n"
    entry += f"{problem}\n\n"

    entry += "**Context:**\n"
    entry += f"{context}\n\n"

    entry += "**Impact:**\n"
    entry += f"{impact}\n\n"

    entry += "**Attempted Solutions:**\n"
    for i, attempt in enumerate(attempts, 1):
        entry += f"{i}. **{attempt.get('approach', 'N/A')}**\n"
        entry += f"   - Result: {attempt.get('result', 'N/A')}\n"
        entry += f"   - Outcome: {attempt.get('outcome', 'N/A')}\n\n"

    entry += "**Final Solution:**\n"
    entry += f"{solution}\n\n"

    entry += "**Prevention:**\n"
    entry += f"{prevention}\n\n"

    entry += "**Lessons Learned:**\n"
    entry += f"{lessons}\n"

    # Append to challenges log
    with open(challenges_path, 'a', encoding='utf-8') as f:
        f.write(entry)

    logger = logging.getLogger(__name__)
    logger.info(f"Logged Challenge #{challenge_number}: {title}")


def checkpoint(phase: str, step: str, findings: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    Create a checkpoint entry in the learning log after completing a major step.

    Args:
        phase: Current phase (e.g., "Phase 2")
        step: Step completed (e.g., "Step 2.1")
        findings: Key findings and observations
        metrics: Optional dict of metrics/measurements

    Example:
        >>> checkpoint(
        ...     phase="Phase 2",
        ...     step="Step 2.1 - LLM Testing",
        ...     findings="Llama3.2:3b provides best balance of speed and accuracy",
        ...     metrics={'avg_latency': '1.2s', 'accuracy': '92%'}
        ... )
    """
    learning_log_path = LOGS_DIR / "learning_log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = f"\n#### {step} - Checkpoint\n"
    entry += f"**Timestamp:** {timestamp}\n\n"
    entry += f"**Findings:**\n{findings}\n\n"

    if metrics:
        entry += "**Metrics:**\n"
        for key, value in metrics.items():
            entry += f"- {key}: {value}\n"
        entry += "\n"

    # This is a simplified append - in practice, you might want to insert
    # in the correct phase section rather than just appending
    with open(learning_log_path, 'a', encoding='utf-8') as f:
        f.write(entry)

    logger = logging.getLogger(__name__)
    logger.info(f"Checkpoint recorded: {phase} - {step}")


# Initialize a default logger for this module
logger = setup_logger(__name__)


if __name__ == "__main__":
    # Test the logging utilities
    logger.info("Testing logging utilities...")

    # Test experiment logging
    log_experiment(
        experiment_number=0,
        name="Test Experiment",
        phase="Phase 0 - Testing",
        objective="Verify logging utilities work correctly",
        hypothesis="Logging functions will create proper markdown entries",
        options_tested=[
            {
                'name': 'Option A',
                'configuration': 'Test config',
                'results': 'Success',
                'observations': 'Works as expected'
            }
        ],
        metrics_used={'Success Rate': 'Whether function executes without error'},
        winner='Option A',
        rationale='Only option tested',
        key_learning='Logging utilities are functional',
        impact='Can now track experiments throughout project',
        next_steps='Use in actual experiments'
    )

    logger.info("Logging utilities test complete. Check logs/ directory for output.")