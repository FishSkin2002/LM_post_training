"""Shared log filtering for vLLM generation with explicit EOS settings."""

import logging


_LOGGER_NAME = "vllm.inputs.preprocess"
_UNINITIALIZED_EOS_WARNING = "Using None for EOS token id because tokenizer is not initialized"


def _keep_log(record):
    return not (record.name == _LOGGER_NAME and record.levelno == logging.WARNING
                and record.getMessage() == _UNINITIALIZED_EOS_WARNING)


def configure_vllm_logging():
    """After vLLM initializes logging, hide only its unused default EOS warning.

    Repeated calls reuse the same filter; logger levels and handlers stay intact.
    """
    logging.getLogger(_LOGGER_NAME).addFilter(_keep_log)
