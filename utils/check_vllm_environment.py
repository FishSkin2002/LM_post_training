"""Check the CUDA 12.8 vLLM environment without loading a model."""

import json

from utils.generation import validate_vllm_environment


if __name__ == "__main__":
    print(json.dumps(validate_vllm_environment(training=True), indent=2))
