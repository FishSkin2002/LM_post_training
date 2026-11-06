"""Boxed final-answer extraction and pinned Math-Verify scoring."""

from functools import lru_cache
import re
from importlib.metadata import version

from math_verify import LatexExtractionConfig, StringExtractionConfig, parse, verify

SCORING = {
    "math_verify": version("math-verify"), "latex2sympy2_extended": version("latex2sympy2_extended"),
    "antlr4": version("antlr4-python3-runtime"),
    "extraction": "last boxed block; balanced braces; no fallback to reasoning numbers",
    "parsing": "LaTeX environment; no_fallback; A-E string choices; timeout=3s",
    "verification": {"strict": True, "float_rounding": 6, "numeric_precision": 15, "timeout_seconds": 3},
    "truncated_is_incorrect": True,
}


def boxed_answer(text):
    """Return the last boxed block, including nested LaTeX braces, or None."""
    start = text.rfind(r"\boxed")
    if start < 0:
        return None
    opening = re.match(r"\\boxed\s*\{", text[start:])
    if opening is None:
        return None
    begin = start + opening.end()
    depth, escaped = 1, False
    for index in range(begin, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index].strip() or None
    return None


@lru_cache(maxsize=8192)
def parse_answer(text):
    text = text.strip()
    if not text or text.lower() in ("proof", "notfound", "nan", "none"):
        return []
    if re.fullmatch(r"[A-Ea-e]", text):
        return parse(text.upper(), extraction_config=[StringExtractionConfig(lowercase=False)],
                     fallback_mode="no_fallback", parsing_timeout=3)
    if not (text.startswith("$") or text.startswith(r"\(") or text.startswith(r"\[")):
        text = "$" + text + "$"
    return parse(text, extraction_config=[LatexExtractionConfig()],
                 fallback_mode="no_fallback", parsing_timeout=3)


def score_answer(text, reference, truncated=False):
    expected = parse_answer(reference)
    if not expected:
        raise ValueError("Reference final answer is not supported by the scoring parser.")
    content = boxed_answer(text)
    predicted = parse_answer(content) if content is not None else []
    valid = bool(predicted)
    correct = valid and not truncated and verify(expected, predicted, **SCORING["verification"])
    return bool(correct), valid, bool(truncated)
