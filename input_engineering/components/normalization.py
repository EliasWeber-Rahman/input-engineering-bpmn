import re
import unicodedata
import ftfy

from .base import ComponentResult

NO_SPACE_BEFORE = re.compile(r"\s+([.,;:!?%\)\]\}])")
NO_SPACE_AFTER = re.compile(r"([\(\[\{])\s+")
MULTI_SPACE = re.compile(r"[ \t]{2,}")
DOUBLE_PERIOD = re.compile(r"\.\s*\.")
BACKTICK_QUOTE = re.compile(r"``")
APOSTROPHE_QUOTE = re.compile(r"''")


def normalize(text: str) -> ComponentResult:
    original = text

    # fix encoding errors
    fixed = ftfy.fix_text(text)

    # Unicode fix
    fixed = unicodedata.normalize("NFKC", fixed)

    # fix quotes
    fixed = BACKTICK_QUOTE.sub('"', fixed)
    fixed = APOSTROPHE_QUOTE.sub('"', fixed)

    # fix spacing
    fixed = NO_SPACE_BEFORE.sub(r"\1", fixed)
    fixed = NO_SPACE_AFTER.sub(r"\1", fixed)

    # fix periods
    fixed = re.sub(r"\.\.\.", "@!", fixed)
    fixed = DOUBLE_PERIOD.sub(".", fixed)
    fixed = fixed.replace("@!", "...")

    # fix whitespace
    fixed = MULTI_SPACE.sub(" ", fixed)
    fixed = re.sub(r"\s+\n", "\n", fixed)
    fixed = fixed.strip()

    return ComponentResult(
        text=fixed,
        changed=(fixed != original),
        metadata={"component": "normalization"},
    )
