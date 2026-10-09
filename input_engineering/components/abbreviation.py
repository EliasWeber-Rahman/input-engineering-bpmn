import json
import re
from typing import Dict, List, Tuple

from .base import ComponentResult
from .. import config

ACRONYM = re.compile(r"\b([A-Z]{2,10})\b")

LONG_THEN_SHORT = re.compile(r"\b([A-Za-z][A-Za-z \-]{2,60}?)\s*\(\s*([A-Za-z]{2,10})\s*\)")
SHORT_THEN_LONG = re.compile(r"\b([A-Z]{2,10})\s*\(\s*([a-z][a-zA-Z \-]{2,60}?)\s*\)")


def load_glossary() -> dict:
    with open(config.GLOSSARY_PATH, "r", encoding="utf-8") as glossary_file:
        return json.load(glossary_file)


def is_valid_abbreviation(long_form: str, short_form: str) -> bool:
    words = long_form.strip().replace("-", " ").split()

    initials = ""
    for word in words:
        initials += word[0]
    initials = initials.upper()

    if short_form.upper() in initials:
        return True
    return short_form.upper() == initials[-len(short_form):]


def find_definitions_in_text(text: str) -> Dict[str, str]:
    found: Dict[str, str] = {}

    for match in LONG_THEN_SHORT.finditer(text):
        long_form = match.group(1).strip()
        short_form = match.group(2)
        if short_form.isupper() and is_valid_abbreviation(long_form, short_form):
            found[short_form] = long_form

    for match in SHORT_THEN_LONG.finditer(text):
        short_form = match.group(1)
        long_form = match.group(2).strip()
        found[short_form] = long_form

    return found


def replace_all(text: str, definitions: Dict[str, str]) -> Tuple[str, List[dict]]:
    applied = []
    result = text
    for short_form, long_form in definitions.items():
        pattern = re.compile(rf"\b{re.escape(short_form)}\b")
        count = len(pattern.findall(result))
        if count:
            result = pattern.sub(long_form, result)
            applied.append({"acronym": short_form, "expansion": long_form, "count": count})
    return result, applied


def expand_abbreviations(text: str) -> ComponentResult:
    glossary = load_glossary()
    entries = glossary["entries"]
    do_not_expand = set(glossary.get("do_not_expand", []))

    in_text_defs: Dict[str, str] = {}
    for acronym, long_form in find_definitions_in_text(text).items():
        if acronym not in do_not_expand:
            in_text_defs[acronym] = long_form

    candidates = set(ACRONYM.findall(text))
    glossary_defs: Dict[str, str] = {}
    for acronym in candidates:
        already_handled = acronym in in_text_defs or acronym in do_not_expand
        if acronym in entries and not already_handled:
            glossary_defs[acronym] = entries[acronym]

    all_defs = dict(glossary_defs)
    all_defs.update(in_text_defs)

    expanded_text, applied = replace_all(text, all_defs)

    skipped = sorted(acronym for acronym in candidates if acronym in do_not_expand)

    return ComponentResult(
        text=expanded_text,
        changed=bool(applied),
        metadata={
            "component": "abbreviation",
            "expanded": applied,
            "skipped_do_not_expand": skipped,
        },
    )
