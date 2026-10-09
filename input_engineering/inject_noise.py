import random
import re
from typing import Dict, List, NamedTuple, Tuple

from . import config
from .noise_metrics import edit_rate


class Edit(NamedTuple):
    start: int
    end: int
    replacement: str


def apply_edits(text: str, edits: List[Edit]) -> str:
    result = text
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result



def add_random_edits_until_target(
    text: str, rng: random.Random, target_edit_rate: float, candidate_edits: List[Edit]
) -> str:
    if not candidate_edits:
        return text

    shuffled = list(candidate_edits)
    rng.shuffle(shuffled)
    accepted: List[Edit] = []
    noised = text
    for edit in shuffled:
        accepted.append(edit)
        noised = apply_edits(text, accepted)
        if edit_rate(text, noised) >= target_edit_rate:
            break
    return noised


def find_normalization_edits(text: str) -> List[Edit]:
    edits: List[Edit] = []
    for match in re.finditer(r"([.,;:!?()])", text):
        edits.append(Edit(match.start(), match.end(), f" {match.group(1)} "))
    for match in re.finditer(r"\.", text):
        edits.append(Edit(match.end(), match.end(), " ."))
    return edits


def inject_normalization_noise(text: str, rng: random.Random, target_edit_rate: float) -> str:
    noised = add_random_edits_until_target(text, rng, target_edit_rate, find_normalization_edits(text))
    return re.sub(r"[ \t]{2,}", " ", noised).strip()


PHRASE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b")
ARTICLE = re.compile(r"^(The|A|An)\s+")


def find_abbreviation_edits(text: str) -> List[Edit]:
    phrase_counts: Dict[str, int] = {}
    for match in PHRASE_PATTERN.finditer(text):
        phrase = ARTICLE.sub("", match.group(1))
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    edits: List[Edit] = []
    for phrase, count in phrase_counts.items():
        if count < 2:
            continue
        initials = "".join(word[0] for word in phrase.split())
        pattern = re.compile(rf"\b(?:(?:The|A|An)\s+)?{re.escape(phrase)}\b")
        occurrences = [(m.start(), m.end()) for m in pattern.finditer(text)]
        for start, end in occurrences[1:]:
            edits.append(Edit(start, end, initials))
    return edits


def inject_abbreviation_noise(text: str, rng: random.Random, target_edit_rate: float) -> str:
    return add_random_edits_until_target(text, rng, target_edit_rate, find_abbreviation_edits(text))


ARTICLES = ("the", "a", "an")
TYPOS = [
    ("the", "teh"),
    ("and", "adn"),
    ("for", "fro"),
    (r"([a-z])\1", r"\1"),
]


def find_gec_edits(text: str) -> List[Edit]:
    edits: List[Edit] = []
    for match in re.finditer(r"\S+", text):
        word = match.group(0)
        lowered = word.lower().strip(".,;:!?")

        if lowered in ARTICLES:
            end = match.end()
            if end < len(text) and text[end] == " ":
                end += 1
            edits.append(Edit(match.start(), end, ""))
            continue

        for pattern, replacement in TYPOS:
            if re.search(pattern, lowered):
                typo = re.sub(pattern, replacement, word, count=1)
                edits.append(Edit(match.start(), match.end(), typo))
                break
    return edits


def inject_gec_noise(text: str, rng: random.Random, target_edit_rate: float) -> str:
    return add_random_edits_until_target(text, rng, target_edit_rate, find_gec_edits(text))



NOUN_PHRASE_PATTERN = re.compile(r"\bthe ([a-z]+)\b", re.IGNORECASE)


def find_coreference_edits(text: str) -> List[Edit]:
    phrase_occurrences: Dict[str, List[Tuple[int, int]]] = {}
    for match in NOUN_PHRASE_PATTERN.finditer(text):
        key = match.group(0).lower()
        phrase_occurrences.setdefault(key, []).append((match.start(), match.end()))

    edits: List[Edit] = []
    for occurrences in phrase_occurrences.values():
        if len(occurrences) < 2:
            continue
        for start, end in occurrences[1:]:
            pronoun = "It" if text[start].isupper() else "it"
            edits.append(Edit(start, end, pronoun))
    return edits


def inject_coreference_noise(text: str, rng: random.Random, target_edit_rate: float) -> str:
    return add_random_edits_until_target(text, rng, target_edit_rate, find_coreference_edits(text))


def inject_sentence_decomposition_noise(text: str, rng: random.Random, target_edit_rate: float) -> str:
    import pysbd
    segmenter = pysbd.Segmenter(language="en", clean=False)
    noised = text
    for i in range(50):
        sentences = segmenter.segment(noised)
        if len(sentences) < 2:
            break

        merge_index = rng.randrange(len(sentences) - 1)
        first = sentences[merge_index].strip().rstrip(".")
        second = sentences[merge_index + 1].strip()
        if second:
            second = second[0].lower() + second[1:]
        sentences[merge_index: merge_index + 2] = [f"{first} and {second}"]
        noised = " ".join(sentences)

        if edit_rate(text, noised) >= target_edit_rate:
            break
    return noised


NOISE_INJECTORS = {
    "normalization": inject_normalization_noise,
    "abbreviation": inject_abbreviation_noise,
    "gec": inject_gec_noise,
    "coreference": inject_coreference_noise,
    "sentence_decomposition": inject_sentence_decomposition_noise,
}


def inject_combined_noise(text: str, rng: random.Random, target_edit_rate: float) -> str:
    noised = text
    for component in config.COMPONENT_ORDER:
        noised = NOISE_INJECTORS[component](noised, rng, target_edit_rate)
    return noised

