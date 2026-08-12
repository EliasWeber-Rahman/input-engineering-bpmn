from typing import List, Tuple
from .base import ComponentResult
from .. import config

PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their", "hers", "theirs"}
spacy_model = None
fastcoref_model = None


def get_spacy_model():
    global spacy_model
    if spacy_model is None:
        import spacy
        spacy_model = spacy.load(config.SPACY_MODEL)
    return spacy_model


def get_fastcoref_model():
    global fastcoref_model
    if fastcoref_model is None:
        from fastcoref import FCoref
        fastcoref_model = FCoref()
    return fastcoref_model


def by_phrase(parsed_text, start_char: int) -> bool:
    span = parsed_text.char_span(start_char, start_char + 1, alignment_mode="expand")
    if span is None:
        return False
    word = span[0]
    for ancestor in word.ancestors:
        if ancestor.dep_ == "agent" or (ancestor.text.lower() == "by" and ancestor.dep_ == "prep"):
            return True
    return False


def filter_clusters(text: str, clusters: List[List[Tuple[int, int]]]) -> List[dict]:
    spacy_nlp = get_spacy_model()
    parsed_text = spacy_nlp(text)
    resolvable = []
    for cluster in clusters:
        pronoun_spans = []
        non_pronoun_mentions = []
        for start, end in cluster:
            mention_text = text[start:end]
            if mention_text.lower().strip() in PRONOUNS:
                pronoun_spans.append((start, end))
            else:
                non_pronoun_mentions.append(mention_text)
        if not pronoun_spans or not non_pronoun_mentions:
            continue

        antecedent = non_pronoun_mentions[0]
        for mention in non_pronoun_mentions[1:]:
            if len(mention) > len(antecedent):
                antecedent = mention

        valid_pronoun_spans = []
        for start, end in pronoun_spans:
            if not by_phrase(parsed_text, start):
                valid_pronoun_spans.append((start, end))
        if not valid_pronoun_spans:
            continue

        resolvable.append({"antecedent": antecedent, "pronoun_spans": valid_pronoun_spans})
    return resolvable


def apply_resolution(text: str, resolvable: List[dict]) -> Tuple[str, List[dict]]:
    edits = []
    for item in resolvable:
        for start, end in item["pronoun_spans"]:
            edits.append((start, end, item["antecedent"]))
    edits.sort(reverse=True)

    result = text
    applied = []
    for start, end, antecedent in edits:
        original_mention = result[start:end]
        result = result[:start] + antecedent + result[end:]
        applied.append({"original": original_mention, "replacement": antecedent, "offset": start})
    return result, applied


def resolve_coreferences(text: str) -> ComponentResult:
    model = get_fastcoref_model()
    predictions = model.predict(texts=[text])[0]
    clusters = predictions.get_clusters(as_strings=False)
    resolvable = filter_clusters(text, clusters)
    resolved_text, applied = apply_resolution(text, resolvable)

    return ComponentResult(
        text=resolved_text,
        changed=bool(applied),
        metadata={"component": "coreference", "resolved": applied},
    )
