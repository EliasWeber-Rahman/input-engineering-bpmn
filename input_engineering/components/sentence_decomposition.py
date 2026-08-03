from __future__ import annotations
from typing import List

from .base import ComponentResult
from .. import config

MIN_SENTENCE_LENGTH_FOR_SPLITTING = 12
CONJUNCTIONS = {"and", "then", "but", "or"}

sentence_segmenter = None
spacy_model = None


def get_sentence_segmenter():
    global sentence_segmenter
    if sentence_segmenter is None:
        import pysbd
        sentence_segmenter = pysbd.Segmenter(language="en", clean=False)
    return sentence_segmenter


def get_spacy_model():
    global spacy_model
    if spacy_model is None:
        import spacy
        spacy_model = spacy.load(config.SPACY_MODEL)
    return spacy_model


def has_own_subject_and_verb(clause_words) -> bool:
    found_subject = False
    found_verb = False
    for word in clause_words:
        if word.dep_ in ("nsubj", "nsubjpass"):
            found_subject = True
        if word.pos_ in ("VERB", "AUX"):
            found_verb = True
    return found_subject and found_verb


def find_root_word(parsed_sentence):
    for word in parsed_sentence:
        if word.dep_ == "ROOT":
            return word
    return None


def find_conjunction_word(words):
    for word in words:
        if word.dep_ == "cc":
            return word
    return None


def split_compound_sentence(sentence: str) -> List[str]:
    spacy_nlp = get_spacy_model()
    parsed_sentence = spacy_nlp(sentence)

    if len(parsed_sentence) < MIN_SENTENCE_LENGTH_FOR_SPLITTING:
        return [sentence]

    root_word = find_root_word(parsed_sentence)
    if root_word is None:
        return [sentence]

    for child_word in root_word.children:
        if child_word.dep_ != "conj" or child_word.pos_ not in ("VERB", "AUX"):
            continue

        conjunct_verb = child_word

        conjunction_word = find_conjunction_word(conjunct_verb.children)
        if conjunction_word is None:
            conjunction_word = find_conjunction_word(root_word.children)
        if conjunction_word is None or conjunction_word.text.lower() not in CONJUNCTIONS:
            continue

        second_clause_word_positions = []
        for word in conjunct_verb.subtree:
            if word.i != conjunction_word.i:
                second_clause_word_positions.append(word.i)
        second_clause_start = min(second_clause_word_positions)
        second_clause_end = max(second_clause_word_positions)

        split_point = min(conjunction_word.i, second_clause_start)
        if split_point > 0:
            first_clause_span = parsed_sentence[0:split_point]
        else:
            first_clause_span = None
        second_clause_span = parsed_sentence[second_clause_start: second_clause_end + 1]

        if first_clause_span is None or len(first_clause_span) == 0 or len(second_clause_span) == 0:
            continue
        if not has_own_subject_and_verb(first_clause_span) or not has_own_subject_and_verb(second_clause_span):
            continue

        first_sentence = first_clause_span.text.strip().rstrip(",")
        second_sentence = second_clause_span.text.strip()
        if not first_sentence.endswith((".", "!", "?")):
            first_sentence += "."
        if second_sentence:
            second_sentence = second_sentence[0].upper() + second_sentence[1:]
        if not second_sentence.endswith((".", "!", "?")):
            second_sentence += "."
        return [first_sentence, second_sentence]

    return [sentence]


def decompose_sentences(text: str) -> ComponentResult:
    segmenter = get_sentence_segmenter()
    sentences = segmenter.segment(text)

    output_sentences: List[str] = []
    split_count = 0
    for sentence in sentences:
        parts = split_compound_sentence(sentence.strip())
        if len(parts) > 1:
            split_count += 1
        output_sentences.extend(parts)

    non_empty_sentences = []
    for sentence in output_sentences:
        if sentence:
            non_empty_sentences.append(sentence)
    result_text = " ".join(non_empty_sentences)

    return ComponentResult(
        text=result_text,
        changed=(split_count > 0),
        metadata={
            "component": "sentence_decomposition",
            "original_sentence_count": len(sentences),
            "output_sentence_count": len(output_sentences),
            "sentences_split": split_count,
        },
    )
