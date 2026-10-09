from .base import ComponentResult
from .abbreviation import load_glossary

tool = None


def get_language_tool():
    global tool
    if tool is None:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
    return tool


def language_tool_python_utils_correct(text: str, matches) -> str:
    import language_tool_python
    return language_tool_python.utils.correct(text, matches)


def correct_grammar(text: str) -> ComponentResult:
    tool = get_language_tool()
    glossary = load_glossary()
    protected = set(glossary.get("do_not_expand", [])) | set(glossary.get("entries", {}).keys())

    matches = tool.check(text)
    matches = [m for m in matches if text[m.offset:m.offset + m.error_length].strip(".,;:!?") not in protected]

    corrected = language_tool_python_utils_correct(text, matches)

    applied = []
    for match in matches:
        applied.append({
            "message": match.message,
            "rule_id": match.rule_id,
            "offset": match.offset,
            "replacements": match.replacements[:3],
        })

    return ComponentResult(
        text=corrected,
        changed=(corrected != text),
        metadata={"component": "gec", "matches": applied},
    )
