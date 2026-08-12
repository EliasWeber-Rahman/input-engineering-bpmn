from .base import ComponentResult

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
    matches = tool.check(text)
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
