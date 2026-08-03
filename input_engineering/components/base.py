from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ComponentResult:
    text: str
    changed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # quick printing/debugging
        return self.text
