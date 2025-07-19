from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from .base import (
    JSONParser,
    NoOutputParser,
    RegexMatchParser,
    RegexExtractionParser,
    OptionsParser,
    OutputParser,
)
from .parsers import MCQJSONParser, MCQPatternParser, SimpleMCQParser


ParserID = str


class ParserType(Enum):
    """Enumeration of all managed parser types."""

    JSON = "JSON"
    NONE = "NONE"
    MATCH = "MATCH"
    EXTRACT = "EXTRACT"
    OPTIONS = "OPTIONS"
    MCQ_JSON = "MCQ_JSON"
    MCQ_PATTERN = "MCQ_PATTERN"
    SIMPLE_MCQ = "SIMPLE_MCQ"


def parser_factory(type_: ParserType, parser_data: dict[str, Any]) -> OutputParser:
    if type_ == ParserType.JSON:
        return JSONParser(**parser_data)
    elif type_ == ParserType.NONE:
        return NoOutputParser(**parser_data)
    elif type_ == ParserType.MATCH:
        return RegexMatchParser(**parser_data)
    elif type_ == ParserType.EXTRACT:
        return RegexExtractionParser(**parser_data)
    elif type_ == ParserType.OPTIONS:
        return OptionsParser(**parser_data)
    elif type_ == ParserType.MCQ_JSON:
        return MCQJSONParser(**parser_data)
    elif type_ == ParserType.MCQ_PATTERN:
        return MCQPatternParser(**parser_data)
    elif type_ == ParserType.SIMPLE_MCQ:
        return SimpleMCQParser(**parser_data)
    else:
        raise ValueError(f"Unsupported parser type: {type_}")


@dataclass
class ParserInfo:
    parser_type: ParserType
    parser_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsersConfig:
    parsers: dict[ParserID, ParserInfo]


class ParserManager:
    def __init__(self, cfg: ParsersConfig):
        self.parsers = {}
        for parser_id, info in cfg.parsers.items():
            self.parsers[parser_id] = parser_factory(info.parser_type, info.parser_data)

    def get(self, parser_id: ParserID) -> OutputParser:
        try:
            return self.parsers[parser_id]
        except KeyError:
            raise ValueError(f"Missing parser data for: '{parser_id}'.")
