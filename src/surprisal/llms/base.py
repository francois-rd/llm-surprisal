from dataclasses import dataclass
from typing import Any
from enum import Enum

from ..prompting import PromptData


# A name to uniquely differentiate different LLMs.
Nickname = str
MISSING_NICKNAME: Nickname = "<missing-llm>"


class LLMImplementation(Enum):
    MISSING = "MISSING"
    DUMMY = "DUMMY"
    OPENAI = "OPENAI"


@dataclass
class LLMOutput:
    # The LLM's generated output text, or None if an error occurred.
    generated_text: str | None

    # Non-None only if an error occurred, in which case the error message is given here.
    error_message: str | None

    # Any other derived output data, or None if an error occurred or no data exists.
    # Known options:
    #  - logprobs
    #  - prompt_logprobs
    derived_data: dict[str, Any] | None = None


class LLM:
    """Wrapper interface for all LLM implementations."""

    def __init__(self, nickname: Nickname, *args, **kwargs):
        self.nickname = nickname

    def invoke(self, prompt_data: PromptData, *args, **kwargs) -> LLMOutput:
        """Invokes the underlying LLM, returning its output."""
        raise NotImplementedError


@dataclass
class LLMsConfig:
    # The nickname of the LLM to load.
    llm: Nickname = MISSING_NICKNAME

    # The implementation to use to load the LLM.
    implementation: LLMImplementation = LLMImplementation.MISSING
