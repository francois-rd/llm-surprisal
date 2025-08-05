from dataclasses import dataclass
import re

from .base import LLM, LLMOutput, Nickname
from ..core import Logprob, Logprobs, RankedLogprob
from ..prompting import PromptData


@dataclass
class DummyConfig:
    dummy_cfg_unused: str = ""


class DummyLLM(LLM):
    pattern = re.compile(r"(\w+\s*)|(\W+)")

    def __init__(self, nickname: Nickname, llm_cfg: DummyConfig, *args, **kwargs):
        super().__init__(nickname, *args, **kwargs)
        self.cfg = llm_cfg

    def invoke(self, prompt_data: PromptData, *args, **kwargs) -> LLMOutput:
        text = prompt_data.additional_data.get("generated_text", prompt_data.label)
        message = prompt_data.additional_data.get("error_message", None)
        data = prompt_data.additional_data.get("derived_data", {})
        data["logprobs"] = self._fake_logprobs(text)
        data["prompt_logprobs"] = self._fake_logprobs(prompt_data.messages[1].text)
        return LLMOutput(generated_text=text, error_message=message, derived_data=data)

    @classmethod
    def _fake_logprobs(cls, text: str) -> Logprobs:
        logprobs = []
        for first, second in cls.pattern.findall(text):
            if len(first) == len(second) == 0:
                raise ValueError(f"Bad regex: Empty split into tokens: {text}")
            elif len(first) != 0 and len(second) != 0:
                raise ValueError(f"Bad regex: Multi split into tokens: {text}")
            elif len(first) == 0:
                token = second
            else:
                token = first
            logprob = RankedLogprob(
                chosen=Logprob(token=token, rank=2, logprob=-1),
                others={},
                ranking="absolute",
            )
            logprobs.append(logprob)
        return Logprobs(sequence=logprobs)
