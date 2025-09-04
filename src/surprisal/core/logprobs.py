from typing import Any, Callable, Iterable, Literal
from dataclasses import dataclass
from enum import Enum
import re

Rank = int
Token = str
PrefixWhitespace = str


@dataclass
class Logprob:
    token: Token
    rank: Rank
    logprob: float


@dataclass
class RankedLogprob:
    chosen: Logprob
    others: dict[Rank, Logprob]
    ranking: Literal["relative", "absolute"]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RankedLogprob":
        return RankedLogprob(
            chosen=Logprob(**data["chosen"]),
            others={rank: Logprob(**d) for rank, d in data["others"].items()},
            ranking=data["ranking"],
        )

    def clean_up_token(self) -> str:
        special = ".^$*+?{}[]()\\"
        return "".join(["\\" * (c in special) + c for c in self.chosen.token.strip()])


@dataclass
class SpacedSubsequence:
    indices: dict[int, PrefixWhitespace]
    reference: "Logprobs"

    def to_text(self) -> str:
        text = []
        for index, whitespace in self.indices.items():
            token = self.reference.sequence[index].chosen.token.strip()
            text.append(whitespace + token)
        return "".join(text)

    def to_chosen_logprobs(self) -> list[float]:
        return self.reference.to_chosen_logprobs(min(self.indices), max(self.indices))

    def verify(self, prefix_text: str | None, suffix_text: str | None) -> bool:
        if not self.indices:
            # If empty, cannot really verify accurately.
            return False
        has_prefix = self._do_verify(prefix_text, min(self.indices.keys()) - 1, max)
        has_suffix = self._do_verify(suffix_text, max(self.indices.keys()) + 1, min)
        return has_prefix and has_suffix

    def _do_verify(self, text: str | None, target_idx: int, agg: Callable) -> bool:
        if text is None:
            return False
        for ss in self.reference.indices_of(text):
            if ss.indices and agg(ss.indices.keys()) == target_idx:
                return True
        return False


@dataclass
class Logprobs:
    sequence: list[RankedLogprob]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Logprobs":
        return Logprobs(sequence=[RankedLogprob.from_dict(d) for d in data["sequence"]])

    def maybe_trim(self, trim_indicator: str | None) -> None:
        if trim_indicator is not None:
            sequences = list(self.indices_of(trim_indicator))
            start_idx = min(sequences[0].indices) if len(sequences) == 1 else 0
            self.sequence = self.sequence[start_idx:]

    def to_text(self, start_idx: int = 0, end_idx: int | None = None) -> str:
        end_idx = len(self.sequence) if end_idx is None else end_idx + 1
        return "".join(self.sequence[i].chosen.token for i in range(start_idx, end_idx))

    def to_chosen_logprobs(
        self, start_idx: int = 0, end_idx: int | None = None
    ) -> list[float]:
        end_idx = len(self.sequence) if end_idx is None else end_idx + 1
        return [self.sequence[i].chosen.logprob for i in range(start_idx, end_idx)]

    def indices_of(
        self, text: str, start_idx: int = 0, end_idx: int | None = None
    ) -> Iterable[SpacedSubsequence]:
        working_sequence, working_prefix = {}, ""
        end_idx = len(self.sequence) if end_idx is None else end_idx
        for i, logprob in enumerate(self.sequence):
            if i < start_idx or i >= end_idx:
                continue
            whitespace = self._determine_viability(logprob, working_prefix, text)
            if whitespace is None:
                # Not viable and working sequence is incomplete, so reset.
                working_sequence, working_prefix = {}, ""
            else:
                working_sequence[i] = whitespace
                working_prefix += whitespace + logprob.clean_up_token()
            subsequence = SpacedSubsequence(indices=working_sequence, reference=self)
            if subsequence.to_text() == text:
                # Working sequence is complete. Yield and then reset to look for more.
                yield subsequence
                working_sequence, working_prefix = {}, ""

    @staticmethod
    def _determine_viability(
        logprob: RankedLogprob, working_prefix: str, text: str
    ) -> str | None:
        pattern = working_prefix + r"(\s*)" + logprob.clean_up_token()
        match = re.search(pattern, text)
        if match is None:
            return None
        return match.group(1)


AggregatorStr = str


class AggregatorOption(Enum):
    SUM = "SUM"
    MEAN = "MEAN"
    FIRST = "FIRST"
    LAST = "LAST"
    MIN = "MIN"
    MAX = "MAX"

    def aggregate(self, logprobs: list[float], negate: bool = False) -> float:
        if self == AggregatorOption.SUM:
            return (-1 if negate else 1) * sum(logprobs)
        elif self == AggregatorOption.MEAN:
            # Logprobs should never be empty, so division by 0 should never happen.
            return (-1 if negate else 1) * sum(logprobs) / len(logprobs)
        elif self == AggregatorOption.FIRST:
            # Logprobs should never be empty, so indexing error should never happen.
            return (-1 if negate else 1) * logprobs[0]
        elif self == AggregatorOption.LAST:
            # Logprobs should never be empty, so indexing error should never happen.
            return (-1 if negate else 1) * logprobs[-1]
        elif self == AggregatorOption.MIN:
            return (-1 if negate else 1) * min(logprobs)
        elif self == AggregatorOption.MAX:
            return (-1 if negate else 1) * max(logprobs)
        else:
            raise ValueError(f"Unsupported aggregator: {self}")
