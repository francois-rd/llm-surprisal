from typing import Callable
import re

from .base import JSONParser, OutputParser, RegexExtractionParser
from ..core import Label, Term


def _get_answer_choices(**kwargs) -> dict[Label, Term]:
    if "answer_choices" not in kwargs:
        raise ValueError(
            f"'kwargs' must contain an 'answer_choices' field of duck type "
            f"'dict[Label, Term]'. Got: {kwargs}"
        )
    return kwargs["answer_choices"]


class MCQParser(OutputParser):
    def __call__(self, generated_text: str, *args, **kwargs) -> Label | None:
        """Parses the generated_text of an LLM to extract a Label or None on failure."""
        raise NotImplementedError


class MCQPatternParser(MCQParser):
    def __init__(
        self,
        pattern_format: str,
        match_label_group: int | None = 1,
        match_term_group: int | None = 1,
        comparator: Callable[[str, str], bool] = lambda x, y: x.lower() == y.lower(),
        flags=re.IGNORECASE,
    ):
        """
        Uses a regex pattern to parse LLM output that corresponds to some option from
        the answer choices of a multiple-choice question. Returns a Label or None on
        failure.

        :param pattern_format: A regex pattern with optional LABEL and TERM placeholders
            that will be replaced inline with the values from the answer choices.
        :param match_label_group: If a group number (not None), attempts to regex
            match against the label value, extracting the corresponding match group.
            Takes precedence over 'match_term_group', but both can be given.
        :param match_term_group: If a group number (not None), attempts to regex
            match against the term value, extracting the corresponding match group.
            'match_label_group' takes precedence over this, but both can be given.
        :param comparator: Comparator between extracted label/term and associated
            label/term as written in the answer choices.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.pattern_format = pattern_format
        self.match_label_group = match_label_group
        self.match_term_group = match_term_group
        self.comparator = comparator
        self.flags = flags
        self.label_parsers = {}
        self.term_parsers = {}

    def __call__(self, generated_text: str, *args, **kwargs) -> Label | None:
        for label, term in _get_answer_choices(**kwargs).items():
            extracted_label = self._label(label, term, generated_text, *args, **kwargs)
            if extracted_label is not None and self.comparator(extracted_label, label):
                return label
            extracted_term = self._term(label, term, generated_text, *args, **kwargs)
            if extracted_term is not None and self.comparator(extracted_term, label):
                return label
        return None

    def _label(
        self,
        label: Label,
        term: Term,
        generated_text: str,
        *args,
        **kwargs,
    ) -> str | None:
        return self._do_extract(
            parsers=self.label_parsers,
            group=self.match_label_group,
            label=label,
            term=term,
            generated_text=generated_text,
            *args,
            **kwargs,
        )

    def _term(
        self,
        label: Label,
        term: Term,
        generated_text: str,
        *args,
        **kwargs,
    ) -> str | None:
        return self._do_extract(
            parsers=self.label_parsers,
            group=self.match_label_group,
            label=label,
            term=term,
            generated_text=generated_text,
            *args,
            **kwargs,
        )

    def _do_extract(
        self,
        parsers: dict[tuple[Label, Term], RegexExtractionParser],
        group: int | None,
        label: Label,
        term: Term,
        generated_text: str,
        *args,
        **kwargs,
    ) -> str | None:
        if group is None:
            return None
        if (label, term) not in parsers:
            pattern = self.pattern_format.replace("LABEL", label).replace("TERM", term)
            parser = RegexExtractionParser(pattern, int(group), flags=self.flags)
            parsers[(label, term)] = parser
        return parsers[(label, term)](generated_text, *args, **kwargs)


class MCQJSONParser(MCQParser):
    def __init__(
        self,
        schema_key: str,
        pattern: str = r"({.*?})",  # NOTE: Doesn't catch JSON objects w/ nested dicts.
        match_label: bool = True,
        match_term: bool = True,
        comparator: Callable[[str, str], bool] = lambda x, y: x.lower() == y.lower(),
        flags=re.IGNORECASE,
    ):
        """
        Extracts JSON objects from generated_text, checking whether the value at the
        schema_key in each object corresponds to a label/term. Returns None on failure.

        :param schema_key: The key into the JSON object containing the answer Label.
        :param pattern: A regex pattern to extract JSON objects from generated_text
            that may also include other text.
        :param match_label: Whether the value corresponding to the schema key in the
            JSON object should match against the label value. Takes precedence over
            match_term, but both can be True.
        :param match_term: Whether the value corresponding to the schema key in the
            JSON object should match against the term value. match_label takes
            precedence over this, but both can be True.
        :param comparator: Comparator between extracted label/term and associated
            label/term as written in the QAData answer choices.
        :param flags: Flags to pass to re.compile(), if any.
        """
        super().__init__()
        self.parser = JSONParser(schema_key=schema_key, pattern=pattern, flags=flags)
        self.match_label = match_label
        self.match_term = match_term
        self.comparator = comparator

    def __call__(self, generated_text: str, *args, **kwargs) -> Label | None:
        value = self.parser(generated_text=generated_text, *args, **kwargs)
        for label, term in _get_answer_choices(**kwargs).items():
            if self.match_label and self.comparator(value, label):
                return label
            if self.match_term and self.comparator(value, term):
                return label
        return None


class SimpleMCQParser(MCQParser):
    # NOTE: These are in approximate descending order of confidence in the pattern.
    default_sub_parsers: list[MCQParser] = [
        MCQPatternParser(r'^\s*"?(LABEL)"?$'),
        MCQJSONParser("answer"),
        MCQPatternParser(r'"?(LABEL)"?\s*:\s*"?TERM"?', match_term_group=None),
        MCQPatternParser(r'Answer:\s*"?(LABEL|TERM)"?'),
        MCQPatternParser(r'{\s*"?answer"?\s*:\s*"?(LABEL|TERM)"?\s*}'),
        MCQPatternParser(r'{\s*"?answer"?\s*:\s*"(LABEL|TERM)"\s*}?'),
        MCQPatternParser(
            r'{\s*"?answer"?\s*:\s*"?(LABEL)\s*:\s*TERM"?\s*}?', match_term_group=None
        ),
        MCQPatternParser(r'answer is:?\s*"?(LABEL|TERM)"?'),
        MCQPatternParser(r'^\s*"?(LABEL|TERM)"?\n'),
    ]

    def __init__(self, sub_parsers: list[MCQParser] = None):
        """
        Parses LLM output using sub-parsers. These are checked in order, and so should
        be given in descending order of confidence in their ability to extract a valid
        label from the generated text. Uses SimpleLLMOutputParser.default_sub_parsers
        if sub_parsers is None.

        :param sub_parsers: Optional list of sub-parsers to call, in order.
        """
        super().__init__()
        self.parsers = self.default_sub_parsers if sub_parsers is None else sub_parsers

    def __call__(
        self,
        generated_text: str,
        *args,
        **kwargs,
    ) -> Label | None:
        """Returns the Label of the first successful sub-parser, or None on failure."""
        for parser in self.parsers:
            label = parser(generated_text, *args, **kwargs)
            if label is not None:
                return label
        return None
