from typing import Any
import json
import re


def re_compile(pattern: str, flags=None) -> re.Pattern:
    return re.compile(pattern) if flags is None else re.compile(pattern, flags=flags)


class OutputParser:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, generated_text: str, *args, **kwargs) -> Any | None:
        """
        Parses the generated_text of an LLM to extract some meaningful result.
        Returns None on parsing failure.
        """
        raise NotImplementedError


class StringOutputParser(OutputParser):
    def __call__(self, generated_text: str, *args, **kwargs) -> str | None:
        """
        Parses the generated_text of an LLM to extract a string result.
        This string can represent the label of an answer choice, or some
        unbounded free text, for example. Returns None on parsing failure.
        """
        raise NotImplementedError


class NoOutputParser(StringOutputParser):
    def __call__(self, generated_text: str, *args, **kwargs) -> str | None:
        """Returns 'generated_text' directly without performing any parsing."""
        return generated_text


class RegexMatchParser(StringOutputParser):
    def __init__(self, pattern: str, flags=re.IGNORECASE):
        """
        Uses a regex pattern to parse LLM output. Returns a re.Match or None on failure.

        :param pattern: A regex pattern from which to extract a match.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.pattern = re_compile(pattern, flags=flags)

    def __call__(self, generated_text: str, *args, **kwargs) -> re.Match | None:
        return self.pattern.search(generated_text)


class RegexExtractionParser(StringOutputParser):
    def __init__(self, pattern: str, match_group: int = 1, flags=re.IGNORECASE):
        """
        Uses a regex pattern to parse LLM output. Returns a string result from
        the match group or None on failure.

        :param pattern: A regex pattern from which to extract a match.
        :param match_group: The group index of result within the pattern Match object.
        :param flags: Flags to pass to re.search(), if any.
        """
        super().__init__()
        self.parser = RegexMatchParser(pattern, flags=flags)
        self.match_group = match_group

    def __call__(self, generated_text: str, *args, **kwargs) -> str | None:
        match = self.parser(generated_text, *args, **kwargs)
        try:
            return match.group(self.match_group)
        except (AttributeError, ValueError):
            return None


class OptionsParser(OutputParser):
    def __init__(self, options: list[str], flags=re.IGNORECASE):
        """
        Returns the option that the generated_text matches (barring whitespace) or
        None if the text is not an exact match.
        """
        super().__init__()
        self.options = {o: RegexExtractionParser(o, 0, flags=flags) for o in options}

    def __call__(self, generated_text: str, *args, **kwargs) -> str | None:
        for option, parser in self.options.items():
            result = parser(generated_text, *args, **kwargs)
            if result is not None:
                return option
        return None


class JSONParser(StringOutputParser):
    def __init__(
        self,
        schema_key: str,
        pattern: str = r"({.*?})",  # NOTE: Doesn't catch JSON objects w/ nested dicts.
        flags=re.IGNORECASE,
    ):
        """
        Extracts JSON objects from generated_text, checking whether the value at the
        schema_key in each object corresponds to a score. Returns None on failure.

        :param schema_key: The key into the JSON object containing the score.
        :param pattern: A regex pattern to extract JSON objects from generated_text
            that may also include other text.
        :param flags: Flags to pass to re.findall(), if any.
        """
        super().__init__()
        self.schema_key = schema_key
        self.pattern = re_compile(pattern, flags=flags)

    def __call__(self, generated_text: str, *args, **kwargs) -> str | None:
        for string in [generated_text, *self.pattern.findall(generated_text)]:
            try:
                return json.loads(string, **kwargs)[self.schema_key]
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                json.decoder.JSONDecodeError,
            ):
                continue
        return None
