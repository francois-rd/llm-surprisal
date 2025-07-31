from dataclasses import dataclass, replace
from typing import Any, Hashable
from enum import Enum
import os
import re

import pandas as pd

from .logprobs import SpacedSubsequence


RelationType = str
Term = str
QueryResult = set[Term]


@dataclass(frozen=True)
class Triplet:
    source: Term
    relation: RelationType
    target: Term


class AntiFactualMethod(Enum):
    """
    The method of ConceptNet-based anti-factual instantiation to use. Each method
    limits the choice of anti-factual variable instantiation in a different way.

    Notes:
    1. The anti-factual target variable, v, is chosen from ConceptNet's vocabulary,
       rather than some alternative vocabulary. This limits out-of-distribution effects.
    2. We assume that if an assertion is not in ConceptNet, then it is anti-factual.
       This is not technically correct, since ConceptNet is not a complete KB of
       factual commonsense assertions. See ACCORD for rationale.

    ALL_RELATIONS:
        For a given relation type, r, and source variable instantiation, s, the
        target variable can take on any target value in ConceptNet, v, that is not an
        existing assertion in ConceptNet. That is, any v from any factual assertion
        (s', r', v) in ConceptNet is allowed so long as (s, r, v) is not an existing
        assertion in ConceptNet.

    SAME_RELATION:
        For a given relation type, r, and source variable instantiation, s, the
        target variable can take on any target value in ConceptNet, v, that is not an
        existing assertion in ConceptNet, but *does* share the same relation type.
        That is, any v from any factual assertion (s', r, v) in ConceptNet is allowed
        so long as (s, r, v) is not an existing assertion in ConceptNet.

    OTHER_RELATIONS:
        For a given relation type, r, and source variable instantiation, s, the
        target variable can take on any target value in ConceptNet, v, that is not an
        existing assertion in ConceptNet, and does *not* share the same relation type.
        That is, any v from any factual assertion (s', r', v) where r' != r is allowed
        so long as (s, r, v) is not an existing assertion in ConceptNet.

    SAME_SOURCE:
        For a given relation type, r, and source variable instantiation, s, the
        target variable can take on any target value in ConceptNet, v, that is not an
        existing assertion in ConceptNet, but *does* share the same source variable
        instantiation. That is, any v from any factual assertion (s, r', v) is allowed
        so long as (s, r, v) is not an existing assertion in ConceptNet.
    """

    ALL_RELATIONS = "ALL_RELATIONS"
    SAME_RELATION = "SAME_RELATION"
    OTHER_RELATIONS = "OTHER_RELATIONS"
    SAME_SOURCE = "SAME_SOURCE"


@dataclass
class Query:
    """
    Contains all information relevant to query ConceptNet.

    relation_type: The RelationType to inform querying.
    source_term: The instantiated Term of the source variable to inform querying.
    method: The anti-factual method employed in constructing a QueryResult. If None,
        factual querying is done instead.
    """

    relation_type: RelationType
    source_term: Term
    method: AntiFactualMethod | None


class TermMatchMethod(Enum):
    IDENTICAL = "IDENTICAL"
    STARTSWITH = "STARTSWITH"


class TermMatcher:
    def __init__(self, method: TermMatchMethod):
        # TODO: Take in a bitwise flag mechanism to decide level of strictness:
        #  1. Level of hierarchy of terms to match:
        #     - https://github.com/commonsense/conceptnet5/wiki/API#the-hierarchy-of-terms
        #     - It goes /c/lang/term/<POS-sense>/<DB-that-sense-comes-from>/<sub-sense?>.
        #     - By default, we are currently having to match the entire hierarchy.
        #     - We could allow for less strict matching at any level of the hierarchy.
        #  2. Stemming or plural/singular
        #     - Might be able to get away with SpaCy or NLTK for that.
        #  NOTE: Anything less than IDENTICAL matching would have to be handled very
        #   carefully to not introduce significant noise.
        #  NOTE: Anything less than IDENTICAL matching would require making an decision
        #   about directional matching. For example, does '/c/en/acorn' match with
        #   '/c/en/acorn/n' under a POS-sense match? Moreover, does '/c/en/acorn/n'
        #   match with '/c/en/acorn' under that same scenario? Option 1: No, they don't
        #   match because they are not identical at the desired granularity. Option 2:
        #   only one of these directional variants matches. Specifically, the one that
        #   makes the most sense is to allow the generic '/c/en/acorn' to match with
        #   the specific '/c/en/acorn/n', but not the other way around.
        #   -> ***This option 2 is what the STARTSWITH method implements.***
        self.method = method

    def find_matches(self, df: pd.DataFrame, col: str, term: Term) -> pd.DataFrame:
        if self.method == TermMatchMethod.IDENTICAL:
            return df.loc[df[col] == term]
        elif self.method == TermMatchMethod.STARTSWITH:
            return df.loc[df[col].str.startswith(term)]
        else:
            raise ValueError(f"Unsupported term match method: {self.method}")

    def is_match(self, known_term: Term, test_term: Term) -> bool:
        if self.method == TermMatchMethod.IDENTICAL:
            return known_term == test_term
        elif self.method == TermMatchMethod.STARTSWITH:
            return test_term.startswith(known_term)
        else:
            raise ValueError(f"Unsupported term match method: {self.method}")


@dataclass
class TermComponents:
    language: str
    main: str
    pos: str | None = None
    kb: str | None = None
    sense: str | None = None

    def matches(
        self,
        other: "TermComponents",
        language: bool = False,
        main: bool = False,
        pos: bool = False,
        kb: bool = False,
        sense: bool = False,
        count_none_as_match: bool = False,
    ) -> bool:
        if language and self.language != other.language:
            return False
        if main and self.main != other.main:
            return False
        if pos:
            if self.pos != other.pos:
                return False
            if not count_none_as_match and self.pos is None:
                return False
        if kb:
            if self.kb != other.kb:
                return False
            if not count_none_as_match and self.kb is None:
                return False
        if sense:
            if self.sense != other.sense:
                return False
            if not count_none_as_match and self.sense is None:
                return False
        return True


class TermFormatter:
    def __init__(self, language: str, cache: bool = False):
        self.has_main_pattern = re.compile("/c/../([^/]+)")
        self.has_pos_pattern = re.compile("/c/../([^/]+)/([^/]+)")
        self.has_kb_pattern = re.compile("/c/../([^/]+)/([^/]+)/([^/]+)")
        self.has_sense_pattern = re.compile("/c/../([^/]+)/([^/]+)/([^/]+)/([^/]+)")
        self.language = language
        self.text_cache = {} if cache else None
        self.comp_cache = {} if cache else None

    @staticmethod
    def _check_cache(cache: dict | None, key: Hashable) -> Any | None:
        return None if cache is None else cache.get(key, None)

    @staticmethod
    def _fill_cache(cache: dict | None, key: Hashable, value: Any) -> Any:
        if cache is not None:
            cache.setdefault(key, value)
        return value

    def ensure_plain_text(self, text: str | Term) -> str:
        """Formats text (either ConceptNet Term or plain text) into plain text."""
        result = self._check_cache(self.text_cache, text)
        if result is None:
            main = self.get_main_tag(text)
            result = text if main is None else main.replace("_", " ")
        return self._fill_cache(self.text_cache, text, result)

    def get_main_tag(self, text: str | Term) -> str | None:
        """
        If the text is a Term of the form:
            '/c/<lang>/<main>'
        or
            '/c/<lang>/<main>/<rest-of-hierarchy>'
        extracts the <main> tag.

        Otherwise, returns None.
        """
        match = self.has_main_pattern.match(text)
        if match is None:
            return None  # Likely already plain text.
        return match.group(1)

    def get_pos_tag(self, text: str | Term) -> str | None:
        """
        If the text is a Term of the form:
            '/c/<lang>/<main>/<pos>'
        or
            '/c/<lang>/<main>/<pos>/<rest-of-hierarchy>'
        extracts the <pos> tag.

        Otherwise, returns None.
        """
        match = self.has_pos_pattern.match(text)
        if match is None:
            return None  # Can't find POS tag.
        return match.group(2)

    def get_kb_tag(self, text: str | Term) -> str | None:
        """
        If the text is a Term of the form:
            '/c/<lang>/<main>/<pos>/<kb>'
        or
            '/c/<lang>/<main>/<pos>/<kb>/<sense>'
        extracts the <kb> tag.

        Otherwise, returns None.
        """
        match = self.has_kb_pattern.match(text)
        if match is None:
            return None  # Can't find KB tag.
        return match.group(3)

    def get_sense_tag(self, text: str | Term) -> str | None:
        """
        If the text is a Term of the form:
            '/c/<lang>/<main>/<pos>/<kb>/<sense>'
        extracts the <sense> tag.

        Otherwise, returns None.
        """
        match = self.has_sense_pattern.match(text)
        if match is None:
            return None  # Can't find sense tag.
        return match.group(4)

    def decompose(self, text: str | Term) -> TermComponents:
        result = self._check_cache(self.comp_cache, text)
        if result is None:
            result = TermComponents(
                language=self.language,
                main=self.get_main_tag(text),
                pos=self.get_pos_tag(text),
                kb=self.get_kb_tag(text),
                sense=self.get_sense_tag(text),
            )
        return self._fill_cache(self.comp_cache, text, result)


class ConceptNet:
    """Interface to an underlying monolingual subset of a ConceptNet database."""

    def __init__(
        self, input_dir: str, term_matcher: TermMatcher, formatter: TermFormatter
    ):
        self.term_matcher = term_matcher
        self.formatter = formatter
        self.relation_to_df_map = {}
        for path, _, files in os.walk(input_dir):
            for file in files:
                df = pd.read_csv(os.path.join(path, file), header=None)
                df.columns = [self.source, self.target]
                relation_type = os.path.splitext(file)[0]
                self.relation_to_df_map[relation_type] = df

    def get_all_triplets(self) -> dict[RelationType, list[Triplet]]:
        result = {}
        for relation_type, df in self.relation_to_df_map.items():
            for data in df.to_dict(orient="records"):
                triplet = Triplet(data[self.source], relation_type, data[self.target])
                result.setdefault(relation_type, []).append(triplet)
        return result

    def query(self, query: Query) -> QueryResult:
        """Returns a set of candidate instantiations for a Query."""
        query = replace(query, source_term=query.source_term)
        if query.method is None:
            return self._factual_query(query)
        return self._anti_factual_query(query)

    def _factual_query(self, query: Query) -> QueryResult:
        df = self.relation_to_df_map[query.relation_type]
        df = self.term_matcher.find_matches(df, self.source, query.source_term)
        return set(df[self.target].tolist())

    def _anti_factual_query(self, query: Query) -> QueryResult:
        factual_blacklist = self._factual_query(query)
        if query.method == AntiFactualMethod.ALL_RELATIONS:
            hits = set()
            for relation_type, df in self.relation_to_df_map.items():
                hits.update(df[self.target].tolist())
        elif query.method == AntiFactualMethod.SAME_RELATION:
            df = self.relation_to_df_map[query.relation_type]
            hits = set(df[self.target].tolist())
        elif query.method == AntiFactualMethod.OTHER_RELATIONS:
            hits = set()
            for relation_type, df in self.relation_to_df_map.items():
                if relation_type != query.relation_type:
                    hits.update(df[self.target].tolist())
        elif query.method == AntiFactualMethod.SAME_SOURCE:
            hits = set()
            for df in self.relation_to_df_map.values():
                df = self.term_matcher.find_matches(df, self.source, query.source_term)
                hits.update(df[self.target].tolist())
        else:
            raise ValueError(f"Unsupported anti-factual method: {query.method}")
        return hits - factual_blacklist

    @staticmethod
    def _find_matches(df, column, word):
        return df[column].apply(lambda x: x == word or x.startswith(word + "/"))

    @property
    def source(self) -> str:
        """Returns the name of the source column in the underlying dataframe."""
        return "source"

    @property
    def target(self) -> str:
        """Returns the name of the target column in the underlying dataframe."""
        return "target"


class FormatMethod(Enum):
    TRIPLET = "TRIPLET"
    ACCORD = "ACCORD"
    NATLANG = "NATLANG"


class ConceptNetFormatter:
    accord_templates = {
        "AtLocation": "Suppose that [{s}] appears near [{t}]",
        "Causes": "Suppose that [{s}] causes [{t}]",
        "HasPrerequisite": "Suppose that [{s}] has prerequisite [{t}]",
        "IsA": "Suppose that [[{s}] is a type of [{t}]",
        "PartOf": "Suppose that [{s}] is a part of [{t}]",
        "UsedFor": "Suppose that [{s}] is used for [{t}]",
    }

    def __init__(self, template: str, method: FormatMethod, formatter: TermFormatter):
        self.template = template
        self.method = method
        self.formatter = formatter

    def __call__(self, triplet: Triplet) -> str:
        s = self.formatter.ensure_plain_text(triplet.source)
        t = self.formatter.ensure_plain_text(triplet.target)
        if self.method == FormatMethod.TRIPLET:
            return self.template.format(data=f"({s}, {triplet.relation}, {t})")
        elif self.method == FormatMethod.ACCORD:
            data = self.accord_templates[triplet.relation].format(s=s, t=t)
            return self.template.format(data=data)
        elif self.method == FormatMethod.NATLANG:
            # TODO: Ideally, grammatical correctness.
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported format method: {self.method}")

    def is_desired_target(self, spaced_sequence: SpacedSubsequence) -> bool:
        if self.method == FormatMethod.TRIPLET:
            return spaced_sequence.verify(",", ")")
        elif self.method == FormatMethod.ACCORD:
            is_source_or_target = spaced_sequence.verify("[", "]")
            is_source = spaced_sequence.verify("that [", "]")
            return is_source_or_target and not is_source
        elif self.method == FormatMethod.NATLANG:
            # TODO: Ideally, grammatical correctness.
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported format method: {self.method}")
