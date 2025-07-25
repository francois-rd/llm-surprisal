from dataclasses import dataclass

import spacy
from spacy.tokens import Token

from .conceptnet import ConceptNetFormatter, FormatMethod, Triplet


@dataclass
class LinguisticFeatures:
    pos: bool = True
    tag: bool = True
    dep: bool = True
    full_morph: bool = True
    lemma: bool = False
    ner_type: bool = False
    sentiment: bool = False
    sentiment_tol: float = 0.001
    # TODO: Could we add semantic similarity? Wrt to factual, I guess...
    #  how to handle multi-token similarity, though? And is spacy really the
    #  right modelling tool for that? Even for sentiment, it's not that great.
    #  For now, no semantics. If we want, we can try adding it later (either spacy
    #  or an alternative/NN-based model.. which is maybe what spacy has anyways).
    children: bool = False


@dataclass
class AnalysisResult:
    text: str
    pos: str | None
    tag: str | None
    dep: str | None
    morph: dict[str, str] | None
    lemma: str | None
    ner_type: str | None
    sentiment: float | None
    children: list["AnalysisResult"]

    @classmethod
    def from_token(cls, token: Token, features: LinguisticFeatures) -> "AnalysisResult":
        return AnalysisResult(
            text=token.text,
            pos=token.pos_ if features.pos else None,
            tag=token.tag_ if features.tag else None,
            dep=token.dep_ if features.dep else None,
            morph=token.morph.to_dict() if features.full_morph else None,
            lemma=token.lemma_ if features.lemma else None,
            ner_type=token.ent_type_ if features.ner_type else None,
            sentiment=token.sentiment if features.sentiment else None,
            children=[cls.from_token(c, features) for c in token.children],
        )

    def is_consistent_with(
        self,
        other: "AnalysisResult",
        features: LinguisticFeatures,
        include_top_level_dep: bool = False,
        exclude_text: bool = False,
    ) -> bool:
        # If either or both are None, compare None-ness. Else, compare floats.
        if self.sentiment is None or other.sentiment is None:
            sentiment = self.sentiment == other.sentiment
        else:
            sentiment = abs(self.sentiment - other.sentiment) < features.sentiment_tol

        # Compare every field other than children.
        consistent = (
            sentiment
            and (True if exclude_text else self.text == other.text)
            and self.pos == other.pos
            and self.tag == other.tag
            and (self.dep == other.dep if include_top_level_dep else True)
            and self.morph == other.morph
            and self.lemma == other.lemma
            and self.ner_type == other.ner_type
        )

        # If consistent at top-level, recursively check matching children, if desired.
        if consistent:
            if features.children:
                return self._process_children(other, features, exclude_text)
            return True
        return False

    def _process_children(
        self,
        other: "AnalysisResult",
        features: LinguisticFeatures,
        exclude_text: bool,
        also_swap: bool = True,
    ) -> bool:
        for c in self.children:
            has_match = False
            for o in other.children:
                # If the number of grand-children don't match, the children won't.
                if len(c.children) != len(o.children):
                    continue
                # If they do match in quantity, recursively check all properties.
                if c.is_consistent_with(o, features, exclude_text=exclude_text):
                    has_match = True
            if not has_match:
                return False
        if also_swap:
            return other._process_children(
                self, features, exclude_text=exclude_text, also_swap=False
            )
        return True


class AccordAnalyzer:
    def __init__(self, features: LinguisticFeatures, formatter: ConceptNetFormatter):
        self.features = features
        self.formatter = self._validate_formatter(formatter)
        self.nlp = spacy.load("en_core_web_md")

    @staticmethod
    def _validate_formatter(formatter: ConceptNetFormatter) -> ConceptNetFormatter:
        if formatter.method == FormatMethod.TRIPLET:
            raise ValueError(f"Linguistic analysis in TRIPLET format is unreliable.")
        elif formatter.method == FormatMethod.ACCORD:
            return formatter
        elif formatter.method == FormatMethod.NATLANG:
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported format method: {formatter.method}")

    def analyze_factual(self, triplet: Triplet) -> AnalysisResult | None:
        """Returns the analysis of this factual triplet, or None if inconsistent."""
        return self._analyze(triplet)

    def is_valid_anti_factual(
        self, triplet: Triplet, factual_analysis: AnalysisResult
    ) -> bool:
        """
        Returns True only if this anti-factual triplet's analysis is
        both self-consistent and consistent with the factual analysis.
        """
        analysis = self._analyze(triplet)
        if analysis is None:
            return False
        return analysis.is_consistent_with(
            factual_analysis,
            features=self.features,
            include_top_level_dep=True,
            exclude_text=True,
        )

    def _analyze(self, triplet: Triplet) -> AnalysisResult | None:
        # Analyze just the target term (as plain text) to understand its raw features.
        raw_analysis, idx = None, None
        target = self.formatter.formatter.ensure_plain_text(triplet.target)
        for token in self.nlp(target):
            if token.dep_ == "ROOT":  # "ROOT" is baked into spacy.
                raw_analysis = AnalysisResult.from_token(token, self.features)
                idx = token.i
        if raw_analysis is None or idx is None:
            raise ValueError(f"Spacy cannot determine a ROOT for: {triplet.target}")

        # Analyze the target term in the context of the full ACCORD-style statement.
        contextual_analysis = self._analyze_contextual(idx, self.formatter(triplet))

        # Compare the two according to desired linguistic features.
        if raw_analysis.is_consistent_with(contextual_analysis, self.features):
            return contextual_analysis
        return None

    def _analyze_contextual(self, target_index: int, context: str) -> AnalysisResult:
        doc = self.nlp(context)
        last_bracket_index = None
        for token in doc:
            if token.text.strip() == "[":
                last_bracket_index = token.i
        if last_bracket_index is None:
            raise ValueError(f"Couldn't find '[' token in: {context}")
        for token in doc:
            if token.i - last_bracket_index - 1 == target_index:
                return AnalysisResult.from_token(token, self.features)
