from dataclasses import dataclass

import spacy
from spacy.tokens import Token
from tqdm import tqdm

from .conceptnet import ConceptNetFormatter, FormatMethod, Term, Triplet


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


LinguisticsID = str
_MISSING = object()
_INCONSISTENT = "_INCONSISTENT"


@dataclass
class LinguisticsConfig:
    features: dict[LinguisticsID, LinguisticFeatures]


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
    raw_term: Term | None  # Only needed for caching and doesn't affect consistency.
    idx: int  # Only needed for caching and doesn't affect consistency.

    @classmethod
    def from_token(
        cls, token: Token, features: LinguisticFeatures, raw_term: Term | None
    ) -> "AnalysisResult":
        return AnalysisResult(
            text=token.text,
            pos=token.pos_ if features.pos else None,
            tag=token.tag_ if features.tag else None,
            dep=token.dep_ if features.dep else None,
            morph=token.morph.to_dict() if features.full_morph else None,
            lemma=token.lemma_ if features.lemma else None,
            ner_type=token.ent_type_ if features.ner_type else None,
            sentiment=token.sentiment if features.sentiment else None,
            children=[cls.from_token(c, features, None) for c in token.children],
            raw_term=raw_term,
            idx=token.i,
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

    def matches_concept_net(self, cnet_pos: str | None) -> bool:
        if cnet_pos is None:
            return self.pos is None
        if cnet_pos == "n":
            return self.pos in ["NOUN", "PROPN"]
        if cnet_pos == "v":
            return self.pos == "VERB"
        if cnet_pos == "a":
            return self.pos == "ADJ"
        if cnet_pos == "r":
            return self.pos == "ADV"
        return False


class AccordAnalyzer:
    def __init__(
        self,
        features: LinguisticFeatures,
        formatter: ConceptNetFormatter,
        verbose: bool,
    ):
        self.features = features
        self.formatter = self._validate_formatter(formatter)
        self.verbose = verbose
        self.nlp = spacy.load("en_core_web_md")
        self.dep_caches: dict[Term, dict[Triplet, str | None]] = {}
        self.analysis_cache: dict[Term, AnalysisResult] = {}
        self.consistency_cache: dict[tuple[Term, Term], bool] = {}

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

    def validate_targets(self, factual_triplets: list[Triplet]) -> None:
        """
        TODO: A target is only valid if it's contextual DEP tag is consistent across all
         triplets in which it occurs (as well as being self-consistent across the same).
        """
        for triplet in tqdm(factual_triplets) if self.verbose else factual_triplets:
            self._cache_target(triplet)
        for target, raw_analysis in self.analysis_cache.items():
            deps = set(self.dep_caches[target].values())
            if len(deps) == 1:
                raw_analysis.dep = deps.pop()
            else:
                raw_analysis.dep = _INCONSISTENT

    def _cache_target(self, triplet: Triplet) -> None:
        # Get the raw analysis, either from cache or by computing.
        raw_analysis = self.analysis_cache.get(triplet.target, _MISSING)
        if raw_analysis is _MISSING:
            raw_analysis = self._analyze_raw(triplet.target)
            self.analysis_cache[triplet.target] = raw_analysis

        # Ensure the dependency sub-cache is lacking this particular triplet.
        dep_cache = self.dep_caches.setdefault(triplet.target, {})
        if dep_cache.get(triplet, _MISSING) is not _MISSING:
            raise ValueError(f"Triplet is not unique in ConceptNet: {triplet}")

        # Record the dependency tag if this triplet is self-consistent. Else None.
        contextual_analysis = self._analyze_contextual(raw_analysis.idx, triplet)
        consistent = raw_analysis.is_consistent_with(contextual_analysis, self.features)
        dep_cache[triplet] = contextual_analysis.dep if consistent else _INCONSISTENT

    def is_valid_factual(self, target: Term) -> bool:
        """Raises KeyError if target has not been preprocessed."""
        return self.analysis_cache[target].dep != _INCONSISTENT

    def is_valid_anti_factual(
        self, target: Term, factual_target: Term, concept_net_pos: str
    ) -> bool:
        """
        Returns True only if this anti-factual triplet's analysis is both consistent
        with the analysis of the factual target and the ConceptNet components' POS
        (if LinguisticFeatures permits). Anti-factual contextual consistency is not
        checked in order to leverage caching; however, factual contextual consistency
        has been previously validated.

        Raises KeyError if either target has not been (factually) validated.
        """
        # Retrieve the consistency result, if any.
        value = self.consistency_cache.get((target, factual_target), None)
        if value is not None:
            return value

        analysis = self.analysis_cache[target]
        factual_analysis = self.analysis_cache[factual_target]

        if _INCONSISTENT in [analysis.dep, factual_analysis.dep]:
            # If either analysis is not (factually) self-consistent, then it's no good.
            value = False
        elif self.features.pos and not analysis.matches_concept_net(concept_net_pos):
            # If the target analysis is not consistent with ConceptNet, then no good.
            value = False
        else:
            # Determine consistency between target and factual.
            value = analysis.is_consistent_with(
                factual_analysis,
                features=self.features,
                include_top_level_dep=True,
                exclude_text=True,
            )

        # Cache and return.
        self.consistency_cache[(target, factual_target)] = value
        return value

    def _analyze_raw(self, target: Term) -> AnalysisResult:
        for tok in self.nlp(self.formatter.formatter.ensure_plain_text(target)):
            if tok.dep_ == "ROOT":  # "ROOT" is baked into spacy.
                return AnalysisResult.from_token(tok, self.features, target)
        raise ValueError(f"Spacy cannot determine a ROOT for: {target}")

    def _analyze_contextual(self, target_idx: int, triplet: Triplet) -> AnalysisResult:
        context = self.formatter(triplet)
        doc = self.nlp(context)
        last_bracket_index = None
        for token in doc:
            if token.text.strip() == "[":
                last_bracket_index = token.i
        if last_bracket_index is None:
            raise ValueError(f"Couldn't find '[' token in: {context}")
        for token in doc:
            if token.i - last_bracket_index - 1 == target_idx:
                return AnalysisResult.from_token(token, self.features, triplet.target)
