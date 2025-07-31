from random import Random
import os

from coma import command
from tqdm import tqdm
import pandas as pd
import spacy

from ....io import ConditionalPrinter, PathConfig, save_dataclass_jsonl
from ....ranking import RankerManager, RankersConfig
from ....core import (
    AccordAnalyzer,
    AnalysisResult,
    AntiFactualMethod,
    ConceptNet,
    ConceptNetFormatter,
    FormatMethod,
    LinguisticsConfig,
    Query,
    RelationType,
    TermFormatter,
    TermMatcher,
    Triplet,
)

from .base import Config, FactualAntiFactualPairing


@command(name="exp.1.preprocess")
class Experiment1Preprocess:
    def __init__(
        self,
        path: PathConfig,
        experiment1: Config,
        rankers: RankersConfig,
        linguistics: LinguisticsConfig,
    ):
        self.path = path
        self.cfg = experiment1
        self.ranker = RankerManager(rankers).get(self.cfg.ranker_id)
        self.formatter = TermFormatter(language="en")
        self.analyzer = AccordAnalyzer(
            features=linguistics.features[self.cfg.linguistics_id],
            formatter=ConceptNetFormatter(
                template=self.cfg.user_template,
                method=FormatMethod.ACCORD,
                formatter=self.formatter,
            ),
            verbose=self.cfg.verbose,
        )
        self.concept_net = ConceptNet(
            input_dir=path.concept_net_dir,
            term_matcher=TermMatcher(self.cfg.concept_net_match_method),
            formatter=self.formatter,
        )
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.stopwords = spacy.load("en_core_web_lg").Defaults.stop_words

    def run(self):
        self.print("Pre-screening all triplets...")
        all_data = self.concept_net.get_all_triplets()
        all_triplets = [t for triplets in all_data.values() for t in triplets]
        # NOTE: You really do need to do ALL triplets here to avoid later Query gaps.
        self.analyzer.validate_targets(all_triplets)
        self.print("Done.")
        for relation_type, triplets in all_data.items():
            for method in AntiFactualMethod:
                self.cfg.concept_net_query_method = method
                self._do_run(relation_type, triplets)
        self.print("Done.")

    def _do_run(self, r_type: RelationType, triplets: list[Triplet]) -> None:
        method = self.cfg.concept_net_query_method
        self.print(f"Processing {r_type=}  method={method.value}")
        Random(self.cfg.preprocess_seed).shuffle(triplets)  # Shuffles in place.
        pairings, verbosity_tracker = [], True
        for trip in tqdm(triplets) if self.cfg.verbose else triplets:
            # verbosity_tracker = self._verbosity(pairings, verbosity_tracker)
            if len(pairings) >= self.cfg.subsampling_per_relation:
                # Got enough data for this relation type.
                break
            if self._fast_skip(trip):
                continue
            query = Query(relation_type=r_type, source_term=trip.source, method=method)
            af_candidates = self._get_af_candidates(query, trip)
            if len(af_candidates) < self.cfg.preprocess_threshold:
                continue
            pairing = FactualAntiFactualPairing(
                factual_query=query,
                factual_target=trip.target,
                anti_factual_candidates=af_candidates,
            )
            pairings.append(pairing)
        out_dir = self.cfg.preprocess_dir(self.path.experiment1_dir)
        out_file = os.path.join(out_dir, f"{r_type}.jsonl")
        save_dataclass_jsonl(out_file, *pairings, ensure_ascii=False)

    def _verbosity(self, pairings: list, first_for_this_percentile: bool) -> bool:
        if len(pairings) % int(self.cfg.subsampling_per_relation / 10) == 0:
            if first_for_this_percentile:
                percent = len(pairings) / self.cfg.subsampling_per_relation * 100
                self.print(f"Found {percent}% of required triplets.", end="\r")
                return False
        return True

    def _fast_skip(self, triplet: Triplet):
        target = self.formatter.ensure_plain_text(triplet.target)
        if self._textual_skip(triplet.source, target):
            return True
        target_comp = self.formatter.decompose(triplet.target)
        if self.ranker.is_likely_low_ranked(main=target_comp):
            return True
        return False

    def _textual_skip(self, main_text: str, check_match_text: str) -> bool:
        if any(char.isdigit() for char in main_text):
            # Skip the chemical compound names, etc., that contain digits.
            return True
        if not main_text.isascii():
            # Skip any terms with at least 1 non-ASCII character, since these don't
            # get transmitted well to the vLLM server and back.
            return True
        main_text = self.formatter.ensure_plain_text(main_text)
        if main_text in self.stopwords:
            # Skipping stopwords.
            return True
        if check_match_text is not None:
            # If the two texts match, skip.
            # TODO: might want to consider stemming, though that might be too harsh.
            if main_text == check_match_text:
                return True
        return False

    def _get_af_candidates(self, query: Query, factual_triplet: Triplet) -> list:
        if not self.analyzer.is_valid_factual(factual_triplet.target):
            # This means the factual target is not linguistically self-consistent.
            return []
        candidates = []
        factual_source_text = self.formatter.ensure_plain_text(factual_triplet.source)
        for term in self.concept_net.query(query):
            term_comps = self.formatter.decompose(term)
            if self._textual_skip(term, factual_source_text):
                continue
            ranker_skip = self.ranker.is_likely_low_ranked(
                main=term_comps,
                comparison=self.formatter.decompose(factual_triplet.target),
            )
            if ranker_skip:
                continue
            valid = self.analyzer.is_valid_anti_factual(
                target=term,
                factual_target=factual_triplet.target,
                concept_net_pos=term_comps.pos,
            )
            if valid:
                candidates.append(term)
        return candidates


@command(name="exp.1.compare.conceptnet.spacy")
class Experiment1Compare:
    def __init__(
        self, path: PathConfig, experiment1: Config, linguistics: LinguisticsConfig
    ):
        self.path = path
        self.cfg = experiment1
        self.formatter = TermFormatter(language="en")
        self.analyzer = AccordAnalyzer(
            features=linguistics.features["POS_ONLY"],
            formatter=ConceptNetFormatter(
                template=self.cfg.user_template,
                method=FormatMethod.ACCORD,
                formatter=self.formatter,
            ),
            verbose=self.cfg.verbose,
        )
        self.concept_net = ConceptNet(
            input_dir=self.path.concept_net_dir,
            term_matcher=TermMatcher(self.cfg.concept_net_match_method),
            formatter=self.formatter,
        )
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.data = {}

    def run(self):
        self.print("Pre-screening all triplets...")
        all_data = self.concept_net.get_all_triplets()
        all_triplets = [t for triplets in all_data.values() for t in triplets]
        # NOTE: You really do need to do ALL triplets here to avoid later Query gaps.
        self.analyzer.validate_targets(all_triplets)
        self.print("Done.")
        for relation_type, triplets in all_data.items():
            self.print("Processing:", relation_type)
            for triplet in triplets:
                self._do_run(triplet)
        # TODO: I'd like to make a report out of this. Specifically, how many hit
        #  rate do we get per relation type?
        # TODO: But more importantly, is there a substantial decrease in hit rate if
        #  we (as now) force consistent DEP across all factual triplets for a given
        #  target? What about words that are sometimes nouns and sometimes verbs?
        # TODO: I'd also like to turn this comparison/report code into a way to save
        #  the triplet validation results to file.. probably saving the entire
        #  AnalysisResult object is not good (too much disk space and slow loading),
        #  but we can at least store all the CONSISTENCY DEPS to file! Then, load these
        #  and in linguistics, we'd have to recache just the raw, but we don't have
        #  to redo all the contextual stuff... add complications to the implementation
        #  but worth the time save? (Like 30 mins off the preprocessing... maybe not actually)
        print(pd.DataFrame(self.data))

    def _do_run(self, triplet: Triplet) -> None:
        tags = self.formatter.decompose(triplet.target)
        if tags.pos is not None:
            analysis = self.analyzer.is_valid_factual(triplet.target)
            self._assess(tags.pos, analysis, triplet.relation)

    def _assess(
        self,
        cnet_pos: str,
        analysis: AnalysisResult | None,
        relation_type: RelationType,
    ) -> None:
        relation_data = self.data.setdefault(
            relation_type, {"Nones": 0, "Successes": 0, "Total": 0}
        )
        relation_data["Total"] += 1
        if analysis is None:
            relation_data["Nones"] += 1
        elif analysis.matches_concept_net(cnet_pos):
            relation_data["Successes"] += 1
