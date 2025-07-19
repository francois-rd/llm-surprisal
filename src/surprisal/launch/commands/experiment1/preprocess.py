from random import Random
import os

from coma import command
from tqdm import tqdm
import spacy

from ....io import ConditionalPrinter, PathConfig, save_dataclass_jsonl
from ....ranking import RankerManager, RankersConfig
from ....core import (
    AntiFactualMethod,
    ConceptNet,
    Query,
    RelationType,
    Term,
    TermFormatter,
    TermMatcher,
    Triplet,
)

from .base import Config, FactualAntiFactualPairing


@command(name="exp.1.preprocess")
class Experiment1Preprocess:
    def __init__(self, path: PathConfig, experiment1: Config, rankers: RankersConfig):
        self.path = path
        self.cfg = experiment1
        self.ranker = RankerManager(rankers).get(self.cfg.ranker_id)
        self.formatter = TermFormatter(language="en")
        self.concept_net = ConceptNet(
            input_dir=path.concept_net_dir,
            term_matcher=TermMatcher(self.cfg.concept_net_match_method),
            formatter=self.formatter,
        )
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.stopwords = spacy.load("en_core_web_lg").Defaults.stop_words

    def run(self):
        for relation_type, triplets in self.concept_net.get_all_triplets().items():
            for method in AntiFactualMethod:
                self.cfg.concept_net_query_method = method
                self._do_run(relation_type, triplets)
        self.print("Done.")

    def _do_run(self, r_type: RelationType, triplets: list[Triplet]) -> None:
        pairings = []
        method = self.cfg.concept_net_query_method
        self.print(f"Processing {r_type=}  method={method.value}")
        Random(self.cfg.preprocess_seed).shuffle(triplets)  # Shuffles in place.
        first_for_this_percentile = True
        for trip in tqdm(triplets):
            if len(pairings) % int(self.cfg.subsampling_per_relation / 10) == 0:
                if first_for_this_percentile:
                    first_for_this_percentile = False
                    percent = len(pairings) / self.cfg.subsampling_per_relation * 100
                    self.print(f"Found {percent}% of required triplets.", end="\r")
            else:
                first_for_this_percentile = True
            if len(pairings) >= self.cfg.subsampling_per_relation:
                break
            if self._skip(trip.source, trip.target) or self._ranker_skip(trip.target):
                continue
            query = Query(relation_type=r_type, source_term=trip.source, method=method)
            af_candidates = [
                term
                for term in self.concept_net.query(query)
                if not self._skip(term, trip.source)
                and not self._ranker_skip(term, trip.target)
            ]
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

    def _skip(self, main_text: str, check_match_text: str | None = None) -> bool:
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
            if main_text == self.formatter.ensure_plain_text(check_match_text):
                return True
        return False

    def _ranker_skip(self, term: Term, comparison: Term | None = None) -> bool:
        return self.ranker.is_likely_low_ranked(
            term, formatter=self.formatter, comparison=comparison
        )
