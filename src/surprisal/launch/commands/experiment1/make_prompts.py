from coma import command

from ....core import AntiFactualMethod, Label, Term, TermFormatter, Triplet
from ....prompting import IDGenerator, Message, MessageType, PromptData, PromptType
from ....ranking import RankerManager, RankersConfig
from ....io import (
    ConditionalPrinter,
    PathConfig,
    load_dataclass_jsonl,
    save_dataclass_jsonl,
    walk_files,
)

from .base import Config, FactualAntiFactualPairing


@command(name="exp.1.make.prompts")
class Experiment1MakePrompts:
    def __init__(self, path: PathConfig, experiment1: Config, rankers: RankersConfig):
        self.path = path
        self.cfg = experiment1
        self.ranker = RankerManager(rankers).get(self.cfg.ranker_id)
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.id_generator = IDGenerator()
        self.formatter = TermFormatter(language="en")

    def run(self):
        for method in AntiFactualMethod:
            self.cfg.concept_net_query_method = method
            self.print("Processing:", method.value)
            prompts = self._make_prompts(self._load_data())
            file_name = self.cfg.prompts_file(self.path.experiment1_dir)
            save_dataclass_jsonl(file_name, *prompts, ensure_ascii=False)
        self.print("Done.")

    def _load_data(self) -> list[tuple[FactualAntiFactualPairing, Term]]:
        # Load raw preprocessing results, rank them, then keep only the first.
        ranked_data = []
        self.print("    Loading data", end="")
        for walk in walk_files(self.cfg.preprocess_dir(self.path.experiment1_dir)):
            for pairing in load_dataclass_jsonl(walk.path, t=FactualAntiFactualPairing):
                top_ranked_terms = self.ranker(
                    result=set(pairing.anti_factual_candidates),
                    formatter=self.formatter,
                    factual_target=pairing.factual_target,
                )
                if len(top_ranked_terms) == 0:
                    continue
                top_ranked_term, metric = top_ranked_terms[0]
                if self.ranker.is_worst_outcome(metric):
                    print(f"Skipping {pairing} because no candidate is good.")
                    continue
                ranked_data.append((pairing, top_ranked_term))
            self.print(".", flush=True, end="")
        self.print("\n    Done.")
        return ranked_data

    def _make_prompts(
        self, ranked_data: list[tuple[FactualAntiFactualPairing, Term]]
    ) -> list[PromptData]:
        self.print("    Making prompts...")
        prompts = []
        for pairing, term in ranked_data:
            self.id_generator.next_group_id()  # Reset for each new group.
            prompts.extend(
                [
                    self._do_make_prompt(pairing, term, factual, label)
                    for label in ["TRUE", "FALSE"]
                    for factual in [True, False]
                ]
            )
        self.print("    Done.")
        return prompts

    def _do_make_prompt(
        self,
        pairing: FactualAntiFactualPairing,
        top_ranked_term: Term,
        factual: bool,
        label: Label,
    ) -> PromptData:
        return PromptData(
            messages=[
                Message(MessageType.SYSTEM, "PLACEHOLDER"),
                Message(MessageType.USER, "PLACEHOLDER"),
                Message(MessageType.ASSISTANT, label),
                Message(MessageType.USER, "CAPSTONE"),
            ],
            prompt_type=PromptType.F if factual else PromptType.AF,
            label=label,
            prompt_id=self.id_generator.next_prompt_id(),
            group_id=self.id_generator.next_group_id(no_increment=True),
            additional_data=dict(
                triplet=Triplet(
                    source=pairing.factual_query.source_term,
                    relation=pairing.factual_query.relation_type,
                    target=pairing.factual_target if factual else top_ranked_term,
                )
            ),
        )
