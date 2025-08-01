from dataclasses import dataclass, field
import os

from ....parsing import ParserID
from ....ranking import RankerID
from ....llms import Nickname
from ....core import (
    AggregatorOption,
    AntiFactualMethod,
    FormatMethod,
    LinguisticsID,
    Query,
    Term,
    TermMatchMethod,
)


@dataclass
class FactualAntiFactualPairing:
    factual_query: Query
    factual_target: Term
    anti_factual_candidates: list[Term]


@dataclass
class Config:
    ranker_id: RankerID = "random"
    parser_id: ParserID = "no_parser"
    linguistics_id: LinguisticsID = "no_ling"
    concept_net_query_method: AntiFactualMethod = AntiFactualMethod.SAME_RELATION
    concept_net_match_method: TermMatchMethod = TermMatchMethod.IDENTICAL
    data_format_method: FormatMethod = FormatMethod.TRIPLET
    subsampling_per_relation: int = 200
    prompt_batch_size: int = 10
    checkpoint_frequency: float = 5 * 60  # Save every 5 minutes.
    chosen_only_logprob: bool = True
    trim_inference_logprobs: bool = True
    verbose: bool = False
    system_prompt_id: str = ""
    system_prompt: dict[str, str] = field(
        default_factory=lambda: {m.value: "" for m in FormatMethod}
    )
    user_template: str = "Is the following relationship true or false?\n\n{data}\n"
    user_template_indicator: str = "Is the following relationship true or false?"
    preprocess_seed: int = 42
    preprocess_threshold: int = 5  # Minimum number of AF candidates to keep a pairing.
    analysis_llms: list[Nickname] = field(default_factory=list)
    aggregators: list[AggregatorOption] = field(
        default_factory=lambda: [
            AggregatorOption.FIRST,
            AggregatorOption.SUM,
            AggregatorOption.MIN,
        ]
    )
    flip_logprobs: bool = True
    cartesian_cross_plot: bool = False

    def _build_id(
        self,
        ranker: bool = True,
        parser: bool = True,
        linguistics: bool = True,
        query_method: bool = True,
        match_method: bool = True,
        format_method: bool = True,
        subsampling: bool = True,
        system_prompt: bool = True,
        preprocess_seed: bool = True,
        preprocess_threshold: bool = True,
    ):
        items = []
        if ranker:
            items.append(self.ranker_id)
        if parser:
            items.append(self.parser_id)
        if linguistics:
            items.append(self.linguistics_id)
        if query_method:
            items.append(self.concept_net_query_method.value)
        if match_method:
            items.append(self.concept_net_match_method.value)
        if format_method:
            items.append(self.data_format_method.value)
        if subsampling:
            items.append(str(self.subsampling_per_relation))
        if system_prompt:
            items.append(self.system_prompt_id)
        if preprocess_seed:
            items.append(str(self.preprocess_seed))
        if preprocess_threshold:
            items.append(str(self.preprocess_threshold))
        return "-".join(items)

    def preprocess_dir(self, root: str) -> str:
        b_id = self._build_id(parser=False, format_method=False, system_prompt=False)
        return str(os.path.join(root, "preprocess", b_id))

    def prompts_file(self, root: str) -> str:
        b_id = self._build_id(parser=False, format_method=False, system_prompt=False)
        return str(os.path.join(root, "prompts", b_id)) + ".jsonl"

    def llm_output_dir(self, root: str) -> str:
        return str(os.path.join(root, "output", self._build_id()))

    def analysis_dir(self, root: str) -> str:
        return str(os.path.join(root, "analysis", self._build_id()))

    def plots_dir(self, root: str) -> str:
        return str(os.path.join(root, "plots", self._build_id()))
