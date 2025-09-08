from typing import Iterable, Union
from dataclasses import dataclass
from datetime import datetime
import os

from coma import command

from ....accord import (
    AccordID,
    AccordLabel,
    AccordMetaData,
    AccordMetrics,
    AccordStatement,
    AccordStatementID,
    AccordStatementKey,
    AccordStatementSurfacer,
    AccordTerm,
    PairedAccordMetrics,
)
from ....inference import Inference
from ....llms import Nickname
from ....prompting import PromptType
from ....core import (
    AggregatorOption,
    ConceptNetFormatter,
    FormatMethod,
    Logprobs,
    SpacedSubsequence,
)
from ....io import (
    ConditionalPrinter,
    PathConfig,
    init_logger,
    load_dataclass_jsonl,
    save_dataclass_jsonl,
    walk_files,
)

from .base import AccordLoader, AccordSubset, Config


SourceTargetPairs = tuple[list[float], list[float]]
LabelTermPairs = tuple[list[float], list[float]]


@dataclass
class _CurrentData:
    llm: Nickname
    inference: Inference
    meta_data: AccordMetaData
    logprobs: Logprobs
    statement_tag_indices: tuple[int, int] | None = None
    question_indices: tuple[int, int] | None = None
    answer_tag_indices: tuple[int, int] | None = None
    logprob_of_instance: list[float] | None = None
    logprob_of_statements: dict[AccordStatementKey, SourceTargetPairs] | None = None
    logprob_of_question: list[float] | None = None
    logprob_of_answer_choices: dict[AccordLabel, LabelTermPairs] | None = None
    logprob_of_forced_answer: list[float] | None = None


@dataclass
class LogprobDataclass:
    accord_group_id: str
    factuality: str
    accord_label: AccordLabel
    csqa_label: AccordLabel
    subset: AccordSubset
    metrics: AccordMetrics
    paired_metrics: PairedAccordMetrics | None = None


class LogprobData:
    def __init__(
        self,
        current_data: _CurrentData,
        subset: AccordSubset,
        flip_logprobs: bool,
        aggregators: list[AggregatorOption],
    ):
        prompt_data = current_data.inference.prompt_data
        self.llm: Nickname = current_data.llm

        # TODO: Prompt type is accurate, but standard group IDs are not... There is a
        #  bug in make_prompts where group_id is based on the instance, but there are
        #  always 2 paired instances if you consider the ACCORD ID (one F and one AF).
        #    Option 1: fix make_prompts and redo analysis.
        #    Option 2: fix it here in post: groupID becomes Accord ID less last letter
        self.accord_group_id: str = current_data.meta_data.id[:-2]

        self.factuality: str = self._get_factuality(prompt_data.prompt_type)
        self.forced_label: AccordLabel = prompt_data.label
        self.accord_label: AccordLabel = current_data.meta_data.label
        self.csqa_label: AccordLabel = prompt_data.additional_data["csqa_label"]
        self.subset: AccordSubset = subset
        self.aggregators = aggregators
        self.flip = flip_logprobs

        # These two are dict[aggregator, list[float] | None] where each float is the
        # aggregation of all tokens making up the logprobs for a source/target and
        # the list is the set of all such terms. These eventually need some form of
        # macro aggregation (e.g., averaging all individual term logprobs) which is
        # handled by the AccordMetrics. A value of None occurs for the BASELINE subset.
        self.source_lps, self.target_lps = self._aggregate_statements(current_data)

        # These two are dict[AccordStatementKey, dict[aggregator, float] | None]
        # where each float is the aggregation of all tokens making up the logprobs for
        # a source/target tracked by which statement they belong to. PairedAccordMetrics
        # subtracts the tracked values (F from AF) and then aggregates the result. A
        # value of None occurs for the BASELINE subset.
        self.tracked_source_lps, self.tracked_target_lps = (
            self._aggregate_tracked_statements(current_data)
        )

        # These two are dict[aggregator, float] where each float is the aggregation of
        # all tokens making up the logprobs for the question or the entire instance.
        self.question_lps = {}
        self._aggregate(self.question_lps, current_data.logprob_of_question)
        self.instance_lps = {}
        self._aggregate(self.instance_lps, current_data.logprob_of_instance)

        # These three are dict[accord_label_A_to_E, dict[aggregator, float]] where
        # each float is the aggregation of all tokens making up the logprobs of the
        # indicated object, and this is done for each label and each aggregator.
        self.label_lps, self.choice_lps = self._aggregate_choices(current_data)
        self.forced_lps = self._aggregate_forced_labels(current_data)

    @staticmethod
    def _get_factuality(prompt_type: PromptType) -> str:
        return "Factual" if prompt_type == PromptType.F else "Anti-Factual"

    def _aggregate_statements(self, current_data: _CurrentData) -> tuple[dict, dict]:
        if current_data.logprob_of_statements is None:
            agg_to_none = {a: None for a in self.aggregators}
            return agg_to_none, agg_to_none
        all_source_lps, all_target_lps = {}, {}
        for source_lps, target_lps in current_data.logprob_of_statements.values():
            self._aggregate(all_source_lps, source_lps, append=True)
            self._aggregate(all_target_lps, target_lps, append=True)
        return all_source_lps, all_target_lps

    def _aggregate_tracked_statements(
        self, current_data: _CurrentData
    ) -> tuple[dict, dict]:
        if current_data.logprob_of_statements is None:
            agg_to_none = {a: None for a in self.aggregators}
            return agg_to_none, agg_to_none
        all_source_lps, all_target_lps = {}, {}
        for key, (source, target) in current_data.logprob_of_statements.items():
            self._aggregate(all_source_lps.setdefault(key, {}), source)
            self._aggregate(all_target_lps.setdefault(key, {}), target)
        return all_source_lps, all_target_lps

    def _aggregate_choices(self, current_data: _CurrentData) -> tuple[dict, dict]:
        all_label_lps, all_choice_lps = {}, {}
        lps = current_data.logprob_of_answer_choices
        for label, (label_lp, choice_lp) in lps.items():
            self._aggregate(all_label_lps.setdefault(label, {}), label_lp)
            self._aggregate(all_choice_lps.setdefault(label, {}), choice_lp)
        return all_label_lps, all_choice_lps

    def _aggregate_forced_labels(self, current_data: _CurrentData) -> dict:
        all_forced_label_lps, lp = {}, current_data.logprob_of_forced_answer
        self._aggregate(all_forced_label_lps.setdefault(self.forced_label, {}), lp)
        return all_forced_label_lps

    def _aggregate(self, data: dict, logprobs: list[float], append: bool = False):
        for a in self.aggregators:
            if append:
                data.setdefault(a, []).append(a.aggregate(logprobs, self.flip))
            else:
                data[a] = a.aggregate(logprobs, self.flip)

    def add_forced_label(self, current_data: _CurrentData) -> None:
        forced_label = current_data.inference.prompt_data.label
        lp = current_data.logprob_of_forced_answer
        self._aggregate(self.forced_lps.setdefault(forced_label, {}), lp)

    def is_complete(self, count: int) -> bool:
        """Returns whether each dict with multiple AccordLabels has 'count' of them."""
        partial = len(self.label_lps) == count and len(self.choice_lps) == count
        return partial and len(self.forced_lps) == count

    def crystalize(
        self, partner: Union["LogprobData", None], start_label: AccordLabel
    ) -> list[LogprobDataclass]:
        if partner is None:
            return [self._crystalize(None, start_label)]
        if self.factuality == "Factual":
            if partner.factuality != "Anti-Factual":
                raise ValueError(
                    f"Self and Partner have the same factuality!\n"
                    f"Self: {self}\n"
                    f"Partner: {partner}"
                )
            f, af = self, partner
        else:
            if partner.factuality != "Factual":
                raise ValueError(
                    f"Self and Partner have the same factuality!\n"
                    f"Self: {self}\n"
                    f"Partner: {partner}"
                )
            f, af = partner, self
        paired_metrics = PairedAccordMetrics.from_data(
            factual_source_lps=f.tracked_source_lps,
            factual_target_lps=f.tracked_target_lps,
            anti_factual_source_lps=af.tracked_source_lps,
            anti_factual_target_lps=af.tracked_target_lps,
        )
        return [
            f._crystalize(paired_metrics, start_label),
            af._crystalize(None, start_label),
        ]

    def _crystalize(
        self, paired_metrics: PairedAccordMetrics | None, start_label: AccordLabel
    ) -> LogprobDataclass:
        return LogprobDataclass(
            accord_group_id=self.accord_group_id,
            factuality=self.factuality,
            accord_label=self.accord_label,
            csqa_label=self.csqa_label,
            subset=self.subset,
            metrics=AccordMetrics.from_data(
                source_lps=self.source_lps,
                target_lps=self.target_lps,
                question_lps=self.question_lps,
                instance_lps=self.instance_lps,
                forced_lps=self.forced_lps,
                label_lps=self.label_lps,
                choice_lps=self.choice_lps,
                accord_label=self.accord_label,
                csqa_label=self.csqa_label,
                start_label=start_label,
            ),
            paired_metrics=paired_metrics,
        )


class SubsetLoader:
    def __init__(
        self,
        accord_loader: AccordLoader,
        accord_subset: AccordSubset,
        pre_analysis_dir: str,
        flip_logprobs: bool,
        aggregators: list[AggregatorOption],
    ):
        self.accord_loader = accord_loader
        self.accord_subset = accord_subset
        self.pre_analysis_dir = pre_analysis_dir
        self.flip = flip_logprobs
        self.aggregators = aggregators
        self.data = self.logger = None
        self.current_data: _CurrentData | None = None
        self.meta_datas = {i.meta_data.id: i.meta_data for i in accord_loader.load()}
        self.statement_surfacer: AccordStatementSurfacer | None = getattr(
            self.accord_loader.surfacer.ordering_surfacer, "surfacer", None
        )
        if self.statement_surfacer is not None:
            # TODO: Having "- " prefix is causing problems with Logprobs.indices_of().
            #  Ideally, we'd fix that bug, but this hack works for now.
            self.statement_surfacer.prefix = ""
        self.formatter = ConceptNetFormatter(
            template="", method=FormatMethod.ACCORD, formatter=...
        )

    def load(self, inference_path: str, llm: Nickname) -> dict[AccordID, LogprobData]:
        self.data, self.logger = {}, None
        for inference in load_dataclass_jsonl(inference_path, t=Inference):
            if inference.error_message is not None:
                continue
            self.current_data = _CurrentData(
                llm=llm,
                inference=inference,
                meta_data=self._get_meta_data(inference),
                logprobs=Logprobs.from_dict(inference.derived_data["prompt_logprobs"]),
            )
            success = (
                self._fill_statement_tag_indices()
                and self._fill_question_indices()
                and self._fill_answer_tag_indices()
                and self._fill_logprob_of_instance()
                and self._fill_logprob_of_statements()
                and self._fill_logprob_of_question()
                and self._fill_logprob_of_answer_choices()
                and self._fill_logprob_of_forced_answer()
            )
            if not success:
                continue
            id_ = self.current_data.meta_data.id
            if self._first_in_group():
                self.data[id_] = self._make_logprob_data(self.current_data)
            else:
                self.data[id_].add_forced_label(self.current_data)
        return self.data

    def _get_meta_data(self, inference: Inference) -> AccordMetaData:
        return self.meta_datas[inference.prompt_data.additional_data["accord_id"]]

    def _is_baseline(self) -> bool:
        return self.accord_subset == AccordSubset.BASELINE

    def _first_in_group(self) -> bool:
        return self.current_data.meta_data.id not in self.data

    def _make_logprob_data(self, data: _CurrentData) -> LogprobData:
        return LogprobData(data, self.accord_subset, self.flip, self.aggregators)

    def _get_sequence(
        self,
        sequences: Iterable[SpacedSubsequence],
        fail_main_text: str,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> SpacedSubsequence | None:
        spaced_sequences = list(sequences)
        if len(spaced_sequences) != 1:
            self._log_failure(fail_main_text, len(spaced_sequences), start_idx, end_idx)
            return None
        return spaced_sequences[0]

    def _fill_statement_tag_indices(self) -> bool:
        if self._is_baseline():
            self.current_data.statement_tag_indices = [-1, -1]
            return True
        statement_tag = self.accord_loader.surfacer.ordering_surfacer.prefix.strip()
        sequences = self.current_data.logprobs.indices_of(statement_tag)
        seq = self._get_sequence(sequences, f"Statement tag '{statement_tag}'")
        if seq is not None:
            statement_tag_indices = min(seq.indices), max(seq.indices) + 1
            self.current_data.statement_tag_indices = statement_tag_indices
        return self.current_data.statement_tag_indices is not None

    def _fill_question_indices(self) -> bool:
        question = self.current_data.meta_data.question
        sequences = self.current_data.logprobs.indices_of(question)
        seq = self._get_sequence(sequences, f"Question '{question}'")
        if seq is not None:
            self.current_data.question_indices = min(seq.indices), max(seq.indices)
        return self.current_data.question_indices is not None

    def _fill_answer_tag_indices(self) -> bool:
        answer_tag = self.accord_loader.surfacer.suffix_surfacer.prefix.strip()
        sequences = self.current_data.logprobs.indices_of(answer_tag)
        seq = self._get_sequence(sequences, f"Answer tag '{answer_tag}'")
        if seq is not None:
            answer_tag_indices = min(seq.indices), max(seq.indices) + 1
            self.current_data.answer_tag_indices = answer_tag_indices
        return self.current_data.answer_tag_indices is not None

    def _fill_logprob_of_instance(self) -> bool:
        c = self.current_data
        if not self._first_in_group():
            return True
        start_idx = c.statement_tag_indices[1]
        if self._is_baseline() or self.accord_loader.invert:
            start_idx = c.question_indices[0]
        end_idx = c.answer_tag_indices[0]
        c.logprob_of_instance = c.logprobs.to_chosen_logprobs(start_idx, end_idx)
        return True

    def _fill_logprob_of_statements(self) -> bool:
        if not self._first_in_group():
            return True
        if self._is_baseline():
            return True
        results = {}
        for lab, tree_data in self.current_data.meta_data.statements.items():
            for s_id, s in tree_data.items():
                extraction = self._do_extract_statement_term_logprobs(s, (lab, s_id))
                if extraction is None:
                    return False
                results[s.to_key()] = extraction
        self.current_data.logprob_of_statements = results
        return True

    def _do_extract_statement_term_logprobs(
        self,
        statement: AccordStatement,
        ordering: tuple[AccordLabel, AccordStatementID],
    ) -> SourceTargetPairs | None:
        # Find the exact boundaries of this particular statement.
        text = self.statement_surfacer(self.current_data.meta_data, ordering=ordering)
        start_idx = self.current_data.statement_tag_indices[1]
        end_idx = self.current_data.question_indices[0]
        if self.accord_loader.invert:
            end_idx = self.current_data.answer_tag_indices[0]
        sequences = self.current_data.logprobs.indices_of(text, start_idx, end_idx)
        seq = self._get_sequence(sequences, f"Statement '{text}'", start_idx, end_idx)
        if seq is None:
            return None

        # Use those tight boundaries to find the source and target terms.
        args = min(seq.indices), max(seq.indices), text
        source_logprobs = self._extract_term_logprobs(statement.source_term, *args)
        if source_logprobs is None:
            return None
        target_logprobs = self._extract_term_logprobs(statement.target_term, *args)
        return None if target_logprobs is None else source_logprobs, target_logprobs

    def _extract_term_logprobs(
        self, term: AccordTerm, start_idx: int, end_idx: int, statement_text: str
    ) -> list[float] | None:
        seqs = list(self.current_data.logprobs.indices_of(term, start_idx, end_idx))
        if len(seqs) == 1:
            return seqs[0].to_chosen_logprobs()
        desired_seqs = [s for s in seqs if self.formatter.is_desired_target(s)]
        if len(desired_seqs) == 1:
            return desired_seqs[0].to_chosen_logprobs()
        text = f"Term '{term}' narrowed to search in statement '{statement_text}'"
        self._log_failure(text, len(desired_seqs), start_idx, end_idx)
        return None

    def _fill_logprob_of_question(self) -> bool:
        if not self._first_in_group():
            return True
        c = self.current_data
        c.logprob_of_question = c.logprobs.to_chosen_logprobs(*c.question_indices)
        return True

    def _fill_logprob_of_answer_choices(self) -> bool:
        if not self._first_in_group():
            return True
        results = {}
        start_idx = self.current_data.question_indices[1]
        end_idx = self.current_data.statement_tag_indices[0]
        if self._is_baseline() or not self.accord_loader.invert:
            end_idx = self.current_data.answer_tag_indices[0]
        for label, term in self.current_data.meta_data.answer_choices.items():
            label_ss = self.current_data.logprobs.indices_of(label, start_idx, end_idx)
            label_s = self._get_sequence(
                label_ss, f"Label '{label}'", start_idx, end_idx
            )
            if label_s is None:
                return False
            term_ss = self.current_data.logprobs.indices_of(term, start_idx, end_idx)
            term_s = self._get_sequence(
                term_ss, f"Answer Choice Term '{term}'", start_idx, end_idx
            )
            if term_s is None:
                return False
            results[label] = label_s.to_chosen_logprobs(), term_s.to_chosen_logprobs()
        self.current_data.logprob_of_answer_choices = results
        return True

    def _fill_logprob_of_forced_answer(self) -> bool:
        start_idx = self.current_data.answer_tag_indices[1]
        forced_label = self.current_data.inference.prompt_data.label
        sequences = self.current_data.logprobs.indices_of(forced_label, start_idx)
        seq = self._get_sequence(sequences, f"Forced Label '{forced_label}'", start_idx)
        if seq is not None:
            self.current_data.logprob_of_forced_answer = seq.to_chosen_logprobs()
        return self.current_data.logprob_of_forced_answer is not None

    def _log_failure(
        self, main_text: str, num_occurrences: int, start_idx: int, end_idx: int | None
    ) -> None:
        llm = self.current_data.llm
        group_id = self.current_data.meta_data.id
        logprobs = self.current_data.logprobs
        logger_name = f"exp.2.analysis.{llm}.{self.accord_subset}"
        log_file = f"{self.accord_subset.value}-{llm}-{datetime.now().isoformat()}.log"
        log_path = os.path.join(self.pre_analysis_dir, log_file)
        self.logger = self.logger or init_logger(logger_name, log_path)
        no_line_breaks = logprobs.to_text(start_idx, end_idx).replace("\n", "<NL>")
        self.logger.info(
            f"GroupID={group_id}: {main_text} has "
            f"{num_occurrences} occurrences in '{no_line_breaks}'."
        )


@command(name="exp.2.pre.analysis")
class Experiment2PreAnalysis:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path, self.cfg, self.loaders = path, experiment2, {}
        self.out_dir = self.cfg.pre_analysis_dir(self.path.experiment2_dir)
        for subset in AccordSubset:
            self.loaders[subset] = SubsetLoader(
                accord_loader=AccordLoader(subset, self.path, self.cfg),
                accord_subset=subset,
                pre_analysis_dir=self.out_dir,
                flip_logprobs=self.cfg.flip_logprobs,
                aggregators=self.cfg.aggregators,
            )
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.data_by_llm = {}

    def run(self):
        for subset in AccordSubset:
            self.cfg.subset = subset
            self._load(subset)
        for llm, data in self.data_by_llm.items():
            out_file = os.path.join(self.out_dir, f"{llm.replace('/', '-')}.jsonl")
            save_dataclass_jsonl(out_file, *data)

    def _load(self, subset: AccordSubset) -> None:
        for walk in walk_files(self.cfg.llm_output_dir(self.path.experiment2_dir)):
            inference_path, nickname = walk.path, walk.no_ext().replace("/", "-")
            if self._skip(nickname):
                continue
            self.print(f"Processing subset {subset.value} for {nickname}...")
            data = self.loaders[subset].load(inference_path, nickname)
            llm_data = self.data_by_llm.setdefault(nickname, [])
            for logprob_data, partner in self._validate_pairs(data, subset):
                llm_data.extend(logprob_data.crystalize(partner, start_label="A"))

    def _skip(self, test_nickname: Nickname) -> bool:
        for nickname in self.cfg.analysis_llms:
            if test_nickname == nickname.replace("/", "-"):
                return False
        return True

    def _validate_pairs(
        self, data: dict[AccordID, LogprobData], subset: AccordSubset
    ) -> list[tuple[LogprobData, LogprobData | None]]:
        pairs: dict[str, list[LogprobData]] = {}
        for logprob_data in data.values():
            if not logprob_data.is_complete(count=self.cfg.label_count):
                continue
            pairs.setdefault(logprob_data.accord_group_id, []).append(logprob_data)
        if subset == AccordSubset.BASELINE:
            return [(single[0], None) for single in pairs.values() if len(single) == 1]
        return [(pair[0], pair[1]) for pair in pairs.values() if len(pair) == 2]
