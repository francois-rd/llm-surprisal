from dataclasses import dataclass
from typing import Iterable
from enum import Enum

from scipy.stats import entropy
import numpy as np

from .base import AccordLabel, AccordStatementID
from ..core import AggregatorOption, AggregatorStr


class MetricSubSubType(Enum):
    pass


class AllType(MetricSubSubType):
    ALL = "ALL"


class AnswerType(MetricSubSubType):
    ANSWER = "ANSWER"


class RankSubSubType(MetricSubSubType):
    MATCHING_ACCORD = "MATCHING_ACCORD"
    MATCHING_CSQA = "MATCHING_CSQA"


class PositionSubSubType(MetricSubSubType):
    ACCORD = "ACCORD"
    CSQA = "CSQA"


class SurprisalSubSubType(MetricSubSubType):
    MATCHING_ACCORD = "MATCHING_ACCORD"
    MATCHING_CSQA = "MATCHING_CSQA"
    NOT_MATCHING_ACCORD = "NOT_MATCHING_ACCORD"
    NOT_MATCHING_CSQA = "NOT_MATCHING_CSQA"
    ALL = "ALL"


class MetricSubType(Enum):
    def get_subtype(self) -> type[MetricSubSubType]:
        raise NotImplementedError


class SurprisalSubType(MetricSubType):
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    STATEMENT = "STATEMENT"
    QUESTION = "QUESTION"
    INSTANCE = "INSTANCE"
    FORCED = "FORCED"
    LABEL = "LABEL"
    CHOICE = "CHOICE"
    LLM = "LLM"

    def get_subtype(self) -> type[MetricSubSubType]:
        st = SurprisalSubType
        if self in [st.SOURCE, st.TARGET, st.STATEMENT, st.QUESTION, st.INSTANCE]:
            return AllType
        elif self in [st.FORCED, st.LABEL, st.CHOICE]:
            return SurprisalSubSubType
        elif self == st.LLM:
            return AnswerType
        else:
            raise ValueError(f"Unsupported surprisal type: {self}")


class RankSubType(MetricSubType):
    FORCED = "FORCED"
    LABEL = "LABEL"
    CHOICE = "CHOICE"

    def get_subtype(self) -> type[RankSubSubType]:
        if self in [RankSubType.FORCED, RankSubType.LABEL, RankSubType.CHOICE]:
            return RankSubSubType
        else:
            raise ValueError(f"Unsupported rank sub type: {self}")


class PositionSubType(MetricSubType):
    OF = "OF"

    def get_subtype(self) -> type[PositionSubSubType]:
        if self == PositionSubType.OF:
            return PositionSubSubType
        else:
            raise ValueError(f"Unsupported position sub type: {self}")


class EntropySubType(MetricSubType):
    FORCED = "FORCED"
    LABEL = "LABEL"
    CHOICE = "CHOICE"

    def get_subtype(self) -> type[AllType]:
        if self in [EntropySubType.FORCED, EntropySubType.LABEL, EntropySubType.CHOICE]:
            return AllType
        else:
            raise ValueError(f"Unsupported entropy sub type: {self}")


class MassSubType(MetricSubType):
    FORCED = "FORCED"
    LABEL = "LABEL"
    CHOICE = "CHOICE"

    def get_subtype(self) -> type[AllType]:
        if self in [MassSubType.FORCED, MassSubType.LABEL, MassSubType.CHOICE]:
            return AllType
        else:
            raise ValueError(f"Unsupported mass sub type: {self}")


class MetricType(Enum):
    SURPRISAL = "SURPRISAL"
    RANK = "RANK"
    POSITION = "POSITION"
    ENTROPY = "ENTROPY"
    MASS = "MASS"

    def get_subtype(self) -> type[MetricSubType]:
        if self == MetricType.SURPRISAL:
            return SurprisalSubType
        elif self == MetricType.RANK:
            return RankSubType
        elif self == MetricType.POSITION:
            return PositionSubType
        elif self == MetricType.ENTROPY:
            return EntropySubType
        elif self == MetricType.MASS:
            return MassSubType
        else:
            raise ValueError(f"Unsupported metric type: {self}")

    def uses_aggregator(self) -> bool:
        mt = MetricType
        if self in [mt.SURPRISAL, mt.RANK, mt.ENTROPY, mt.MASS]:
            return True
        elif self == mt.POSITION:
            return False
        else:
            raise ValueError(f"Unsupported metric type: {self}")


@dataclass
class MetricID:
    metric: MetricType
    sub_metric: MetricSubType
    sub_sub_metric: MetricSubSubType
    agg: AggregatorOption | None

    @staticmethod
    def yield_all(aggregators: list[AggregatorOption]) -> Iterable["MetricID"]:
        for metric in MetricType:
            for sub_metric in metric.get_subtype():
                for sub_sub_metric in sub_metric.get_subtype():
                    if metric.uses_aggregator():
                        for agg in aggregators:
                            yield MetricID(metric, sub_metric, sub_sub_metric, agg)
                    else:
                        yield MetricID(metric, sub_metric, sub_sub_metric, None)


# TODO: Logic for all the 'not matching' is that there may be a trend where a metric
#  (either F or AF) has a slight differentiation where the metric gets worse for ALL
#  options EXCEPT either the CSQA answer or the ACCORD answer, where it gets better.
@dataclass
class AccordMetrics:
    surprisal_source_all: dict[AggregatorStr, float | None]
    surprisal_target_all: dict[AggregatorStr, float | None]
    surprisal_statement_all: dict[AggregatorStr, float | None]

    surprisal_question_all: dict[AggregatorStr, float]
    surprisal_instance_all: dict[AggregatorStr, float]

    surprisal_forced_matching_accord: dict[AggregatorStr, float]
    surprisal_label_matching_accord: dict[AggregatorStr, float]
    surprisal_choice_matching_accord: dict[AggregatorStr, float]
    surprisal_forced_matching_csqa: dict[AggregatorStr, float]
    surprisal_label_matching_csqa: dict[AggregatorStr, float]
    surprisal_choice_matching_csqa: dict[AggregatorStr, float]

    surprisal_forced_not_matching_accord: dict[AggregatorStr, float]
    surprisal_label_not_matching_accord: dict[AggregatorStr, float]
    surprisal_choice_not_matching_accord: dict[AggregatorStr, float]
    surprisal_forced_not_matching_csqa: dict[AggregatorStr, float]
    surprisal_label_not_matching_csqa: dict[AggregatorStr, float]
    surprisal_choice_not_matching_csqa: dict[AggregatorStr, float]

    surprisal_forced_all: dict[AggregatorStr, float]
    surprisal_label_all: dict[AggregatorStr, float]
    surprisal_choice_all: dict[AggregatorStr, float]

    surprisal_llm_answer: dict[AggregatorStr, float]

    rank_forced_matching_accord: dict[AggregatorStr, int]
    rank_label_matching_accord: dict[AggregatorStr, int]
    rank_choice_matching_accord: dict[AggregatorStr, int]
    rank_forced_matching_csqa: dict[AggregatorStr, int]
    rank_label_matching_csqa: dict[AggregatorStr, int]
    rank_choice_matching_csqa: dict[AggregatorStr, int]

    position_of_accord: int
    position_of_csqa: int

    entropy_forced_all: dict[AggregatorStr, float]
    entropy_label_all: dict[AggregatorStr, float]
    entropy_choice_all: dict[AggregatorStr, float]

    # Technically, these are only true probability masses is aggregator is SUM.
    mass_forced_all: dict[AggregatorStr, float]
    mass_label_all: dict[AggregatorStr, float]
    mass_choice_all: dict[AggregatorStr, float]

    @staticmethod
    def as_attribute_name(metric_id: MetricID) -> str:
        metrics = [
            metric_id.metric.value,
            metric_id.sub_metric.value,
            metric_id.sub_sub_metric.value,
        ]
        return "_".join(metrics).lower()

    def get(self, metric_id: MetricID) -> float | int:
        attr, agg = self.as_attribute_name(metric_id), metric_id.agg
        return getattr(self, attr) if agg is None else getattr(self, attr)[agg.value]

    def compute_correctness(self) -> bool:
        # The LLM is correct iff the token of the forced answer matching the
        # ground truth label (the ACCORD label) is top ranked compared to all
        # other forced answer labels.
        return self.rank_forced_matching_accord[AggregatorOption.MIN.value] == 1

    @classmethod
    def from_data(
        cls,
        source_lps: dict[AggregatorOption, list[float] | None],
        target_lps: dict[AggregatorOption, list[float] | None],
        question_lps: dict[AggregatorOption, float],
        instance_lps: dict[AggregatorOption, float],
        forced_lps: dict[AccordLabel, dict[AggregatorOption, float]],
        label_lps: dict[AccordLabel, dict[AggregatorOption, float]],
        choice_lps: dict[AccordLabel, dict[AggregatorOption, float]],
        accord_label: AccordLabel,
        csqa_label: AccordLabel,
        start_label: AccordLabel,
    ) -> "AccordMetrics":
        return AccordMetrics(
            # All x source/target/both surprisal.
            surprisal_source_all=cls._surprisal_basic(source_lps),
            surprisal_target_all=cls._surprisal_basic(target_lps),
            surprisal_statement_all=cls._surprisal_basic(source_lps, target_lps),
            # All x question/context surprisal.
            surprisal_question_all={a.value: lp for a, lp in question_lps.items()},
            surprisal_instance_all={a.value: lp for a, lp in instance_lps.items()},
            # Match/no-match x forced/label/choice x accord/csqa surprisal.
            surprisal_forced_matching_accord={
                a.value: lp for a, lp in forced_lps[accord_label].items()
            },
            surprisal_label_matching_accord={
                a.value: lp for a, lp in label_lps[accord_label].items()
            },
            surprisal_choice_matching_accord={
                a.value: lp for a, lp in choice_lps[accord_label].items()
            },
            surprisal_forced_matching_csqa={
                a.value: lp for a, lp in forced_lps[csqa_label].items()
            },
            surprisal_label_matching_csqa={
                a.value: lp for a, lp in label_lps[csqa_label].items()
            },
            surprisal_choice_matching_csqa={
                a.value: lp for a, lp in choice_lps[csqa_label].items()
            },
            surprisal_forced_not_matching_accord=cls._surprisal_skip(
                forced_lps, accord_label
            ),
            surprisal_label_not_matching_accord=cls._surprisal_skip(
                label_lps, accord_label
            ),
            surprisal_choice_not_matching_accord=cls._surprisal_skip(
                choice_lps, accord_label
            ),
            surprisal_forced_not_matching_csqa=cls._surprisal_skip(
                forced_lps, csqa_label
            ),
            surprisal_label_not_matching_csqa=cls._surprisal_skip(
                label_lps, csqa_label
            ),
            surprisal_choice_not_matching_csqa=cls._surprisal_skip(
                choice_lps, csqa_label
            ),
            # All x forced/label/choice surprisal.
            surprisal_forced_all=cls._surprisal_skip(forced_lps, None),
            surprisal_label_all=cls._surprisal_skip(label_lps, None),
            surprisal_choice_all=cls._surprisal_skip(choice_lps, None),
            # Answer x llm surprisal.
            surprisal_llm_answer=cls._surprisal_llm_answer(forced_lps),
            # Forced/label/choice x accord/csqa rank.
            rank_forced_matching_accord=cls._rank(forced_lps, accord_label),
            rank_label_matching_accord=cls._rank(label_lps, accord_label),
            rank_choice_matching_accord=cls._rank(choice_lps, accord_label),
            rank_forced_matching_csqa=cls._rank(forced_lps, csqa_label),
            rank_label_matching_csqa=cls._rank(label_lps, csqa_label),
            rank_choice_matching_csqa=cls._rank(choice_lps, csqa_label),
            # Accord/csqa label position.
            position_of_accord=cls._position(accord_label, start_label),
            position_of_csqa=cls._position(csqa_label, start_label),
            # All x forced/label/choice entropy.
            entropy_forced_all=cls._entropy(forced_lps),
            entropy_label_all=cls._entropy(label_lps),
            entropy_choice_all=cls._entropy(choice_lps),
            # All x forced/label/choice entropy.
            mass_forced_all=cls._mass(forced_lps),
            mass_label_all=cls._mass(label_lps),
            mass_choice_all=cls._mass(choice_lps),
        )

    @staticmethod
    def _invert(
        data: dict[AccordLabel, dict[AggregatorOption, float]],
        skip: AccordLabel | None = None,
        collapse: bool = False,
    ):
        data_by_agg = {}
        for label, lp_by_agg in data.items():
            if skip is not None and label == skip:
                continue
            for agg, lp in lp_by_agg.items():
                if collapse:
                    data_by_agg.setdefault(agg, []).append(lp)
                else:
                    data_by_agg.setdefault(agg, {})[label] = lp
        return data_by_agg

    @staticmethod
    def _surprisal_basic(
        data: dict[AggregatorOption, list[float] | None],
        data2: dict[AggregatorOption, list[float] | None] | None = None,
    ) -> dict[AggregatorStr, float | None]:
        data_all = data
        if data2 is not None:
            for agg in data_all:
                if data_all[agg] is None or data2[agg] is None:
                    data_all[agg] = None
                else:
                    data_all[agg] += data2[agg]
        return {
            agg.value: None if lp is None else agg.aggregate(lp)
            for agg, lp in data.items()
        }

    @classmethod
    def _surprisal_skip(
        cls,
        data: dict[AccordLabel, dict[AggregatorOption, float]],
        skip: AccordLabel | None,
    ) -> dict[AggregatorStr, float]:
        data_by_agg = cls._invert(data, skip, collapse=True)
        return {agg.value: agg.aggregate(lp) for agg, lp in data_by_agg.items()}

    @classmethod
    def _surprisal_llm_answer(
        cls,
        data: dict[AccordLabel, dict[AggregatorOption, float]],
    ) -> dict[AggregatorStr, float]:
        by_agg = cls._invert(data)
        return {a.value: cls._sort_by_top_rank(lps)[0][1] for a, lps in by_agg.items()}

    @staticmethod
    def _sort_by_top_rank(
        data: dict[AccordLabel, float]
    ) -> list[tuple[AccordLabel, float]]:
        all_neg = all(x < 0 for x in data.values())
        return sorted(data.items(), key=lambda x: x[1], reverse=all_neg)

    @classmethod
    def _rank(
        cls, data: dict[AccordLabel, dict[AggregatorOption, float]], label: AccordLabel
    ) -> dict[AggregatorStr, int]:
        data_by_agg = cls._invert(data)
        return {agg.value: cls._do_rank(lps, label) for agg, lps in data_by_agg.items()}

    @classmethod
    def _do_rank(cls, data: dict[AccordLabel, float], label: AccordLabel) -> int:
        return list(dict(cls._sort_by_top_rank(data))).index(label) + 1

    @staticmethod
    def _position(test_label: AccordLabel, start_label: AccordLabel) -> int:
        return ord(test_label) - ord(start_label) + 1

    @classmethod
    def _entropy(
        cls, data: dict[AccordLabel, dict[AggregatorOption, float]]
    ) -> dict[AggregatorStr, float]:
        data_by_agg = cls._invert(data, collapse=True)
        return {agg.value: cls._do_entropy(lps) for agg, lps in data_by_agg.items()}

    @staticmethod
    def _do_entropy(data: list[float]) -> float:
        # LLMs use base e. Internally, dist is normalized. If pos logprobs, need neg.
        return float(entropy(np.exp([-abs(x) for x in data])))

    @classmethod
    def _mass(
        cls, data: dict[AccordLabel, dict[AggregatorOption, float]]
    ) -> dict[AggregatorStr, float]:
        # Technically, these are only true probability masses is aggregator is SUM.
        data_by_agg = cls._invert(data, collapse=True)
        return {agg.value: cls._do_mass(lps) for agg, lps in data_by_agg.items()}

    @staticmethod
    def _do_mass(data: list[float]) -> float:
        return float(sum(np.exp([-abs(x) for x in data])))  # If pos logprobs, need neg.


class PairedMetricType(Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"
    STATEMENT = "STATEMENT"


@dataclass
class PairedMetricID:
    metric: PairedMetricType
    agg: AggregatorOption | None

    @staticmethod
    def yield_all(aggregators: list[AggregatorOption]) -> Iterable["PairedMetricID"]:
        for metric in PairedMetricType:
            for agg in aggregators:
                yield PairedMetricID(metric, agg)


@dataclass
class PairedAccordMetrics:
    paired_source: dict[AggregatorStr, float]
    paired_target: dict[AggregatorStr, float]
    paired_statement: dict[AggregatorStr, float]

    @staticmethod
    def as_attribute_name(paired_metric_id: PairedMetricID) -> str:
        return f"paired_{paired_metric_id.metric.value.lower()}"

    def get(self, paired_metric_id: PairedMetricID) -> float:
        attr, agg = self.as_attribute_name(paired_metric_id), paired_metric_id.agg
        return getattr(self, attr)[agg.value]

    @classmethod
    def from_data(
        cls,
        factual_or_correct_source_lps: dict[
            tuple[AccordLabel, AccordStatementID], dict[AggregatorOption, float]
        ],
        factual_or_correct_target_lps: dict[
            tuple[AccordLabel, AccordStatementID], dict[AggregatorOption, float]
        ],
        af_or_incorrect_source_lps: dict[
            tuple[AccordLabel, AccordStatementID], dict[AggregatorOption, float]
        ],
        af_or_incorrect_target_lps: dict[
            tuple[AccordLabel, AccordStatementID], dict[AggregatorOption, float]
        ],
    ) -> "PairedAccordMetrics":
        source = cls._pair_up(factual_or_correct_source_lps, af_or_incorrect_source_lps)
        target = cls._pair_up(factual_or_correct_target_lps, af_or_incorrect_target_lps)
        combine = {agg: lps + target[agg] for agg, lps in source.items()}
        return PairedAccordMetrics(
            paired_source=cls._aggregate(source),
            paired_target=cls._aggregate(target),
            paired_statement=cls._aggregate(combine),
        )

    @staticmethod
    def _pair_up(
        factual_or_correct_lps: dict[
            tuple[AccordLabel, AccordStatementID], dict[AggregatorOption, float]
        ],
        af_or_incorrect_lps: dict[
            tuple[AccordLabel, AccordStatementID], dict[AggregatorOption, float]
        ],
    ) -> dict[AggregatorOption, list[float]]:
        result = {}
        for key, f_or_c_data in factual_or_correct_lps.items():
            for agg, af_or_i_lp in af_or_incorrect_lps[key].items():
                result.setdefault(agg, []).append(f_or_c_data[agg] - af_or_i_lp)
        return result

    @staticmethod
    def _aggregate(
        data: dict[AggregatorOption, list[float]]
    ) -> dict[AggregatorStr, float]:
        return {agg.value: agg.aggregate(lp) for agg, lp in data.items()}
