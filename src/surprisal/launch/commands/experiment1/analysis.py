from datetime import datetime
from enum import Enum
import os

from coma import command
from scipy.stats import ttest_rel, ttest_ind
import plotly.express as px
import pandas as pd

from ....inference import Inference
from ....llms import Nickname
from ....prompting import PromptType
from ....core import (
    ConceptNetFormatter,
    Label,
    Logprobs,
    SpacedSubsequence,
    TermFormatter,
    Triplet,
)
from ....io import (
    ConditionalPrinter,
    PathConfig,
    ensure_path,
    init_logger,
    load_dataclass_jsonl,
    walk_files,
)

from .base import AggregatorOption, Config


class LogprobType(Enum):
    IN = "IN"
    OUT = "OUT"


class DataLoader:
    def __init__(
        self,
        formatter: ConceptNetFormatter,
        analysis_dir: str,
        user_template_indicator: str,
        flip_logprobs: bool,
        aggregators: list[AggregatorOption],
    ):
        self.formatter = formatter
        self.analysis_dir = analysis_dir
        self.indicator = user_template_indicator
        self.flip = flip_logprobs
        self.aggregators = aggregators
        self.data = self.logger = None

    def load(self, inference_path: str, llm: Nickname) -> None:
        self.data, self.logger = {}, None
        for inference in load_dataclass_jsonl(inference_path, t=Inference):
            # TODO: When it comes time for exp 2, we need to differentiate between the
            #  logprobs of the answer choices as they first appear and are given to
            #  the LLM and the logprob of the eventual teacher-forced answer.
            #  -> We need a way to extract the RIGHT:
            #     a. logprobs of answer label, for all answer labels in one inference
            #     b. logprobs of answer choice value, for all answer choice values in one inference
            #     c. logprobs of teacher-forced output answer label, for all answer labels over MULTIPLE inferences
            #         -> IDEA: Make sure the structure of the ACCORD prompt has ANSWER
            #            CHOICES (options) vs ANSWER (label). Find the index of ANSWER
            #            CHOICES. Next, find the index of ANSWER *after* that. Now,
            #            implement an 'end_idx' in Logprobs.indices_of(). Look for
            #            choices *in between* ANSWER CHOICES and ANSWER. Look for label
            #            *after* ANSWER. Works as long as 'answer' isn't in the MCQ, or
            #            it does if we look for 'ANSWER:' and it is case sensitive.
            #         -> HOWEVER: When/if we have a few-shot prompt, we have have
            #            several ANSWER CHOICES and ANSWER, so we need to employ a
            #            smart self.indicator to overcome that.
            if inference.error_message is not None:
                continue
            for logprob_type in LogprobType:
                logprobs = self._get_logprobs(inference, logprob_type, llm)
                self._fill(inference, logprobs, logprob_type)

    def get(self) -> pd.DataFrame:
        # Keep only complete groups.
        complete_group_size = len(["F", "AF"]) * len(["T", "F"]) * len(LogprobType)
        return (
            pd.DataFrame(self.data)
            .groupby(by=["GroupID"])
            .filter(lambda df_: len(df_.index) == complete_group_size)
        )

    def _get_logprobs(
        self,
        inference: Inference,
        logprob_type: LogprobType,
        llm: Nickname,
    ) -> list[float] | None:
        target = self._get_target(inference, logprob_type)
        logprobs = Logprobs.from_dict(inference.derived_data["prompt_logprobs"])

        # Validate the start index (if any).
        start_idx = self._get_start_idx(logprob_type, logprobs)
        if start_idx is None:
            no_line_breaks = logprobs.to_text().replace("\n", "<NL>")
            self._log_failure(
                llm,
                f"GroupID={inference.prompt_data.group_id}: Start index based on target"
                f" '{self.indicator}' has multiple occurrences in '{no_line_breaks}'.",
            )
            return None

        # Extract and validate the desired spaced sequence.
        spaced_sequences = list(logprobs.indices_of(target, start_idx=start_idx))
        spaced_sequence = self._extract(spaced_sequences)
        if spaced_sequence is None:
            no_line_breaks = logprobs.to_text().replace("\n", "<NL>")
            self._log_failure(
                llm,
                f"GroupID={inference.prompt_data.group_id}: Target '{target}' has "
                f"{len(spaced_sequences)} occurrences in '{no_line_breaks}'.",
            )
            return None
        return spaced_sequence.to_chosen_logprobs()

    def _log_failure(self, llm: Nickname, text: str) -> None:
        logger_name = f"exp.1.analysis.{llm}"
        now = datetime.now().isoformat()
        log_file = os.path.join(self.analysis_dir, f"{llm}-{now}.log")
        self.logger = self.logger or init_logger(logger_name, log_file)
        self.logger.info(text)

    def _get_target(self, inference: Inference, logprob_type: LogprobType) -> str:
        if logprob_type == LogprobType.IN:
            triplet = Triplet(**inference.prompt_data.additional_data["triplet"])
            return self.formatter.formatter.ensure_plain_text(triplet.target)
        elif logprob_type == LogprobType.OUT:
            # Can't use inference.output for Experiment 1, since it is always None.
            # Instead, directly grab the teacher-forced label.
            return inference.prompt_data.label
        else:
            raise ValueError(f"Unsupported LogprobType: {logprob_type}")

    def _get_start_idx(
        self, logprob_type: LogprobType, logprobs: Logprobs
    ) -> int | None:
        if logprob_type == LogprobType.IN:
            return 0
        elif logprob_type == LogprobType.OUT:
            spaced_sequences = list(logprobs.indices_of(self.indicator))
            if len(spaced_sequences) == 1:
                return min(spaced_sequences[0].indices)
            return None
        else:
            raise ValueError(f"Unsupported logprob type: {logprob_type}")

    def _extract(self, seqs: list[SpacedSubsequence]) -> SpacedSubsequence | None:
        if len(seqs) == 1:
            return seqs[0]
        desired_seqs = [s for s in seqs if self.formatter.is_desired_target(s)]
        if len(desired_seqs) == 1:
            return desired_seqs[0]
        return None

    def _fill(
        self,
        inference: Inference,
        logprobs: list[float] | None,
        logprob_type: LogprobType,
    ) -> None:
        if logprobs is None:
            # Don't add if logprobs could not be found.
            return

        prompt_type = inference.prompt_data.prompt_type
        factuality = "Factual" if prompt_type == PromptType.F else "Anti-Factual"
        self.data.setdefault("Factuality", []).append(factuality)

        # Can't use inference.output for Experiment 1, since it is always None.
        # Instead, directly grab the teacher-forced label.
        self.data.setdefault("Label", []).append(inference.prompt_data.label)

        triplet = Triplet(**inference.prompt_data.additional_data["triplet"])
        self.data.setdefault("RelationType", []).append(triplet.relation)
        self.data.setdefault("GroupID", []).append(inference.prompt_data.group_id)
        self.data.setdefault("LogprobType", []).append(logprob_type.value)
        for a in self.aggregators:
            self.data.setdefault(a.value, []).append(a.aggregate(logprobs, self.flip))


def make_plot_path(plots_dir: str, main_type: str, llm: Nickname, sub_type: str) -> str:
    base_path = os.path.join(plots_dir, main_type, llm, sub_type)
    return ensure_path(base_path, is_dir=True)


def save_figure(fig, plot_path: str, file_base_name: str) -> None:
    fig.write_image(os.path.join(plot_path, file_base_name + ".png"), scale=3.0)
    fig.write_image(os.path.join(plot_path, file_base_name + ".pdf"))


def post_process_faceted_plot(fig, x_label: str, y_label: str) -> None:
    # Replace facet names: "RelationType=value" -> "<b>value</b>"
    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                f"<b>{a.text.replace(f'RelationType=', '')}</b>"
                if "RelationType" in a.text
                else a.text
            ),
        ),
    )

    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    fig.for_each_xaxis(lambda x_axis: x_axis.update(showticklabels=True))
    fig.update_layout(margin=dict(l=60, r=0, b=70, t=20))

    # Y-axis label.
    fig.add_annotation(
        showarrow=False,
        text=y_label,
        textangle=-90,
        x=0,
        xanchor="center",
        xref="paper",
        y=0.5,
        yanchor="middle",
        yref="paper",
        xshift=-50,
        font=dict(size=16),
    )

    # X-axis label.
    fig.add_annotation(
        showarrow=False,
        text=x_label,
        textangle=0,
        x=0.5,
        xanchor="center",
        xref="paper",
        y=0,
        yanchor="middle",
        yref="paper",
        yshift=-50,
        font=dict(size=16),
    )


class HistogramMaker:
    def __init__(self, plots_dir: str, color_column_name: str):
        self.plots_dir = plots_dir
        self.color = color_column_name

    def make(
        self, df: pd.DataFrame, llm: Nickname, aggregators: list[AggregatorOption]
    ) -> None:
        # Convert raw sequence information into histogram plots.
        for aggregator in aggregators:
            self._make_overall_histogram(df, aggregator, llm)
            self._make_relation_histogram(df, aggregator, llm)

    def _make_overall_histogram(
        self, df: pd.DataFrame, agg: AggregatorOption, llm: Nickname
    ) -> None:
        fig = px.histogram(
            data_frame=df,
            x=agg.value,
            color=self.color,
            template="simple_white",
            barmode="overlay",
            histnorm="percent",
            category_orders={
                "Label": ["TRUE", "FALSE"],
                "Factuality": ["Factual", "Anti-Factual"],
            },
            labels={agg.value: f"Surprisal: {agg.value.lower()}(logprobs)"},
            width=700,  # default is 700
            height=500,  # default is 500,
            # range_x and range_y might need updating for a camera ready version.
        )
        fig.update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            yaxis=dict(title=dict(text="Histogram (%)")),
            margin=dict(l=0, r=0, b=60, t=0),
        )
        plot_path = make_plot_path(self.plots_dir, "histogram", llm, self.color)
        save_figure(fig, plot_path, f"overall_{agg.value.title()}")

    def _make_relation_histogram(
        self, df: pd.DataFrame, agg: AggregatorOption, llm: Nickname
    ) -> None:
        fig = px.histogram(
            data_frame=df,
            x=agg.value,
            barmode="overlay",
            histnorm="percent",
            color=self.color,
            facet_col="RelationType",
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders={
                "RelationType": [
                    "AtLocation",
                    "Causes",
                    "PartOf",
                    "IsA",
                    "UsedFor",
                    "HasPrerequisite",
                ],
                "Label": ["TRUE", "FALSE"],
                "Factuality": ["Factual", "Anti-Factual"],
            },
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            # range_x and range_y might need updating for a camera ready version.
        )
        post_process_faceted_plot(
            fig,
            x_label=f"Surprisal: {agg.value.lower()}(logprobs)",
            y_label="Histogram (%)",
        )
        plot_path = make_plot_path(self.plots_dir, "histogram", llm, self.color)
        save_figure(fig, plot_path, f"relation_{agg.value.title()}")


class BinaryScatterplotMaker:
    def __init__(
        self, plots_dir: str, binary_column_name: str, binary_options: tuple[str, str]
    ):
        self.plots_dir = plots_dir
        self.binary = binary_column_name
        self.options = binary_options

    def make(
        self, df: pd.DataFrame, llm: Nickname, aggregators: list[AggregatorOption]
    ) -> None:
        df = self._pivot_on_binary(df)
        for aggregator in aggregators:
            self._make_overall_scatterplot(df, aggregator, llm)
            self._make_relation_scatterplot(df, aggregator, llm)

    def _pivot_on_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[df[self.binary] == self.options[0]].drop(self.binary, axis=1)
        y = df[df[self.binary] == self.options[1]].drop(self.binary, axis=1)
        on = ["RelationType", "GroupID", "LogprobType"]
        if "Label" in x.columns:
            on.append("Label")
        if "Factuality" in x.columns:
            on.append("Factuality")
        return x.merge(y, how="inner", on=on)

    def _make_overall_scatterplot(
        self, df: pd.DataFrame, agg: AggregatorOption, llm: Nickname
    ) -> None:
        fig = px.scatter(
            data_frame=df,
            x=f"{agg.value}_x",
            y=f"{agg.value}_y",
            template="simple_white",
            labels={
                f"{agg.value}_x": f"{self.options[0]} Surprisal: {agg.value.lower()}(logprobs)",
                f"{agg.value}_y": f"{self.options[1]} Surprisal: {agg.value.lower()}(logprobs)",
            },
            width=700,  # default is 700
            height=500,  # default is 500,
            # range_x and range_y might need updating for a camera ready version.
        )
        fig.update_layout(margin=dict(l=0, r=0, b=60, t=0))
        self._diagonalize(fig, df, agg)
        plot_path = make_plot_path(self.plots_dir, "scatterplot", llm, self.binary)
        save_figure(fig, plot_path, f"overall_{agg.value.title()}")

    def _make_relation_scatterplot(
        self, df: pd.DataFrame, agg: AggregatorOption, llm: Nickname
    ) -> None:
        fig = px.scatter(
            data_frame=df,
            x=f"{agg.value}_x",
            y=f"{agg.value}_y",
            facet_col="RelationType",
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders={
                "RelationType": [
                    "AtLocation",
                    "Causes",
                    "PartOf",
                    "IsA",
                    "UsedFor",
                    "HasPrerequisite",
                ],
            },
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            # range_x and range_y might need updating for a camera ready version.
        )
        x_label = f"{self.options[0]} Surprisal: {agg.value.lower()}(logprobs)"
        y_label = f"{self.options[1]} Surprisal: {agg.value.lower()}(logprobs)"
        post_process_faceted_plot(fig, x_label, y_label)
        self._diagonalize(fig, df, agg)
        plot_path = make_plot_path(self.plots_dir, "scatterplot", llm, self.binary)
        save_figure(fig, plot_path, f"relation_{agg.value.title()}")

    @staticmethod
    def _diagonalize(fig, df: pd.DataFrame, agg: AggregatorOption) -> None:
        x = df[f"{agg.value}_x"].tolist()
        y = df[f"{agg.value}_y"].tolist()
        min_max = [min(min(x), min(y)) - 1, max(max(x), max(y)) + 1]
        fig.add_scatter(
            x=min_max, y=min_max, mode="lines", line_color="gray", showlegend=False
        )
        fig.update_yaxes(range=min_max)
        fig.update_xaxes(range=min_max)


class CrossScatterplotMaker:
    def __init__(self, plots_dir: str, cartesian_cross_plot: bool):
        self.plots_dir = plots_dir
        self.prod = cartesian_cross_plot

    def make(
        self, df: pd.DataFrame, llm: Nickname, aggregators: list[AggregatorOption]
    ) -> None:
        df = self._pivot_on_logprob_type(df)
        if self.prod:
            df["Legend"] = df.apply(
                lambda x: x["Factuality"] + ", " + x["Label"], axis=1
            )
        for aggregator in aggregators:
            for label, group_df in df.groupby(by="Label"):
                # TODO: An alternative to plotting each label separately is to
                #  plot the most certain (highest logprob) label and so only make one
                #  plot.
                # TODO: If copying this code from exp1 to exp2, consider that there
                #  will be 5 labels. You could simplify to just 1 as above, or you
                #  could use an aggregate measure like entropy of the label
                #  distribution, or you could do something like certainty in the GT
                #  answer, split by whether that GT is F or AF.
                self._make_overall_scatterplot(df, aggregator, llm, str(label))
                self._make_relation_scatterplot(df, aggregator, llm, str(label))

    @staticmethod
    def _pivot_on_logprob_type(df: pd.DataFrame) -> pd.DataFrame:
        x = df[df["LogprobType"] == LogprobType.IN.value]
        x = x.drop("LogprobType", axis=1)
        y = df[df["LogprobType"] == LogprobType.OUT.value]
        y = y.drop("LogprobType", axis=1)
        on = ["RelationType", "GroupID", "Label", "Factuality"]
        return x.merge(y, how="inner", on=on)

    def _make_overall_scatterplot(
        self, df: pd.DataFrame, agg: AggregatorOption, llm: Nickname, label: Label
    ) -> None:
        fig = px.scatter(
            data_frame=df,
            x=f"{agg.value}_x",
            y=f"{agg.value}_y",
            color="Legend" if self.prod else "Factuality",
            template="simple_white",
            category_orders={
                "Label": ["TRUE", "FALSE"],
                "Factuality": ["Factual", "Anti-Factual"],
                "Legend": [
                    "Factual, TRUE",
                    "Anti-Factual, TRUE",
                    "Factual, FALSE",
                    "Anti-Factual, FALSE",
                ],
            },
            labels={
                f"{agg.value}_x": f"Surprisal: {agg.value.lower()}(logprobs)",
                f"{agg.value}_y": f"{label.title()} Uncertainty: {agg.value.lower()}(logprobs)",
            },
            width=700,  # default is 700
            height=500,  # default is 500,
            # range_x and range_y might need updating for a camera ready version.
        )
        fig.update_layout(margin=dict(l=0, r=0, b=60, t=0))
        plot_path = make_plot_path(self.plots_dir, "scatterplot", llm, "cross")
        save_figure(fig, plot_path, f"{label}_overall_{agg.value.title()}")

    def _make_relation_scatterplot(
        self, df: pd.DataFrame, agg: AggregatorOption, llm: Nickname, label: Label
    ) -> None:
        fig = px.scatter(
            data_frame=df,
            x=f"{agg.value}_x",
            y=f"{agg.value}_y",
            color="Legend" if self.prod else "Factuality",
            facet_col="RelationType",
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders={
                "RelationType": [
                    "AtLocation",
                    "Causes",
                    "PartOf",
                    "IsA",
                    "UsedFor",
                    "HasPrerequisite",
                ],
                "Label": ["TRUE", "FALSE"],
                "Factuality": ["Factual", "Anti-Factual"],
                "Legend": [
                    "Factual, TRUE",
                    "Anti-Factual, TRUE",
                    "Factual, FALSE",
                    "Anti-Factual, FALSE",
                ],
            },
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            # range_x and range_y might need updating for a camera ready version.
        )
        x_label = f"Surprisal: {agg.value.lower()}(logprobs)"
        y_label = f"{label.title()} Uncertainty: {agg.value.lower()}(logprobs)"
        post_process_faceted_plot(fig, x_label, y_label)
        plot_path = make_plot_path(self.plots_dir, "scatterplot", llm, "cross")
        save_figure(fig, plot_path, f"{label}_relation_{agg.value.title()}")


class TTest:
    def __init__(
        self,
        analysis_dir: str,
        binary_column_name: str,
        binary_options: tuple[str, str],
        test_type: str,  # Options are: {'independent', 'relative'}
        test_alternative: str,  # Options are: {'two-sided', 'less', 'greater'}
    ):
        self.analysis_dir = analysis_dir
        self.binary = binary_column_name
        self.options = binary_options
        if test_type == "independent":
            self.test = ttest_ind
            self.test_kwargs = dict(
                equal_var=False, nan_policy="raise", alternative=test_alternative
            )
        elif test_type == "relative":
            self.test = ttest_rel
            self.test_kwargs = dict(nan_policy="raise", alternative=test_alternative)
        else:
            raise ValueError(f"Unsupported t-test type: {test_type}")
        self.data = None

    def run(
        self, df: pd.DataFrame, llm: Nickname, aggregators: list[AggregatorOption]
    ) -> None:
        # Convert raw sequence information into stat test results.
        # TODO: Maybe ANOVA is statistically more rigorous than independent t-tests.
        self.data = {}
        for aggregator in aggregators:
            self._do_run_for_aggregator(df, aggregator)
        self.end_run(llm)

    def run_for_aggregator(self, df: pd.DataFrame, agg: AggregatorOption) -> None:
        self.data = self.data or {}
        self._do_run_for_aggregator(df, agg)

    def _do_run_for_aggregator(self, df: pd.DataFrame, agg: AggregatorOption) -> None:
        overall_p_value = self._run_overall_test(df, agg)
        p_value_by_relation = self._run_relation_based_test(df, agg)
        self.data.setdefault("Aggregator", []).append(agg.value.title())
        self.data.setdefault("Overall", []).append(overall_p_value)
        for relation_type, p_value in p_value_by_relation.items():
            self.data.setdefault(relation_type, []).append(p_value)

    def end_run(self, llm: Nickname):
        file_path = os.path.join(self.analysis_dir, self.binary, llm + ".csv")
        pd.DataFrame(self.data).to_csv(ensure_path(file_path), index=False)
        self.data = None

    def _run_overall_test(self, df: pd.DataFrame, agg: AggregatorOption) -> float:
        split_data = {}
        for option, group_df in df.groupby(by=self.binary):
            data = group_df.sort_values(by="GroupID")[agg.value].tolist()
            if option in self.options:
                split_data[option] = data
            else:
                raise ValueError(f"Unknown binary option: {option}")
        for option in self.options:
            if option not in split_data:
                # This means we are missing data for one or both binary options.
                # Can happen if, for example, the LLM outputted FALSE for all
                # prompts (known to be the case for AtLocation + First in POS).
                return -1
        result = self.test(
            a=split_data[self.options[0]],
            b=split_data[self.options[1]],
            **self.test_kwargs,
        )
        return result.pvalue

    def _run_relation_based_test(
        self, df: pd.DataFrame, agg: AggregatorOption
    ) -> dict[str, float]:
        data = {}
        for relation_type, group_df in df.groupby(by="RelationType"):
            data[str(relation_type)] = self._run_overall_test(group_df, agg)
        return data


class Analyzer:
    def __init__(self, path: PathConfig, cfg: Config):
        self.analysis_dir = cfg.analysis_dir(path.experiment1_dir)
        self.plots_dir = cfg.plots_dir(path.experiment1_dir)
        self.print = ConditionalPrinter(cfg.verbose)
        self.path = path
        self.cfg = cfg

    def run(self, df: pd.DataFrame, nickname: Nickname):
        raise NotImplementedError


class FactualityAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config):
        super().__init__(path, cfg)
        self.histograms = HistogramMaker(self.plots_dir, "Factuality")
        self.scatter_plots = BinaryScatterplotMaker(
            plots_dir=self.plots_dir,
            binary_column_name="Factuality",
            binary_options=("Factual", "Anti-Factual"),
        )
        self.t_tests = TTest(
            analysis_dir=self.analysis_dir,
            binary_column_name="Factuality",
            binary_options=("Factual", "Anti-Factual"),
            test_type="relative",
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
        )

    def run(self, df: pd.DataFrame, nickname: Nickname):
        self.print("        Preprocessing data...")
        df = self._preprocess(df)
        self.print("        Plotting histograms...")
        self.histograms.make(df, nickname, self.cfg.aggregators)
        self.print("        Plotting scatter plots...")
        self.scatter_plots.make(df, nickname, self.cfg.aggregators)
        self.print("        Running t-tests...")
        self.t_tests.run(df, nickname, self.cfg.aggregators)
        self.print("        Done.")

    @staticmethod
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["LogprobType"] == LogprobType.IN.value]
        # For all rows in current df, there should be exactly 2 entries with the same
        # 'prompt_logprobs' aggregation based on the labels (which only affects the
        # output logprobs). We only need to keep one of these (arbitrarily, "TRUE").
        # NOTE: In practice, TRUE and FALSE won't have exactly the same aggregations
        # (probably because of some model randomness?), but the values are very close
        # (typically less than 1% error, with 3 sig. figs., often entirely identical).
        return df[df["Label"] == "TRUE"]


class LabelAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config):
        super().__init__(path, cfg)
        self.histograms = HistogramMaker(self.plots_dir, "Label")
        self.t_tests = TTest(
            analysis_dir=self.analysis_dir,
            binary_column_name="Label",
            binary_options=("TRUE", "FALSE"),
            # TODO: An alternative to having independent tests is to further subset the
            #  data by keeping only those where the LLM outputs both TRUE and FALSE in
            #  a given GroupID (typically, F -> TRUE and AF -> FALSE, but not always).
            # TODO: Pro: relative tests are more sensitive. Con: Substantial self-
            #  selection bias creeps in (not to mention potential catastrophic loss of
            #  data quantity, leading to possibly loosing statistical power).
            test_type="independent",
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
        )
        # TODO: Similarly, the labels cannot be used to create the binary scatter plots
        #  unless we restrict to only those GroupIDs where both T and F are outputted.
        # TODO: If we do decide to implement that, consider colouring the points
        #  according to their alignment. Where the GT is Factual and the LLM outputs
        #  TRUE, or where the GT is AF and the LLM outputs FALSE, colour those points
        #  blue. Where the output is opposite, colour those points orange. If there is
        #  no strong pattern, consider a split into 4 colours representing TP/TN/FP/FN.

    def run(self, df: pd.DataFrame, nickname: Nickname):
        for agg in self.cfg.aggregators:
            self.print(f"    Running label analysis for: {agg.value}...")
            self.print("        Preprocess data...")
            agg_df = self._preprocess(df, agg)
            self.print("        Plotting histograms...")
            self.histograms.make(agg_df, nickname, [agg])
            self.print("        Running t-tests...")
            self.t_tests.run_for_aggregator(agg_df, agg)
            self.print("        Done.")
        self.t_tests.end_run(nickname)

    def _preprocess(
        self, df: pd.DataFrame, aggregator: AggregatorOption
    ) -> pd.DataFrame:
        # For each (factuality, group_id) pair, compute the LLM softmax output, then
        # keep only the more probable output label.
        kept_indices = []
        for _, sub_df in df.groupby(by=["Factuality", "GroupID"]):
            true_prob = self._get_label_logprob(sub_df, "TRUE", aggregator)
            false_prob = self._get_label_logprob(sub_df, "FALSE", aggregator)
            kept_label = self._compute_label(true_prob, false_prob)
            mask = self._get_label_mask(sub_df, kept_label, LogprobType.IN)
            kept_index = sub_df.index[mask].tolist()
            if len(kept_index) != 1:
                raise ValueError(f"Logic is off. Debug.")
            kept_indices.extend(kept_index)

        # For each group ID, the above logic + this filtering ensures that exactly one
        # prompt_logprob is kept for Factual and exactly one is kept for Anti-Factual.
        return df.loc[kept_indices]

    @staticmethod
    def _get_label_mask(df: pd.DataFrame, label: Label, logprob_type: LogprobType):
        return (df["Label"] == label) & (df["LogprobType"] == logprob_type.value)

    def _get_label_logprob(
        self, df: pd.DataFrame, label: Label, agg: AggregatorOption
    ) -> float:
        mask = self._get_label_mask(df, label, LogprobType.OUT)
        logprob = df[mask][agg.value].tolist()
        if len(logprob) != 1:
            raise ValueError(f"Logic is off. Debug.")
        return logprob[0]

    def _compute_label(self, true_logprob: float, false_logprob: float) -> str:
        # TODO: Should this actually be a softmax over the two logprobs with a random
        #  seed? Picking the larger one when their difference is large is basically the
        #  same. It is less clear-cut they are very close in value, but it isn't
        #  technically incorrect since the larger one is technically always more likely,
        #  even if they have almost the same likelihood.
        if self.cfg.flip_logprobs:
            return "TRUE" if true_logprob < false_logprob else "FALSE"
        return "TRUE" if true_logprob > false_logprob else "FALSE"


class CrossAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config):
        super().__init__(path, cfg)
        self.scatter_plots = CrossScatterplotMaker(
            self.plots_dir, self.cfg.cartesian_cross_plot
        )

    def run(self, df: pd.DataFrame, nickname: Nickname):
        self.print("        Plotting scatter plots...")
        self.scatter_plots.make(df, nickname, self.cfg.aggregators)
        self.print("        Done.")


@command(name="exp.1.analysis")
class Experiment1Analysis:
    def __init__(self, path: PathConfig, experiment1: Config):
        self.path = path
        self.cfg = experiment1
        self.data = DataLoader(
            formatter=ConceptNetFormatter(
                template=self.cfg.user_template,
                method=self.cfg.data_format_method,
                formatter=TermFormatter(language="en"),
            ),
            analysis_dir=self.cfg.analysis_dir(path.experiment1_dir),
            user_template_indicator=self.cfg.user_template_indicator,
            flip_logprobs=self.cfg.flip_logprobs,
            aggregators=self.cfg.aggregators,
        )
        self.print = ConditionalPrinter(self.cfg.verbose)

    def _skip(self, test_nickname: Nickname) -> bool:
        for nickname in self.cfg.analysis_llms:
            if test_nickname == nickname.replace("/", "-"):
                return False
        return True

    def run(self):
        for walk in walk_files(self.cfg.llm_output_dir(self.path.experiment1_dir)):
            inference_path, nickname = walk.path, walk.no_ext().replace("/", "-")
            if self._skip(nickname):
                continue
            self.print("Analyzing results of model:", nickname)
            self.print("    Loading data...")
            self.data.load(inference_path, nickname)
            self.print("    Running factuality analysis...")
            FactualityAnalyzer(self.path, self.cfg).run(self.data.get(), nickname)
            self.print("    Running label analysis...")
            LabelAnalyzer(self.path, self.cfg).run(self.data.get(), nickname)
            self.print("    Running cross analysis...")
            CrossAnalyzer(self.path, self.cfg).run(self.data.get(), nickname)
