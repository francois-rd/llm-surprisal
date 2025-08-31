# TODO:
#  Either include tree size 0 (baseline) as a first facet, or else normalize
#        every other facet by subtracting baseline (somehow... since each baseline
#        attaches to multiple different non-baseline instances).


# TODO: We need to implement multiple label simplification methods (for both true labels
#     (A-E) as well as the answer choice text):
#     a. Binarize (metric: surprisal):
#        a.1 LLM is correct/incorrect (llm.output.label == meta_data.chosen_label)
#            - Analogous to the False_overall_Min cross-plots of exp1 by instead of FALSE
#              or TRUE, you plot surprisal of CORRECT in one and INCORRECT in the other.
#            - Possibly double-colored by (b).. or we could 'mark' by the label letter?
#        a.2 CSQA-base label (factual label) vs aggregation (sum/mean/min) over all 4
#            other labels... alternatively, chosen label (accord label) vs aggregation
#            of the others, then further colored/marked by F/AF of the chosen label..oh wait, we are already doing that!
#             - Logic for both is that there may be a trend between surprisal (F or AF)
#               in the statements and the mean/sum surprisal level of all answers (see
#               (e) below for that exact idea). Here, we are seeing if there is a slight
#               differentiation where surprisal goes up for ALL options EXCEPT either
#               the CSQA-base answer or the chosen answer.

# Histogram:
# QUESTION: Is there anything besides factuality that makes sense as a colour?
#  - LLM correctness? But how to represent when we have 1 correct and 4 incorrect?

# TODO: Scatter plot:
#  First, you can scatter any 2 metrics from the histograms against each other
#  - but to make the number of plots tractable, only do this when at least one metric
#    has a strong t-test outcomes
#  Second, you can scatter any of those same metrics against a new one:
#   - aggregate (for matching aggregator metric) over all forced labels, all answer labels, all answer choices
#     EXCEPT either the correct one (accord label) or the base one (csqa label)
#   - actually, is there any reason this cannot simply be another histogram metric????
import os

from coma import command
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
import pandas as pd
import plotly.express as px

from ....core import AggregatorOption
from ....accord import AccordMetrics, MetricID
from ....io import ConditionalPrinter, PathConfig, load_dataclass_jsonl, ensure_path
from ....llms import Nickname

from .base import AccordSubset, Config
from .pre_analysis import LogprobDataclass


def metric_as_string(metric_id: MetricID, add_agg: bool = False) -> str:
    if metric_id.metric.uses_aggregator() and add_agg:
        sub_title = f"_{metric_id.agg.value.title()}"
    else:
        sub_title = ""
    return f"{AccordMetrics.as_attribute_name(metric_id)}{sub_title}"


def factuality_diff_by_subsets(group_df):
    af = group_df[group_df["Factuality"] == "Anti-Factual"]
    if af.empty:  # This is the case for subset 0 (i.e., the baseline).
        return None
    f = group_df[group_df["Factuality"] == "Factual"]["Metric"].tolist()[0]
    return f - af["Metric"].tolist()[0]


def as_factuality_diff(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(by=["GroupID", "Subset"])
    result = grouped.apply(factuality_diff_by_subsets, include_groups=False)
    return result.reset_index().rename(columns={0: "Diff"})


def compute_outlier_mask(data: np.array, threshold: float) -> np.array:
    # Credit: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    return s < threshold


class DataLoader:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path, self.cfg = path, experiment2
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.id_counter = 0
        self.data_by_id: dict[int, LogprobDataclass] = {}
        self.data_by_llm: dict[Nickname, dict] = {}

    def load(self) -> None:
        in_dir = self.cfg.pre_analysis_dir(self.path.experiment2_dir)
        for nickname in self.cfg.analysis_llms:
            nickname = nickname.replace("/", "-")
            self.print(f"    Loading data for: {nickname}...")
            in_file = os.path.join(in_dir, f"{nickname}.jsonl")
            for data in load_dataclass_jsonl(in_file, t=LogprobDataclass):
                self._add_to_data(nickname, data)
                self.data_by_id[self.id_counter] = data
                self.id_counter += 1

    def _add_to_data(self, llm: Nickname, data: LogprobDataclass):
        llm_data = self.data_by_llm.setdefault(llm, {})
        llm_data.setdefault("GroupID", []).append(data.accord_group_id)
        llm_data.setdefault("Factuality", []).append(data.factuality)
        llm_data.setdefault("AccordLabel", []).append(data.accord_label)
        llm_data.setdefault("CsqaLabel", []).append(data.csqa_label)
        llm_data.setdefault("Subset", []).append(data.subset.value)
        llm_data.setdefault("DataID", []).append(self.id_counter)

    def make_df(self, llm: Nickname, metric_id: MetricID) -> pd.DataFrame:
        df = pd.DataFrame(self.data_by_llm[llm.replace("/", "-")])
        df["Metric"] = df["DataID"].apply(
            lambda id_: self.data_by_id[id_].metrics.get(metric_id)
        )
        return df


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
    def __init__(self, plots_dir: str, data: DataLoader):
        self.plots_dir = plots_dir
        self.data = data

    def make(self, llm: Nickname, aggregators: list[AggregatorOption]) -> None:
        for metric_id in MetricID.yield_all(aggregators):
            df = self.data.make_df(llm, metric_id)
            self._do_make_factuality(df, llm, metric_id)
            self._do_make_diff(df, llm, metric_id)

    def make_select(self, llm: Nickname, selection: list[MetricID]) -> None:
        for metric_id in selection:
            df = self.data.make_df(llm, metric_id)
            self._do_make_factuality(df, llm, metric_id)
            self._do_make_diff(df, llm, metric_id)

    def _do_make_factuality(self, df: pd.DataFrame, llm: Nickname, metric_id: MetricID):
        fig = px.histogram(
            data_frame=df,
            x="Metric",
            barmode="overlay",
            histnorm="percent",
            color="Factuality",
            facet_col="Subset",
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders={
                "Subset": [subset.value for subset in AccordSubset],
                "Factuality": ["Factual", "Anti-Factual"],
            },
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            # range_x and range_y might need updating for a camera ready version.
        )

        if metric_id.metric.uses_aggregator():
            sub_label = f": {metric_id.agg.value.lower()}(logprobs)"
        else:
            sub_label = ""

        post_process_faceted_plot(
            fig,
            x_label=f"{metric_id.metric.value.title()}{sub_label}",
            y_label="Histogram (%)",
        )
        plot_path = make_plot_path(self.plots_dir, "histogram", llm, "Factuality")
        save_figure(fig, plot_path, metric_as_string(metric_id, add_agg=True))

    def _do_make_diff(self, df: pd.DataFrame, llm: Nickname, metric_id: MetricID):
        fig = px.histogram(
            data_frame=as_factuality_diff(df),
            x="Diff",
            histnorm="percent",
            facet_col="Subset",
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders={"Subset": [subset.value for subset in AccordSubset]},
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            nbins=25,
            # range_x and range_y might need updating for a camera ready version.
        )

        if metric_id.metric.uses_aggregator():
            sub_label = f": {metric_id.agg.value.lower()}(logprobs)"
        else:
            sub_label = ""
        sub_title = f"{metric_id.metric.value.title()}{sub_label}"

        post_process_faceted_plot(
            fig,
            x_label=f"Paired difference in Factuality of {sub_title}",
            y_label="Histogram (%)",
        )
        fig.update_traces(marker_color="green")
        plot_path = make_plot_path(self.plots_dir, "histogram", llm, "Diff")
        save_figure(fig, plot_path, metric_as_string(metric_id, add_agg=True))


class ViolinMaker:
    def __init__(self, plots_dir: str, data: DataLoader, outlier_threshold: float):
        self.plots_dir = plots_dir
        self.data = data
        self.outliers = outlier_threshold

    def make(self, llm: Nickname, aggregators: list[AggregatorOption]) -> None:
        for metric_id in MetricID.yield_all(aggregators):
            df = self._make_df(llm, metric_id)
            self._do_make(df, llm, metric_id, reject_outliers=False)
            self._do_make(df, llm, metric_id, reject_outliers=True)

    def make_select(self, llm: Nickname, selection: list[MetricID]) -> None:
        for metric_id in selection:
            df = self._make_df(llm, metric_id)
            self._do_make(df, llm, metric_id, reject_outliers=False)
            self._do_make(df, llm, metric_id, reject_outliers=True)

    def _make_df(self, llm: Nickname, metric_id: MetricID) -> pd.DataFrame:
        df = as_factuality_diff(self.data.make_df(llm, metric_id))
        return df[df["Subset"] != 0]

    def _reject_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outliers <= 0:
            return df
        return df[compute_outlier_mask(np.array(df["Diff"]), self.outliers)]

    def _do_make(
        self,
        df: pd.DataFrame,
        llm: Nickname,
        metric_id: MetricID,
        reject_outliers: bool,
    ):
        fig = px.violin(
            data_frame=self._reject_outliers(df) if reject_outliers else df,
            y="Diff",
            color="Subset",
            box=True,
            category_orders={
                "Subset": [
                    subset.value
                    for subset in AccordSubset
                    if subset != AccordSubset.BASELINE
                ]
            },
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            # range_x and range_y might need updating for a camera ready version.
        )

        if metric_id.metric.uses_aggregator():
            sub_label = f": {metric_id.agg.value.lower()}(logprobs)"
        else:
            sub_label = ""
        sub_title = f"{metric_id.metric.value.title()}{sub_label}"

        post_process_faceted_plot(
            fig,
            x_label="Subset",
            y_label=f"Paired difference in Factuality of {sub_title}",
        )
        fig.add_hline(y=0.0, opacity=0.5, line_width=2, line_dash="dash")
        fig.update_traces(meanline_visible=True)
        fig.update_xaxes(showticklabels=False)  # TODO: This isn't doing anything?
        sub_type = "No_Outliers" if reject_outliers else "All"
        plot_path = make_plot_path(self.plots_dir, "violin", llm, sub_type)
        save_figure(fig, plot_path, metric_as_string(metric_id, add_agg=True))


class TTest:
    def __init__(
        self,
        data: DataLoader,
        analysis_dir: str,
        p_value_threshold: float,
        min_subsets_passing_threshold: int,
        outlier_threshold: float,
        binary_column_name: str,
        binary_options: tuple[str, str],
        test_type: str,  # Options are: {'independent', 'relative'}
        test_alternative: str,  # Options are: {'two-sided', 'less', 'greater'}
    ):
        self.data = data
        self.analysis_dir = analysis_dir
        self.threshold = p_value_threshold
        self.min_count = min_subsets_passing_threshold
        self.outliers = outlier_threshold
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
        self.results = None

    def run(self, llm: Nickname, aggregators: list[AggregatorOption]) -> list[MetricID]:
        self.results, good_p_values = {}, []
        for metric_id in MetricID.yield_all(aggregators):
            if self._do_run(llm, metric_id):
                good_p_values.append(metric_id)
        file_path = os.path.join(self.analysis_dir, self.binary, llm + ".csv")
        pd.DataFrame(self.results).to_csv(ensure_path(file_path), index=False)
        self.results = None
        return good_p_values

    def _do_run(self, llm: Nickname, metric_id: MetricID) -> bool:
        metric_name = metric_as_string(metric_id, add_agg=True)
        self.results.setdefault("Metric", []).append(metric_name)
        df = self.data.make_df(llm, metric_id)
        count = 0
        for subset, group_df in df.groupby(by="Subset"):
            p_value = self._run_test(group_df)
            if 0 < p_value < self.threshold:
                count += 1
            self.results.setdefault(subset, []).append(p_value)
        return count >= self.min_count

    def _reject_outliers(self, a, b) -> tuple[list[float], list[float]]:
        if self.outliers <= 0:
            return a, b
        a, b = np.array(a), np.array(b)
        mask = compute_outlier_mask(a - b, self.outliers)
        return a[mask].tolist(), b[mask].tolist()

    def _run_test(self, df: pd.DataFrame) -> float:
        split_data = {}
        for option, group_df in df.groupby(by=self.binary):
            data = group_df.sort_values(by="GroupID")["Metric"].tolist()
            if option in self.options:
                split_data[option] = data
            else:
                raise ValueError(f"Unknown binary option: {option}")
        for option in self.options:
            if option not in split_data:
                # This means we are missing data for one or both binary options.
                # Can happen if, for example, the LLM outputted FALSE for all
                # prompts (which is definitely true for the BASELINE subset).
                return -1
        a, b = self._reject_outliers(
            a=split_data[self.options[0]],
            b=split_data[self.options[1]],
        )
        result = self.test(a=a, b=b, **self.test_kwargs)
        return result.pvalue


class Analyzer:
    def __init__(self, path: PathConfig, cfg: Config):
        self.analysis_dir = cfg.analysis_dir(path.experiment2_dir)
        self.plots_dir = cfg.plots_dir(path.experiment2_dir)
        self.print = ConditionalPrinter(cfg.verbose)
        self.path = path
        self.cfg = cfg

    def run(self, nickname: Nickname):
        raise NotImplementedError


class FactualityAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config, data: DataLoader):
        super().__init__(path, cfg)
        self.histograms = HistogramMaker(self.plots_dir, data)
        self.violins = ViolinMaker(
            plots_dir=self.plots_dir,
            data=data,
            outlier_threshold=self.cfg.outlier_threshold,
        )
        self.t_tests = TTest(
            data=data,
            analysis_dir=self.analysis_dir,
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Factuality",
            binary_options=("Factual", "Anti-Factual"),
            test_type="relative",
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
        )

    def run(self, nickname: Nickname):
        self.print("        Running t-tests...")
        selection = self.t_tests.run(nickname, self.cfg.aggregators)
        self.print("        Plotting select histograms...")
        self.histograms.make_select(nickname, selection)
        self.print("        Plotting select violins...")
        self.violins.make_select(nickname, selection)
        self.print("        Done.")


@command(name="exp.2.analysis")
class Experiment2Analysis:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path = path
        self.cfg = experiment2
        self.data = DataLoader(self.path, self.cfg)
        self.print = ConditionalPrinter(self.cfg.verbose)

    def run(self):
        self.print("Loading data...")
        self.data.load()
        self.print("Done.")
        for nickname in self.cfg.analysis_llms:
            self.print("Analyzing results of model:", nickname)
            self.print("    Running factuality analysis...")
            FactualityAnalyzer(self.path, self.cfg, self.data).run(nickname)
            self.print("    Running cross analysis...")
            # CrossAnalyzer(self.path, self.cfg, self.data).run(nickname)
