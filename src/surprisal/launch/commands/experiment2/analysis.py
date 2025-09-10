from typing import Any, Callable, Literal
import warnings
import os

from coma import command
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ....core import AggregatorOption
from ....io import ConditionalPrinter, PathConfig, load_dataclass_jsonl, ensure_path
from ....llms import Nickname
from ....accord import (
    AccordMetrics,
    PairedAccordMetrics,
    MetricID,
    PairedMetricID,
    MetricType,
    AnswerType,
    SurprisalSubType,
    SurprisalSubSubType1,
    SurprisalSubSubType2,
)

from .base import AccordSubset, Config
from .pre_analysis import LogprobDataclass

PairingType = Literal["Factuality", "Correctness"]


def all_category_orders() -> dict[str, list]:
    return {
        "Subset": [subset.value for subset in AccordSubset],
        "Factuality": ["Factual", "Anti-Factual"],
        "Correctness": ["True", "False"],
        "ReasoningHops": list(range(max(s.value for s in AccordSubset) + 1)),
        "Distractors": [None] + list(range(max(s.value for s in AccordSubset))),
        "Legend": [
            "Factual, True",
            "Anti-Factual, True",
            "Factual, False",
            "Anti-Factual, False",
        ],
    }


def metric_as_string(metric_id: MetricID, add_agg: bool = False) -> str:
    if metric_id.metric.uses_aggregator() and add_agg:
        sub_title = f"_{metric_id.agg.value.title()}"
    else:
        sub_title = ""
    return f"{AccordMetrics.as_attribute_name(metric_id)}{sub_title}"


def paired_metric_as_string(paired_id: PairedMetricID, add_agg: bool = False) -> str:
    if add_agg:
        suffix = f"_{paired_id.agg.value.title()}"
    else:
        suffix = ""
    return f"{PairedAccordMetrics.as_attribute_name(paired_id)}{suffix}"


def factuality_diff_fn(group_df):
    af = group_df[group_df["Factuality"] == "Anti-Factual"]
    if af.empty:  # This is the case for subset 0 (i.e., the baseline).
        return None
    f = group_df[group_df["Factuality"] == "Factual"]["Metric"].tolist()[0]
    return f - af["Metric"].tolist()[0]


def as_factuality_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = df.groupby(by=["GroupID", col])
    result = grouped.apply(factuality_diff_fn, include_groups=False)
    return result.reset_index().rename(columns={0: "Diff"})


def correctness_diff_fn(group_df):
    t = group_df[group_df["Correctness"]]
    if t.empty:  # This can be the case for subset 0 (i.e., the baseline).
        return None
    f = group_df[~group_df["Correctness"]]
    if f.empty:  # This can be the case for subset 0 (i.e., the baseline).
        return None
    return t["Metric"].tolist()[0] - f["Metric"].tolist()[0]


def as_correctness_diff(df: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = df.groupby(by=["GroupID", col])
    result = grouped.apply(correctness_diff_fn, include_groups=False)
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
        correctness = data.metrics.compute_correctness()
        llm_data = self.data_by_llm.setdefault(llm, {})
        llm_data.setdefault("GroupID", []).append(data.accord_group_id)
        llm_data.setdefault("Factuality", []).append(data.factuality)
        llm_data.setdefault("Correctness", []).append(correctness)
        llm_data.setdefault("ReasoningHops", []).append(data.reasoning_hops)
        llm_data.setdefault("Distractors", []).append(data.distractors)
        llm_data.setdefault("AccordLabel", []).append(data.accord_label)
        llm_data.setdefault("CsqaLabel", []).append(data.csqa_label)
        llm_data.setdefault("Subset", []).append(data.subset.value)
        llm_data.setdefault("DataID", []).append(self.id_counter)

    def make_df(self, llm: Nickname) -> pd.DataFrame:
        return pd.DataFrame(self.data_by_llm[llm.replace("/", "-")])

    def make_metric_df(
        self,
        llm: Nickname,
        metric_id_1: MetricID,
        metric_id_2: MetricID | None = None,
        restrict_to_opposites_only_in_col: str | None = None,
    ) -> pd.DataFrame:
        df = self.make_df(llm)
        if restrict_to_opposites_only_in_col is not None:
            df = self._keep_opposites_only(df, restrict_to_opposites_only_in_col)
        col_name = "Metric" if metric_id_2 is None else "Metric1"
        df[col_name] = df["DataID"].apply(
            lambda id_: self.data_by_id[id_].metrics.get(metric_id_1)
        )
        if metric_id_2 is not None:
            df["Metric2"] = df["DataID"].apply(
                lambda id_: self.data_by_id[id_].metrics.get(metric_id_2)
            )
        return df

    def make_paired_df(
        self, llm: Nickname, paired_metric_id: PairedMetricID, which_type: PairingType
    ) -> pd.DataFrame:
        df = self.make_df(llm)
        df["PairedMetric"] = df["DataID"].apply(
            self._get_paired_fn(paired_metric_id, which_type)
        )
        # BASELINE doesn't have any paired data. For others, F->data and AF->None.
        return df[~df["PairedMetric"].isna()]

    def _get_paired_fn(
        self, paired_metric_id: PairedMetricID, which_type: PairingType
    ) -> Callable[[int], float | None]:
        def helper(data_id: int):
            if which_type == "Factuality":
                metrics = self.data_by_id[data_id].factuality_metrics
            elif which_type == "Correctness":
                metrics = self.data_by_id[data_id].correctness_metrics
            else:
                raise ValueError(f"Unsupported paired metric type: {which_type}")
            return None if metrics is None else metrics.get(paired_metric_id)

        return helper

    @staticmethod
    def _keep_opposites_only(df: pd.DataFrame, col: str) -> pd.DataFrame:
        keep = []
        for group_id, group_df in df.groupby(by="GroupID"):
            if len(group_df[col].unique()) > 1:
                keep.append(group_id)
        return df[df["GroupID"].isin(keep)]


def make_path(root_dir: str, main_type: str, llm: Nickname, sub_type: str) -> str:
    base_path = os.path.join(root_dir, main_type, llm, sub_type)
    return ensure_path(base_path, is_dir=True)


def save_figure(fig, plot_path: str, file_base_name: str) -> None:
    fig.write_image(os.path.join(plot_path, file_base_name + ".png"), scale=3.0)
    fig.write_image(os.path.join(plot_path, file_base_name + ".pdf"))


def post_process_faceted_plot(fig, x_label: str, y_label: str) -> None:
    # Replace facet names: "Subset=value" -> "<b>subset_name</b>"
    fig.for_each_annotation(
        lambda a: a.update(
            text=(
                f"<b>{AccordSubset(int(a.text.replace(f'Subset=', ''))).name}</b>"
                if "Subset" in a.text
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


# TODO: This could use some updating, but the histograms are not as informative as the
#  violin plots (visually), so it's been neglected.
class HistogramMaker:
    def __init__(self, plots_dir: str, data: DataLoader):
        self.plots_dir = plots_dir
        self.data = data

    def make_select(
        self, llm: Nickname, facet_col: str, selection: list[MetricID]
    ) -> None:
        for metric_id in selection:
            df = self.data.make_metric_df(llm, metric_id)
            self._do_make_factuality(df, llm, facet_col, metric_id)
            self._do_make_diff(df, llm, facet_col, metric_id)

    def _do_make_factuality(
        self, df: pd.DataFrame, llm: Nickname, facet_col: str, metric_id: MetricID
    ):
        fig = px.histogram(
            data_frame=df,
            x="Metric",
            barmode="overlay",
            histnorm="percent",
            color="Factuality",
            facet_col=facet_col,
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders=all_category_orders(),
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
        path = make_path(self.plots_dir, "histogram", llm, f"Factuality-{facet_col}")
        save_figure(fig, path, metric_as_string(metric_id, add_agg=True))

    def _do_make_diff(
        self, df: pd.DataFrame, llm: Nickname, facet_col: str, metric_id: MetricID
    ):
        fig = px.histogram(
            data_frame=as_factuality_diff(df, facet_col),
            x="Diff",
            histnorm="percent",
            facet_col=facet_col,
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders=all_category_orders(),
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
        path = make_path(self.plots_dir, "histogram", llm, f"Diff-{facet_col}")
        save_figure(fig, path, metric_as_string(metric_id, add_agg=True))


class DiffViolinMaker:
    def __init__(
        self,
        plots_dir: str,
        data: DataLoader,
        outlier_threshold: float,
        which_type: PairingType,
    ):
        self.plots_dir = plots_dir
        self.data = data
        self.outliers = outlier_threshold
        self.which_type = which_type
        if which_type == "Factuality":
            self.diff_fn = as_factuality_diff
        elif which_type == "Correctness":
            self.diff_fn = as_correctness_diff
        else:
            raise ValueError(f"Unsupported paired metric type: {which_type}")

    def make_metric_select(
        self, llm: Nickname, color_col: str, selection: list[MetricID]
    ) -> None:
        for metric_id in selection:
            df = self.diff_fn(self.data.make_metric_df(llm, metric_id), color_col)
            if color_col == "Subset":
                df = df[df["Subset"] != AccordSubset.BASELINE.value]
            kwargs = self._get_metric_kwargs(color_col, metric_id)
            self._do_make(df, llm, reject_outliers=False, **kwargs)
            self._do_make(df, llm, reject_outliers=True, **kwargs)

    def make_paired_select(
        self, llm: Nickname, color_col: str, selection: list[PairedMetricID]
    ) -> None:
        for paired_id in selection:
            df = self.data.make_paired_df(llm, paired_id, self.which_type)
            kwargs = self._get_paired_kwargs(color_col, paired_id)
            self._do_make(df, llm, reject_outliers=False, **kwargs)
            self._do_make(df, llm, reject_outliers=True, **kwargs)

    @staticmethod
    def _get_metric_kwargs(color_col: str, metric_id: MetricID) -> dict[str, Any]:
        if metric_id.metric.uses_aggregator():
            sub_label = f": {metric_id.agg.value.lower()}(logprobs)"
        else:
            sub_label = ""
        return dict(
            y_col="Diff",
            y_axis_sub_title=f"{metric_id.metric.value.title()}{sub_label}",
            file_base_name=metric_as_string(metric_id, add_agg=True),
            color_col=color_col,
        )

    @staticmethod
    def _get_paired_kwargs(color_col: str, paired_id: PairedMetricID) -> dict[str, Any]:
        sub_label = f": {paired_id.agg.value.lower()}(logprobs)"
        return dict(
            y_col="PairedMetric",
            y_axis_sub_title=f"{paired_id.metric.value.title()}{sub_label}",
            file_base_name=paired_metric_as_string(paired_id, add_agg=True),
            color_col=color_col,
        )

    def _reject_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if self.outliers <= 0:
            return df
        return df[compute_outlier_mask(np.array(df[col]), self.outliers)]

    def _do_make(
        self,
        df: pd.DataFrame,
        llm: Nickname,
        reject_outliers: bool,
        y_col: str,
        y_axis_sub_title: str,
        file_base_name: str,
        color_col: str,
    ):
        fig = px.violin(
            data_frame=self._reject_outliers(df, y_col) if reject_outliers else df,
            y=y_col,
            color=color_col,
            box=True,
            category_orders=all_category_orders(),
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            # range_x and range_y might need updating for a camera ready version.
        )

        fig.update_layout(
            xaxis=dict(title="ACCORD Subset", showticklabels=False),
            yaxis=dict(title=f"Paired diff in {self.which_type} of {y_axis_sub_title}"),
            margin=dict(l=0, r=0, b=60, t=0),
        )
        fig.add_hline(y=0.0, opacity=0.5, line_width=2, line_dash="dash")
        fig.update_traces(meanline_visible=True)
        fig.update_xaxes(showticklabels=False)  # TODO: This isn't doing anything?
        sub_type = "Diff-No-Outliers" if reject_outliers else "Diff-All"
        sub_type = f"{self.which_type}-{sub_type}-{color_col}"
        path = make_path(self.plots_dir, "violin", llm, sub_type)
        save_figure(fig, path, file_base_name)


class BinaryViolinMaker:
    def __init__(
        self,
        plots_dir: str,
        data: DataLoader,
        outlier_threshold: float,
        binary_column_name: str,
        binary_options: tuple[Any, Any],
    ):
        self.plots_dir = plots_dir
        self.data = data
        self.outliers = outlier_threshold
        self.binary = binary_column_name
        self.options = binary_options

    def make_metric_select(
        self, llm: Nickname, color_col: str, selection: list[MetricID], restrict: bool
    ) -> None:
        for metric_id in selection:
            kw = {"restrict_to_opposites_only_in_col": self.binary} if restrict else {}
            df = self.data.make_metric_df(llm, metric_id, **kw)
            df["Subset"] = df["Subset"].apply(lambda x: AccordSubset(x).name)
            kwargs = self._get_metric_kwargs(color_col, metric_id, restrict)
            self._do_make(df, llm, reject_outliers=False, **kwargs)
            self._do_make(df, llm, reject_outliers=True, **kwargs)

    def _reject_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if self.outliers <= 0:
            return df
        dfs = []
        for option in self.options:
            data = np.array(df[df[self.binary] == option][col])
            mask = compute_outlier_mask(data, self.outliers)
            dfs.append(df[df[self.binary] == option][mask])
        return pd.concat(dfs)

    @staticmethod
    def _get_metric_kwargs(
        color_col: str, metric_id: MetricID, restrict: bool
    ) -> dict[str, Any]:
        if metric_id.metric.uses_aggregator():
            sub_label = f": {metric_id.agg.value.lower()}(logprobs)"
        else:
            sub_label = ""
        return dict(
            y_col="Metric",
            y_axis_title=f"{metric_id.metric.value.title()}{sub_label}",
            file_base_name=metric_as_string(metric_id, add_agg=True),
            x_range=[
                min(s.value for s in AccordSubset) - 1,
                max(s.value for s in AccordSubset) + 1,
            ],
            color_col=color_col,
            restrict=restrict,
        )

    def _do_make(
        self,
        df: pd.DataFrame,
        llm: Nickname,
        reject_outliers: bool,
        y_col: str,
        y_axis_title: str,
        file_base_name: str,
        x_range: tuple[int, int],
        color_col: str,
        restrict: bool,
    ):
        df = self._reject_outliers(df, y_col) if reject_outliers else df
        fig = go.Figure()
        for option in self.options:
            self._add_trace(fig, df, option, color_col, y_col)

        fig.update_layout(
            violingap=0,
            violinmode="overlay",
            xaxis=dict(title=f"ACCORD {color_col}", range=x_range),
            yaxis=dict(title=y_axis_title),
            margin=dict(l=0, r=0, b=60, t=0),
            template="simple_white",  # TODO: For camera ready, Correctness should not have same color as Factuality.
            legend=dict(title=self.binary),
        )
        sub_type = self.binary + ("-No-Outliers" if reject_outliers else "-All")
        sub_type = f"{sub_type}-{color_col}{'-restricted' if restrict else ''}"
        path = make_path(self.plots_dir, "violin", llm, sub_type)
        save_figure(fig, path, file_base_name)

    def _add_trace(
        self, fig, df: pd.DataFrame, option: Any, color_col: str, y_col: str
    ) -> None:
        fig.add_trace(
            go.Violin(
                x=df[color_col][df[self.binary] == option],
                y=df[y_col][df[self.binary] == option],
                box=dict(visible=True),
                legendgroup=option,
                scalegroup=option,
                name=option,
                side="positive" if option == self.options[1] else "negative",
            )
        )


class ScatterplotMaker:
    def __init__(self, plots_dir: str, data: DataLoader):
        self.plots_dir = plots_dir
        self.data = data

    def make_select(
        self,
        llm: Nickname,
        facet_col: str,
        selection_pairs: list[tuple[MetricID, MetricID]],
    ) -> None:
        colors = ["Factuality", "Correctness", "Legend"]
        for pair in selection_pairs:
            df = self.data.make_metric_df(llm, pair[0], pair[1])
            df["Legend"] = df.apply(
                lambda x: x["Factuality"] + ", " + str(x["Correctness"]), axis=1
            )
            for color in colors:
                self._do_make(df, pair[0], pair[1], llm, facet_col, color)

    def _do_make(
        self,
        df: pd.DataFrame,
        metric_id_x: MetricID,
        metric_id_y: MetricID,
        llm: Nickname,
        facet_col: str,
        color: str,
    ) -> None:
        fig = px.scatter(
            data_frame=df,
            x=f"Metric1",
            y=f"Metric2",
            color=color,
            facet_col=facet_col,
            facet_col_wrap=3,
            facet_row_spacing=0.17,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            category_orders=all_category_orders(),
            template="simple_white",
            width=700,  # default is 700
            height=500,  # default is 500
            # range_x and range_y might need updating for a camera ready version.
        )

        if metric_id_x.metric.uses_aggregator():
            sub_label_x = f": {metric_id_x.agg.value.lower()}(logprobs)"
        else:
            sub_label_x = ""
        if metric_id_y.metric.uses_aggregator():
            sub_label_y = f": {metric_id_y.agg.value.lower()}(logprobs)"
        else:
            sub_label_y = ""
        post_process_faceted_plot(
            fig,
            x_label=f"{metric_id_x.metric.value.title()}{sub_label_x}",
            y_label=f"{metric_id_y.metric.value.title()}{sub_label_y}",
        )
        fig.update_traces(marker=dict(opacity=0.25))
        path = make_path(self.plots_dir, "scatterplot", llm, f"{color}-{facet_col}")
        metric_x_name = metric_as_string(metric_id_x, add_agg=True)
        metric_y_name = metric_as_string(metric_id_y, add_agg=True)
        save_figure(fig, path, f"{metric_x_name}_{metric_y_name}")


class BinaryTTest:
    def __init__(
        self,
        data: DataLoader,
        analysis_dir: str,
        p_value_threshold: float,
        min_subsets_passing_threshold: int,
        outlier_threshold: float,
        binary_column_name: str,
        binary_options: tuple[Any, Any],
        test_alternative: str,  # Options are: {'two-sided', 'less', 'greater'}
        include_restricted: bool,
    ):
        self.data = data
        self.analysis_dir = analysis_dir
        self.threshold = p_value_threshold
        self.min_count = min_subsets_passing_threshold
        self.outliers = outlier_threshold
        self.binary = binary_column_name
        self.options = binary_options
        self.test_alternative = test_alternative
        self.include_restricted = include_restricted
        self.results = None

    def run(
        self, llm: Nickname, group_col: str, aggregators: list[AggregatorOption]
    ) -> dict[bool, list[MetricID]]:
        self.results, good_p_values = {}, {}
        for metric_id in MetricID.yield_all(aggregators):
            if self._do_metric_run(llm, group_col, metric_id, False):
                good_p_values.setdefault(False, []).append(metric_id)
            if self.include_restricted:
                if self._do_metric_run(llm, group_col, metric_id, True):
                    good_p_values.setdefault(True, []).append(metric_id)
        path = os.path.join(self.analysis_dir, self.binary, group_col, llm + ".csv")
        pd.DataFrame(self.results).to_csv(ensure_path(path), index=False)
        self.results = None
        return good_p_values

    def _do_metric_run(
        self, llm: Nickname, group_col: str, metric_id: MetricID, restrict: bool
    ) -> bool:
        metric_name = metric_as_string(metric_id, add_agg=True)
        self.results.setdefault("Name", []).append(metric_name)
        self.results.setdefault("Restricted", []).append(restrict)
        # Add dummy value for missing BASELINE subset or reasoning hops.
        if group_col in ["Subset", "ReasoningHops"] and restrict:
            self.results.setdefault(AccordSubset.BASELINE.value, []).append(-1)

        kw = {"restrict_to_opposites_only_in_col": self.binary} if restrict else {}
        df = self.data.make_metric_df(llm, metric_id, **kw)
        count = 0
        for group_id, group_df in df.groupby(by=group_col):
            p_value = self._run_test(group_df, "Metric", restrict)
            if 0 < p_value < self.threshold:
                count += 1
            self.results.setdefault(group_id, []).append(p_value)
        return count >= self.min_count

    def _reject_outliers(self, a, b, paired: bool) -> tuple[list[float], list[float]]:
        if self.outliers <= 0:
            return a, b
        a, b = np.array(a), np.array(b)
        if paired:
            a_mask = b_mask = compute_outlier_mask(a - b, self.outliers)
        else:
            a_mask = compute_outlier_mask(a, self.outliers)
            b_mask = compute_outlier_mask(b, self.outliers)
        return a[a_mask].tolist(), b[b_mask].tolist()

    def _run_test(self, df: pd.DataFrame, col: str, restrict: bool) -> float:
        split_data = {}
        for option, group_df in df.groupby(by=self.binary):
            data = group_df.sort_values(by="GroupID")[col].tolist()
            if option in self.options:
                split_data[option] = data
            else:
                raise ValueError(f"Unknown binary option: {option}")
        for option in self.options:
            if option not in split_data:
                # This means we are missing data for one or both binary options.
                return -1
        a, b = self._reject_outliers(
            a=split_data[self.options[0]],
            b=split_data[self.options[1]],
            paired=restrict,
        )
        return self._run_with_caught_warnings(a, b, restrict)

    def _run_with_caught_warnings(
        self, a: list[float], b: list[float], relative: bool
    ) -> float:
        if relative:
            test = ttest_rel
            test_kwargs = dict(nan_policy="raise", alternative=self.test_alternative)
        else:
            test = ttest_ind
            test_kwargs = dict(
                equal_var=False, nan_policy="raise", alternative=self.test_alternative
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                return test(a=a, b=b, **test_kwargs).pvalue
            except RuntimeWarning as e:
                # This is to catch:
                #  RuntimeWarning: Precision loss occurred in moment calculation due
                #  to catastrophic cancellation. This occurs when the data are nearly
                #  identical. Results may be unreliable.
                # or:
                #  scipy.stats._axis_nan_policy.SmallSampleWarning: One or more sample
                #  arguments is too small; all returned values will be NaN. See
                #  documentation for sample size requirements.
                if "Precision loss occurred in moment calculation" in str(e):
                    return -1
                elif "One or more sample arguments is too small" in str(e):
                    return -1
                else:
                    raise


class DiffTTest:
    def __init__(
        self,
        data: DataLoader,
        analysis_dir: str,
        sub_dir_name: str,
        p_value_threshold: float,
        min_subsets_passing_threshold: int,
        outlier_threshold: float,
        test_alternative: str,  # Options are: {'two-sided', 'less', 'greater'}
        which_type: PairingType,
    ):
        self.data = data
        self.analysis_dir = analysis_dir
        self.sub_name = sub_dir_name
        self.threshold = p_value_threshold
        self.min_count = min_subsets_passing_threshold
        self.outliers = outlier_threshold
        self.test_kwargs = dict(nan_policy="raise", alternative=test_alternative)
        self.results = None
        self.which_type = which_type

    def run(
        self, llm: Nickname, group_col: str, aggregators: list[AggregatorOption]
    ) -> list[PairedMetricID]:
        self.results, good_p_values = {}, []
        for paired_id in PairedMetricID.yield_all(aggregators):
            if self._do_paired_run(llm, group_col, paired_id):
                good_p_values.append(paired_id)
        path = os.path.join(self.analysis_dir, self.sub_name, group_col, llm + ".csv")
        pd.DataFrame(self.results).to_csv(ensure_path(path), index=False)
        self.results = None
        return good_p_values

    def _do_paired_run(
        self, llm: Nickname, group_col: str, paired_id: PairedMetricID
    ) -> bool:
        paired_name = paired_metric_as_string(paired_id, add_agg=True)
        self.results.setdefault("Name", []).append(paired_name)
        # Add dummy value for missing BASELINE subset.
        if group_col == "Subset":
            self.results.setdefault(AccordSubset.BASELINE.value, []).append(-1)

        df = self.data.make_paired_df(llm, paired_id, self.which_type)
        count = 0
        for group_id, group_df in df.groupby(by=group_col):
            diff = self._reject_outliers(group_df["PairedMetric"].tolist())
            p_value = self._run_with_caught_warnings(diff)
            if 0 < p_value < self.threshold:
                count += 1
            self.results.setdefault(group_id, []).append(p_value)
        return count >= self.min_count

    def _reject_outliers(self, diff) -> list[float]:
        if self.outliers <= 0:
            return diff
        diff = np.array(diff)
        return diff[compute_outlier_mask(diff, self.outliers)].tolist()

    def _run_with_caught_warnings(self, diff: list[float]) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                return ttest_1samp(diff, popmean=0, **self.test_kwargs).pvalue
            except RuntimeWarning as e:
                # This is to catch:
                #  RuntimeWarning: Precision loss occurred in moment calculation due
                #  to catastrophic cancellation. This occurs when the data are nearly
                #  identical. Results may be unreliable.
                # or:
                #  scipy.stats._axis_nan_policy.SmallSampleWarning: One or more sample
                #  arguments is too small; all returned values will be NaN. See
                #  documentation for sample size requirements.
                if "Precision loss occurred in moment calculation" in str(e):
                    return -1
                elif "One or more sample arguments is too small" in str(e):
                    return -1
                else:
                    raise


class Analyzer:
    def __init__(self, path: PathConfig, cfg: Config):
        self.analysis_dir = cfg.analysis_dir(path.experiment2_dir)
        self.plots_dir = cfg.plots_dir(path.experiment2_dir)
        self.print = ConditionalPrinter(cfg.verbose)
        self.path = path
        self.cfg = cfg

    def run(self, nickname: Nickname):
        raise NotImplementedError


class AccuracyAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config, data: DataLoader):
        super().__init__(path, cfg)
        self.data = data

    def run(self, nickname: Nickname):
        df = self.data.make_df(nickname)
        for group_col in self.cfg.analysis_groups:
            result = df.groupby(by=[group_col, "Factuality"]).apply(self._get_acc)
            result = result.reset_index().rename(columns={0: "Accuracy"})
            file_path = os.path.join(self.analysis_dir, group_col, nickname + ".csv")
            pd.DataFrame(result).to_csv(ensure_path(file_path), index=False)

    @staticmethod
    def _get_acc(group_df):
        return group_df["Correctness"].value_counts()[True] / len(group_df)


class FactualityAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config, data: DataLoader):
        super().__init__(path, cfg)
        self.histograms = HistogramMaker(self.plots_dir, data)
        self.diff_violins = DiffViolinMaker(
            plots_dir=self.plots_dir,
            data=data,
            outlier_threshold=self.cfg.outlier_threshold,
            which_type="Factuality",
        )
        self.binary_violins = BinaryViolinMaker(
            plots_dir=self.plots_dir,
            data=data,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Factuality",
            binary_options=("Factual", "Anti-Factual"),
        )
        self.binary_t_tests = BinaryTTest(
            data=data,
            analysis_dir=self.analysis_dir,
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Factuality",
            binary_options=("Factual", "Anti-Factual"),
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
            include_restricted=False,
        )
        self.paired_t_tests = DiffTTest(
            data=data,
            analysis_dir=self.analysis_dir,
            sub_dir_name="Factuality-Diff",
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
            which_type="Factuality",
        )

    def run(self, nickname: Nickname):
        for col in self.cfg.analysis_groups:
            self.print(f"        Analyzing group: {col}...")
            self.print("            Running t-tests...")
            binary = self.binary_t_tests.run(nickname, col, self.cfg.aggregators)[False]
            paired = self.paired_t_tests.run(nickname, col, self.cfg.aggregators)
            self.print("            Plotting select histograms...")
            # self.histograms.make_select(nickname, col, binary)
            self.print("            Plotting select diff violins...")
            self.diff_violins.make_metric_select(nickname, col, binary)
            self.diff_violins.make_paired_select(nickname, col, paired)
            self.print("            Plotting select binary violins...")
            self.binary_violins.make_metric_select(nickname, col, binary, False)
            self.print("        Done.")


class CorrectnessAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config, data: DataLoader):
        super().__init__(path, cfg)
        self.diff_violins = DiffViolinMaker(
            plots_dir=self.plots_dir,
            data=data,
            outlier_threshold=self.cfg.outlier_threshold,
            which_type="Correctness",
        )
        self.binary_violins = BinaryViolinMaker(
            plots_dir=self.plots_dir,
            data=data,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Correctness",
            binary_options=(True, False),
        )
        self.binary_t_tests = BinaryTTest(
            data=data,
            analysis_dir=self.analysis_dir,
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Correctness",
            binary_options=(True, False),
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
            include_restricted=True,
        )
        self.paired_t_tests = DiffTTest(
            data=data,
            analysis_dir=self.analysis_dir,
            sub_dir_name="Correctness-Diff",
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
            which_type="Correctness",
        )

    def run(self, nickname: Nickname):
        for col in self.cfg.analysis_groups:
            self.print(f"        Analyzing group: {col}...")
            self.print("            Running t-tests...")
            metrics = self.binary_t_tests.run(nickname, col, self.cfg.aggregators)
            paired = self.paired_t_tests.run(nickname, col, self.cfg.aggregators)
            self.print("            Plotting select diff violins...")
            self.diff_violins.make_metric_select(nickname, col, metrics[True])
            self.diff_violins.make_paired_select(nickname, col, paired)
            self.print("            Plotting select binary violins...")
            self.binary_violins.make_metric_select(nickname, col, metrics[True], True)
            self.binary_violins.make_metric_select(nickname, col, metrics[False], False)
            self.print("        Done.")


# Scatter plot y-options:
#  surprisal_source_all  => probably less likely than target, right?
#  surprisal_target_all
#  surprisal_statement_all
#  surprisal_question_all => most important for exp3
#  surprisal_instance_all => most important for exp3
#  surprisal_choice_matching_accord  => empirically, this is best from t-tests
#  ... as well as all others with label/choice for both CSQA/ACCORD (of which there are A LOT!)
#   => however, we can argue that if statement and question are both no good but instance
#      is good, then that implies either label/choice/both was the important element.
#      (Of course, if instance is good while statement/question is good, can't tell what the source is).

# Scatter plot x-options:
#  surprisal_forced_matching_accord
#  surprisal_forced_matching_csqa  => not relevant for exp3
#  surprisal_forced_not_matching_accord
#  surprisal_forced_not_matching_csqa  => not relevant for exp3
#  surprisal_forced_all
#  surprisal_llm_answer
#  rank_forced_matching_accord  => non-continuous and makes for a poor plot
#  rank_forced_matching_csqa  => not relevant for exp3
#  entropy_forced_all  => interesting in theory, but empirically bad results
#  mass_forced_all  => annoying because it's not "real" mass

# NOTE: Almost all of these options are PER AGGREGATOR, so even MORE in total.
#       -> to minimize, we only do SUM (for short text spans, min ~ sum, and for
#          long text spans, SUM has empirically best t-tests)
#          -> logic is that SUM is 'cumulative' surprisal => indicative of global issue
#          -> whereas MIN is 'shock' surprisal (in any long enough text, at least one
#             token will be surprising)

#  AND on top of that we are repeating these options along 3 colors: Factuality, Correctness, and Both.


# Assuming quite cutthroat, we have 5 y-axis * 4 x-axis * 1 aggregator * 3 colors = 60 plots/llm
class CrossAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config, data: DataLoader):
        super().__init__(path, cfg)
        self.scatter_plots = ScatterplotMaker(self.plots_dir, data)
        self.selection_pairs = self._generate_selection_pairs()

    @classmethod
    def _generate_selection_pairs(cls) -> list[tuple[MetricID, MetricID]]:
        selection_pairs = []
        for x in cls._generate_x_selection():
            for y in cls._generate_y_selection():
                selection_pairs.append((x, y))
        return selection_pairs

    @staticmethod
    def _generate_x_selection():
        for surprisal_sub_sub in SurprisalSubSubType2:
            if "csqa" in surprisal_sub_sub.value.lower():
                continue
            yield MetricID(
                metric=MetricType.SURPRISAL,
                sub_metric=SurprisalSubType.FORCED,
                sub_sub_metric=surprisal_sub_sub,
                agg=AggregatorOption.SUM,
            )
        yield MetricID(
            metric=MetricType.SURPRISAL,
            sub_metric=SurprisalSubType.LLM,
            sub_sub_metric=AnswerType.ANSWER,
            agg=AggregatorOption.SUM,
        )

    @staticmethod
    def _generate_y_selection():
        sst = SurprisalSubType
        for surprisal_sub in [sst.TARGET, sst.STATEMENT, sst.QUESTION, sst.INSTANCE]:
            yield MetricID(
                metric=MetricType.SURPRISAL,
                sub_metric=surprisal_sub,
                sub_sub_metric=SurprisalSubSubType1.TOP_3,
                agg=AggregatorOption.SUM,
            )
        yield MetricID(
            metric=MetricType.SURPRISAL,
            sub_metric=sst.CHOICE,
            sub_sub_metric=SurprisalSubSubType2.MATCHING_ACCORD,
            agg=AggregatorOption.SUM,
        )

    def run(self, nickname: Nickname):
        for col in self.cfg.analysis_groups:
            self.print(f"        Analyzing group: {col}...")
            self.print("            Plotting select scatter plots...")
            self.scatter_plots.make_select(nickname, col, self.selection_pairs)
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
            self.print("    Running accuracy analysis...")
            AccuracyAnalyzer(self.path, self.cfg, self.data).run(nickname)
            self.print("    Running factuality analysis...")
            FactualityAnalyzer(self.path, self.cfg, self.data).run(nickname)
            self.print("    Running correctness analysis...")
            CorrectnessAnalyzer(self.path, self.cfg, self.data).run(nickname)
            self.print("    Running cross analysis...")
            CrossAnalyzer(self.path, self.cfg, self.data).run(nickname)
