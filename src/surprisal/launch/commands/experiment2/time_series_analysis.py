from typing import Any, Callable, Iterable
import warnings
import os

from coma import command
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ....accord import SerialComponent, SerialOption
from ....io import ConditionalPrinter, PathConfig, load_dataclass_jsonl, ensure_path
from ....llms import Nickname

from .base import AccordSubset, Config
from .pre_analysis import LogprobDataclass
from .analysis import (
    Analyzer,
    RelativeType,
    all_category_orders,
    as_factuality_diff,
    as_opp_correctness_diff,
    compute_outlier_mask,
    make_path,
    save_figure,
)


class DataLoader:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path, self.cfg = path, experiment2
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.id_counter = 0
        self.data_by_id: dict[int, LogprobDataclass] = {}
        self.data_by_llm: dict[Nickname, dict] = {}
        self.in_dir = self.cfg.pre_analysis_dir(self.path.experiment2_dir)

    def load(self) -> None:
        for nickname in self.cfg.analysis_llms:
            nickname = nickname.replace("/", "-")
            self.print(f"    Loading data for: {nickname}...")
            in_file = os.path.join(self.in_dir, f"{nickname}.jsonl")
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

    def make_abs_df(
        self,
        llm: Nickname,
        component: SerialComponent,
        option_1: SerialOption,
        option_2: SerialOption | None = None,
        restrict_to_opposites_only_in_col: str | None = None,
    ) -> pd.DataFrame:
        df = self.make_df(llm)
        if restrict_to_opposites_only_in_col is not None:
            df = self._keep_opposites_only(df, restrict_to_opposites_only_in_col)
        col_name = "Metric" if option_2 is None else "Metric1"
        df[col_name] = df["DataID"].apply(
            lambda id_: self.data_by_id[id_]
            .metrics.serial_metrics.get(component)
            .get(option_1)
        )
        if option_2 is not None:
            df["Metric2"] = df["DataID"].apply(
                lambda id_: self.data_by_id[id_]
                .metrics.serial_metrics.get(component)
                .get(option_2)
            )
        return df

    def make_rel_df(
        self,
        llm: Nickname,
        component: SerialComponent,
        option: SerialOption,
        which_type: RelativeType,
    ) -> pd.DataFrame:
        df = self.make_df(llm)
        df["RelMetric"] = df["DataID"].apply(
            self._get_rel_fn(component, option, which_type)
        )
        # BASELINE doesn't have any relative data.
        # For others, F->data and AF->None. This is just a storage convention. That is,
        # the data is always attached to the factual item in the pair, regardless of
        # which RelativeType is involved
        return df[~df["RelMetric"].isna()]

    def _get_rel_fn(
        self, component: SerialComponent, option: SerialOption, which_type: RelativeType
    ) -> Callable[[int], float | None]:
        def helper(data_id: int):
            if which_type == "Factuality":
                metrics = self.data_by_id[data_id].factuality_metrics
            elif which_type == "True-Correct":
                metrics = self.data_by_id[data_id].true_correctness_metrics
            elif which_type == "False-Correct":
                metrics = self.data_by_id[data_id].false_correctness_metrics
            elif which_type == "Opp-Correct":
                metrics = self.data_by_id[data_id].opposite_correctness_metrics
            else:
                raise ValueError(f"Unsupported relative metric type: {which_type}")
            if metrics is None:
                return None
            return metrics.serial_metrics.get(component).get(option)

        return helper

    @staticmethod
    def _keep_opposites_only(df: pd.DataFrame, col: str) -> pd.DataFrame:
        keep = []
        for group_id, group_df in df.groupby(by="GroupID"):
            if len(group_df[col].unique()) > 1:
                keep.append(group_id)
        return df[df["GroupID"].isin(keep)]


def cartesian(
    components: Iterable[SerialComponent], options: Iterable[SerialOption]
) -> Iterable[tuple[SerialComponent, SerialOption]]:
    for c in components:
        for o in options:
            yield c, o


class DiffViolinMaker:
    def __init__(
        self,
        plots_dir: str,
        data: DataLoader,
        outlier_threshold: float,
        which_type: RelativeType,
    ):
        self.plots_dir = plots_dir
        self.data = data
        self.outliers = outlier_threshold
        self.which_type = which_type

    def make_abs_select(
        self,
        llm: Nickname,
        color_col: str,
        components: list[SerialComponent],
        options: list[SerialOption],
    ) -> None:
        if self.which_type == "Factuality":
            diff_fn = as_factuality_diff
        elif self.which_type == "Opp-Correct":
            diff_fn = as_opp_correctness_diff
        else:
            raise ValueError(f"Unsupported metric type: {self.which_type}")
        for component, option in cartesian(components, options):
            df = diff_fn(self.data.make_abs_df(llm, component, option), color_col)
            if df.empty:
                # Skip when data is missing.
                continue
            if color_col == "Subset":
                df = df[df["Subset"] != AccordSubset.BASELINE.value]
            kwargs = self._get_abs_kwargs(color_col, component, option)
            self._do_make(df, llm, reject_outliers=False, **kwargs)
            self._do_make(df, llm, reject_outliers=True, **kwargs)

    def make_rel_select(
        self,
        llm: Nickname,
        color_col: str,
        components: list[SerialComponent],
        options: list[SerialOption],
    ) -> None:
        for component, option in cartesian(components, options):
            df = self.data.make_rel_df(llm, component, option, self.which_type)
            if df.empty:
                # Skip when data is missing.
                continue
            kwargs = self._get_rel_kwargs(color_col, component, option)
            self._do_make(df, llm, reject_outliers=False, **kwargs)
            self._do_make(df, llm, reject_outliers=True, **kwargs)

    @staticmethod
    def _get_abs_kwargs(
        color_col: str, component: SerialComponent, option: SerialOption
    ) -> dict[str, Any]:
        return dict(
            y_col="Diff",
            y_axis_sub_title=f"{option.value.title()} of {component.value.title()}",
            file_base_name=f"{component.as_attribute()}-{option.as_attribute()}",
            color_col=color_col,
        )

    @staticmethod
    def _get_rel_kwargs(
        color_col: str, component: SerialComponent, option: SerialOption
    ) -> dict[str, Any]:
        return dict(
            y_col="RelMetric",
            y_axis_sub_title=f"{option.value.title()} of {component.value.title()}",
            file_base_name=f"{component.as_attribute()}-{option.as_attribute()}",
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
            xaxis=dict(title=f"ACCORD {color_col}", showticklabels=False),
            yaxis=dict(
                title=f"Relative difference in {self.which_type} of {y_axis_sub_title}"
            ),
            margin=dict(l=0, r=0, b=60, t=0),
        )
        fig.add_hline(y=0.0, opacity=0.5, line_width=2, line_dash="dash")
        fig.update_traces(meanline_visible=True)
        fig.update_xaxes(showticklabels=False)  # TODO: This isn't doing anything?
        sub_type = "Diff-No-Outliers" if reject_outliers else "Diff-All"
        sub_type = f"{self.which_type}-{sub_type}-{color_col}"
        path = make_path(self.plots_dir, "serial-violin", llm, sub_type)
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

    def make_abs_select(
        self,
        llm: Nickname,
        color_col: str,
        restrict: bool,
        components: list[SerialComponent],
        options: list[SerialOption],
    ) -> None:
        for component, option in cartesian(components, options):
            kw = {"restrict_to_opposites_only_in_col": self.binary} if restrict else {}
            df = self.data.make_abs_df(llm, component, option, **kw)
            df["Subset"] = df["Subset"].apply(lambda x: AccordSubset(x).name)
            kwargs = self._get_metric_kwargs(color_col, component, option, restrict)
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
        color_col: str, component: SerialComponent, option: SerialOption, restrict: bool
    ) -> dict[str, Any]:
        return dict(
            y_col="Metric",
            y_axis_title=f"{option.value.title()} of {component.value.title()}",
            file_base_name=f"{component.as_attribute()}-{option.as_attribute()}",
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
        if df[y_col].isnull().all():
            # Return early if data is missing.
            return
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
        path = make_path(self.plots_dir, "serial-violin", llm, sub_type)
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


class AbsoluteTTest:
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
        which_components: list[SerialComponent],
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
        self.sub_name = f"Serial-{self.binary}"
        self.which_components = which_components

    def run(
        self, llm: Nickname, group_col: str
    ) -> dict[bool, tuple[list[SerialComponent], list[SerialOption]]]:
        self.results, good_p_values = {}, {True: ([], []), False: ([], [])}
        for component, option in cartesian(self.which_components, SerialOption):
            if self._do_abs_run(llm, group_col, component, option, False):
                good_p_values[False][0].append(component)
                good_p_values[False][1].append(option)
            if self.include_restricted:
                if self._do_abs_run(llm, group_col, component, option, True):
                    good_p_values[True][0].append(component)
                    good_p_values[True][1].append(option)
        path = os.path.join(self.analysis_dir, self.sub_name, group_col, llm + ".csv")
        pd.DataFrame(self.results).to_csv(ensure_path(path), index=False)
        self.results = None
        return good_p_values

    def _do_abs_run(
        self,
        llm: Nickname,
        group_col: str,
        component: SerialComponent,
        option: SerialOption,
        restrict: bool,
    ) -> bool:
        self.results.setdefault("Component", []).append(component.as_attribute())
        self.results.setdefault("Metric", []).append(option.as_attribute())
        self.results.setdefault("Restricted", []).append(restrict)
        # Add dummy value for missing BASELINE subset or reasoning hops.
        if group_col in ["Subset", "ReasoningHops"] and restrict:
            self.results.setdefault(AccordSubset.BASELINE.value, []).append(-1)

        kw = {"restrict_to_opposites_only_in_col": self.binary} if restrict else {}
        df = self.data.make_abs_df(llm, component, option, **kw)
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
        try:
            a, b = self._reject_outliers(
                a=split_data[self.options[0]],
                b=split_data[self.options[1]],
                paired=restrict,
            )
        except TypeError:
            return -1
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


class RelativeTTest:
    def __init__(
        self,
        data: DataLoader,
        analysis_dir: str,
        sub_dir_name: str,
        p_value_threshold: float,
        min_subsets_passing_threshold: int,
        outlier_threshold: float,
        test_alternative: str,  # Options are: {'two-sided', 'less', 'greater'}
        which_type: RelativeType,
        which_components: list[SerialComponent],
    ):
        self.data = data
        self.analysis_dir = analysis_dir
        self.sub_name = f"Serial-{sub_dir_name}"
        self.threshold = p_value_threshold
        self.min_count = min_subsets_passing_threshold
        self.outliers = outlier_threshold
        self.test_kwargs = dict(nan_policy="raise", alternative=test_alternative)
        self.results = None
        self.which_type = which_type
        self.which_components = which_components

    def run(
        self, llm: Nickname, group_col: str
    ) -> tuple[list[SerialComponent], list[SerialOption]]:
        self.results, good_p_values = {}, ([], [])
        for component, option in cartesian(self.which_components, SerialOption):
            if self._do_rel_run(llm, group_col, component, option):
                good_p_values[0].append(component)
                good_p_values[1].append(option)
        path = os.path.join(self.analysis_dir, self.sub_name, group_col, llm + ".csv")
        pd.DataFrame(self.results).to_csv(ensure_path(path), index=False)
        self.results = None
        return good_p_values

    def _do_rel_run(
        self,
        llm: Nickname,
        group_col: str,
        component: SerialComponent,
        option: SerialOption,
    ) -> bool:
        self.results.setdefault("Component", []).append(component.as_attribute())
        self.results.setdefault("Metric", []).append(option.as_attribute())
        # Add dummy value for missing BASELINE subset.
        if group_col == "Subset":
            self.results.setdefault(AccordSubset.BASELINE.value, []).append(-1)

        df = self.data.make_rel_df(llm, component, option, self.which_type)
        count = 0
        for group_id, group_df in df.groupby(by=group_col):
            diff = self._reject_outliers(group_df["RelMetric"].tolist())
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
                # or:
                #  RuntimeWarning: divide by zero encountered in divide
                if "Precision loss occurred in moment calculation" in str(e):
                    return -1
                elif "One or more sample arguments is too small" in str(e):
                    return -1
                elif "divide by zero encountered in divide" in str(e):
                    return -1
                else:
                    raise


class FactualityAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config, data: DataLoader):
        super().__init__(path, cfg)
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
        self.abs_t_tests = AbsoluteTTest(
            data=data,
            analysis_dir=self.analysis_dir,
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Factuality",
            binary_options=("Factual", "Anti-Factual"),
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
            include_restricted=False,
            which_components=self.cfg.serial_components,
        )
        self.rel_t_tests = RelativeTTest(
            data=data,
            analysis_dir=self.analysis_dir,
            sub_dir_name="Factuality-Diff",
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
            which_type="Factuality",
            which_components=self.cfg.serial_components,
        )

    def run(self, nickname: Nickname):
        for col in self.cfg.analysis_groups:
            self.print(f"        Analyzing group: {col}...")
            self.print("            Running t-tests...")
            abs_, which = self.abs_t_tests.run(nickname, col)[False], "select"
            if not self.cfg.plot_select:
                abs_, which = (list(SerialComponent), list(SerialOption)), "all"
            rel = self.rel_t_tests.run(nickname, col)
            if not self.cfg.plot_select:
                rel = (list(SerialComponent), list(SerialOption))
            self.print(f"            Plotting {which} diff violins...")
            self.diff_violins.make_abs_select(nickname, col, *abs_)
            self.diff_violins.make_rel_select(nickname, col, *rel)
            self.print(f"            Plotting {which} binary violins...")
            self.binary_violins.make_abs_select(nickname, col, False, *abs_)
            self.print("        Done.")


class CorrectnessAnalyzer(Analyzer):
    def __init__(self, path: PathConfig, cfg: Config, data: DataLoader):
        super().__init__(path, cfg)
        # Type hint needed to satisfy the type checker.
        self.opts: list[RelativeType] = ["True-Correct", "False-Correct", "Opp-Correct"]

        self.diff_violins = {}
        for which_type in self.opts:
            self.diff_violins[which_type] = DiffViolinMaker(
                plots_dir=self.plots_dir,
                data=data,
                outlier_threshold=self.cfg.outlier_threshold,
                which_type=which_type,
            )
        self.binary_violins = BinaryViolinMaker(
            plots_dir=self.plots_dir,
            data=data,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Correctness",
            binary_options=(True, False),
        )
        self.abs_t_tests = AbsoluteTTest(
            data=data,
            analysis_dir=self.analysis_dir,
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
            outlier_threshold=self.cfg.outlier_threshold,
            binary_column_name="Correctness",
            binary_options=(True, False),
            test_alternative="less" if self.cfg.flip_logprobs else "greater",
            include_restricted=True,
            which_components=self.cfg.serial_components,
        )
        self.rel_t_tests = {}
        for which_type in self.opts:
            self.rel_t_tests[which_type] = RelativeTTest(
                data=data,
                analysis_dir=self.analysis_dir,
                sub_dir_name=which_type,
                p_value_threshold=self.cfg.p_value_threshold,
                min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
                outlier_threshold=self.cfg.outlier_threshold,
                test_alternative="less" if self.cfg.flip_logprobs else "greater",
                which_type=which_type,
                which_components=self.cfg.serial_components,
            )

    def run(self, nickname: Nickname):
        restricted_correctness = self.diff_violins["Opp-Correct"]
        for col in self.cfg.analysis_groups:
            self.print(f"        Analyzing group: {col}...")
            self.print("            Running absolute t-tests...")
            abs_, which = self.abs_t_tests.run(nickname, col), "select"
            if not self.cfg.plot_select:
                abs_ = {
                    True: (list(SerialComponent), list(SerialOption)),
                    False: (list(SerialComponent), list(SerialOption)),
                }
                which = "all"
            self.print(f"            Plotting {which} absolute diff violins...")
            restricted_correctness.make_abs_select(nickname, col, *abs_[True])
            self.print(f"            Plotting {which} absolute binary violins...")
            self.binary_violins.make_abs_select(nickname, col, True, *abs_[True])
            self.binary_violins.make_abs_select(nickname, col, False, *abs_[False])
            for opt in self.opts:
                self.print(f"            Analyzing relative metric: {opt}")
                self.print("                Running relative t-tests...")
                rel = self.rel_t_tests[opt].run(nickname, col)
                if not self.cfg.plot_select:
                    rel = (list(SerialComponent), list(SerialOption))
                self.print(f"                Plotting {which} relative diff violins...")
                self.diff_violins[opt].make_rel_select(nickname, col, *rel)


@command(name="exp.2.serial.analysis")
class Experiment2SerialAnalysis:
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
            self.print("    Running correctness analysis...")
            CorrectnessAnalyzer(self.path, self.cfg, self.data).run(nickname)
