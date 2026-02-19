from enum import Enum
import os

from coma import command
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ....accord import SerialComponent, SerialMetrics
from ....io import ConditionalPrinter, PathConfig, load_dataclass_jsonl
from ....llms import Nickname

from .base import AccordSubset, Config
from .pre_analysis import LogprobDataclass, FACTUAL, ANTI_FACTUAL
from .analysis import make_path, post_process_faceted_plot, save_figure

from kaleido._kaleido_tab import KaleidoError  # noqa


class SerialType(Enum):
    F_ABS = "Absolute Factual"
    AF_ABS = "Absolute Anti-Factual"
    FACT_REL = "Relative Factuality"
    CORR_TRUE = "True-Only Correctness"
    CORR_FALSE = "False-Only Correctness"
    CORR_OPP = "Opposites Correctness"

    def get_matching_serial_metric(
        self, component: SerialComponent, data: LogprobDataclass
    ) -> SerialMetrics | None:
        if self in [SerialType.F_ABS, SerialType.AF_ABS]:
            metrics = data.metrics
        elif self == SerialType.FACT_REL:
            metrics = data.factuality_metrics
        elif self == SerialType.CORR_TRUE:
            metrics = data.true_correctness_metrics
        elif self == SerialType.CORR_FALSE:
            metrics = data.false_correctness_metrics
        elif self == SerialType.CORR_OPP:
            metrics = data.opposite_correctness_metrics
        else:
            raise ValueError(f"Unsupported: {self}")
        return None if metrics is None else metrics.serial_metrics.get(component)


class AnomalyType(Enum):
    OUTLIERS = "Outliers"
    CPS = "Change Points"


class AnomaliesTracker:
    def __init__(self):
        self.anomalies = {}

    def add(
        self,
        component: SerialComponent,
        f: LogprobDataclass,
        af: LogprobDataclass | None,
    ) -> None:
        metrics = {}
        for t in SerialType:
            data = af if t == SerialType.AF_ABS else f
            if data is None:
                continue
            metric = t.get_matching_serial_metric(component, data)
            if metric is None:
                continue
            metrics[t] = metric
        row_data = self.anomalies.setdefault(self._get_row(f.subset), {})
        outliers_data = row_data.setdefault(AnomalyType.OUTLIERS, {})
        cp_data = row_data.setdefault(AnomalyType.CPS, {})
        self._do_add(metrics, "outlier_indices", col_data=outliers_data)
        self._do_add(metrics, "change_indices", col_data=cp_data)

    @staticmethod
    def _get_row(subset: AccordSubset):
        return len(AccordSubset) - subset.value

    @staticmethod
    def _get_col(t: AnomalyType):
        if t == AnomalyType.OUTLIERS:
            return 1
        elif t == AnomalyType.CPS:
            return 2
        else:
            raise ValueError(f"Unsupported: {t}")

    @staticmethod
    def _do_add(
        metrics: dict[SerialType, SerialMetrics],
        indices_attribute: str,
        col_data: dict[SerialType, dict[str, int | float]],
    ):
        for t, metric in metrics.items():
            indices = getattr(metric, indices_attribute)
            if indices is None or metric.logprobs is None:
                continue
            col_data[t] = dict(x=indices, y=[metric.logprobs[i] for i in indices])

    def get(
        self, subset: AccordSubset, anomaly: AnomalyType, type_: SerialType
    ) -> dict[str, int | float] | None:
        row, col = self._get_row(subset), self._get_col(anomaly)
        x_y: dict | None = self.anomalies.get(row, {}).get(anomaly, {}).get(type_, None)
        if x_y is None:
            return None
        return dict(row=row, col=col, **x_y)


class DataLoader:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path, self.cfg = path, experiment2
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.data_by_llm: dict[Nickname, dict[SerialComponent, dict]] = {}
        self.anomalies_by_llm: dict[
            Nickname, dict[SerialComponent, AnomaliesTracker]
        ] = {}
        self.skip_tracker: dict[Nickname, dict[AccordSubset, set[str]]] = {}
        self.last_added: dict[Nickname, dict[AccordSubset, str]] = {}
        self.in_dir = self.cfg.pre_analysis_dir(self.path.experiment2_dir)

    def load(self) -> None:
        for nickname in self.cfg.analysis_llms:
            nickname = nickname.replace("/", "-")
            self.print(f"    Loading data for: {nickname}...")
            self._load_for_llm(nickname)

    def _load_for_llm(self, llm: Nickname):
        pairs, in_file = {}, os.path.join(self.in_dir, f"{llm}.jsonl")
        for data in load_dataclass_jsonl(in_file, t=LogprobDataclass):
            if self._skip(llm, data):
                continue
            pairs.setdefault(data.accord_group_id, []).append(data)
        for pair in pairs.values():
            if len(pair) == 1:
                f, af = pair[0], None
            else:
                f = pair[0] if pair[0].factuality == FACTUAL else pair[1]
                af = pair[0] if pair[0].factuality == ANTI_FACTUAL else pair[1]
            for component in self.cfg.serial_components:
                self._add_to_data(llm, component, f, af)
                self._add_to_anomalies(llm, component, f, af)

    def _skip(self, llm: Nickname, data: LogprobDataclass):
        # If negative, skip nothing.
        if self.cfg.serial_visualization_instance == 0:
            return False

        # Add ACCORD GroupID to set of seen instances.
        seen = self.skip_tracker.setdefault(llm, {}).setdefault(data.subset, set())
        seen.add(data.accord_group_id)

        # Once we have exactly the correct number of instances, that's our instance!
        if len(seen) == self.cfg.serial_visualization_instance:
            # Record it's GroupID so that it's Group pair (F or AF) can also be found.
            self.last_added.setdefault(llm, {})[data.subset] = data.accord_group_id
            return False
        elif data.accord_group_id == self.last_added.get(llm, {})[data.subset]:
            # This is the Group pair!
            return False
        else:
            return True

    def _add_to_data(
        self,
        llm: Nickname,
        component: SerialComponent,
        f: LogprobDataclass,
        af: LogprobDataclass | None,
    ):
        lps, size = self._gather_lps(component, f, af)
        data_dict = self.data_by_llm.setdefault(llm, {}).setdefault(component, {})
        data_dict.setdefault("Tokens", []).extend(list(range(size)) * 2)
        data_dict.setdefault("Anomaly", []).extend(
            [AnomalyType.OUTLIERS.value] * size + [AnomalyType.CPS.value] * size
        )
        data_dict.setdefault("Subset", []).extend([f.subset.value] * size * 2)
        for lp_type, lp in lps.items():
            data_dict.setdefault(lp_type.value, []).extend(lp + lp)

    def _gather_lps(
        self,
        component: SerialComponent,
        f: LogprobDataclass,
        af: LogprobDataclass | None,
    ) -> tuple[dict[SerialType, list[float | None]], int]:
        lps = {}
        for t in SerialType:
            data = af if t == SerialType.AF_ABS else f
            lps[t] = self._get_logprobs(component, t, data)
        max_len = max(len(lp) for lp in lps.values())
        if max_len == 0:
            return {t: [None] for t in SerialType}, 1
        return {t: lp + [None] * (max_len - len(lp)) for t, lp in lps.items()}, max_len

    @staticmethod
    def _get_logprobs(
        component: SerialComponent, type_: SerialType, data: LogprobDataclass | None
    ) -> list[float | None]:
        if data is None:
            return []
        serial_metrics = type_.get_matching_serial_metric(component, data)
        if serial_metrics is None or serial_metrics.logprobs is None:
            return []
        return serial_metrics.logprobs

    def _add_to_anomalies(
        self,
        llm: Nickname,
        component: SerialComponent,
        f: LogprobDataclass,
        af: LogprobDataclass | None,
    ) -> None:
        if self.cfg.serial_visualization_instance > 0:
            llm_anomalies = self.anomalies_by_llm.setdefault(llm, {})
            tracker = llm_anomalies.setdefault(component, AnomaliesTracker())
            tracker.add(component, f, af)

    def get(
        self, llm: Nickname, component: SerialComponent
    ) -> tuple[pd.DataFrame, AnomaliesTracker | None]:
        if self.cfg.serial_visualization_instance > 0:
            anomalies = self.anomalies_by_llm[llm][component]
        else:
            anomalies = None
        return pd.DataFrame(self.data_by_llm[llm][component]), anomalies


class SerialPlotter:
    def __init__(self, plots_dir: str, data: DataLoader):
        self.plots_dir = plots_dir
        self.data = data

    def plot(self, llm: Nickname, component: SerialComponent) -> None:
        s = SerialType
        for t in SerialType:
            if t == SerialType.F_ABS:
                self._do_plot(llm, component, [t, SerialType.AF_ABS])
            elif t == SerialType.AF_ABS:
                continue
            elif t in [s.FACT_REL, s.CORR_TRUE, s.CORR_FALSE, s.CORR_OPP]:
                self._do_plot(llm, component, [t])
            else:
                raise ValueError(f"Unsupported: {t}")

    def _do_plot(
        self, llm: Nickname, component: SerialComponent, types: list[SerialType]
    ):
        df, anomalies = self.data.get(llm, component)
        fig = px.line(
            df,
            x="Tokens",
            y=[t.value for t in types],
            facet_col="Anomaly",
            facet_row="Subset",
            facet_row_spacing=0.03,  # default is 0.07
            facet_col_spacing=0.03,  # default is 0.03
            template="simple_white",
        )
        if anomalies is not None:
            self._add_anomaly_legend(fig, anomalies, types)
            self._add_anomalies(fig, anomalies, types)

            # Replace facet names: "Anomaly=value" -> "<b>value</b>"
            fig.for_each_annotation(
                lambda a: a.update(
                    text=(
                        f"<b>{a.text.replace(f'Anomaly=', '')}</b>"
                        if "Anomaly" in a.text
                        else a.text
                    ),
                ),
            )

        post_process_faceted_plot(fig, x_label="Token Position", y_label="Logprob")
        fig.update_layout(legend=dict(title="Legend"))
        path = make_path(self.plots_dir, "serial-viz", llm, component.value.title())
        save_figure(fig, path, types[0].value.title(), width=4000, height=2000)

    def _add_anomaly_legend(
        self, fig, anomalies: AnomaliesTracker, types: list[SerialType]
    ):
        # This goes first. It plots a slightly too small circle UNDERNEATH a real
        # circle. This forces the item to show in the legend as a gray circle as a
        # stand-in for all the other traces. Setting 'visible="legendonly"' doesn't
        # work (opacity in legend is forcibly reduced to 50%) and putting a real trace
        # out of range (with explicit settings for axes ranges) looks bad (removes the
        # nice buffer area).
        data = anomalies.get(AccordSubset.FIVE, AnomalyType.OUTLIERS, types[0])
        if data is None:
            return
        self._do_add_anomalies(
            fig,
            **data,
            showlegend=True,
            name="Anomalies",
            marker=dict(size=15 - 0.1, color="gray"),
        )

    def _add_anomalies(self, fig, anomalies: AnomaliesTracker, types: list[SerialType]):
        for subset in AccordSubset:
            for anomaly in AnomalyType:
                for serial in types:
                    data = anomalies.get(subset, anomaly, serial)
                    if data is None:
                        continue
                    color = "#FF7F0E" if serial == SerialType.AF_ABS else "#1F77B4"
                    kwargs = dict(showlegend=False, marker=dict(size=15, color=color))
                    self._do_add_anomalies(fig, **data, **kwargs)

    @staticmethod
    def _do_add_anomalies(fig, row: int, col: int, x: list, y: list, **kwargs) -> None:
        fig.add_trace(
            trace=go.Scatter(x=x, y=y, mode="markers", **kwargs),
            row=row,
            col=col,
        )


@command(name="exp.2.serial.viz")
class Experiment2SerialVisualization:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path = path
        self.cfg = experiment2
        self.plots_dir = self.cfg.plots_dir(self.path.experiment2_dir)
        self.data = DataLoader(self.path, self.cfg)
        self.print = ConditionalPrinter(self.cfg.verbose)

    def run(self):
        self.print("Loading data...")
        self.data.load()
        self.print("Done.")
        for nickname in self.cfg.analysis_llms:
            self.print("Analyzing results of model:", nickname)
            for component in self.cfg.serial_components:
                SerialPlotter(self.plots_dir, self.data).plot(nickname, component)
