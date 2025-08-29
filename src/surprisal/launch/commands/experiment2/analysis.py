#  1. We need code that grabs the right targets from the right statements in logprobs.
#     We have notes in exp1/analysis.py on that.
#     - We also need to be able to grab the surprisal of the labels (A-E) as well as the
#       text of the answer choices (the stuff immediately after the label).
#  2. We need to decide on aggregation for not only the within-target multi-tokens, but
#     also the collection of all targets:
#      - min/max are simple
#      - sum will be tree-size dependent
#      - avg is macro/micro dependent
#      - first/last don't really make sense... maybe last a little but definitely not first

# TODO:
#  3. Regardless of the aggregation in (2), but ESPECIALLY if we use SUM, the analysis
#     should be split by tree size. This is not only because there are way more of some
#     tree sizes than others, but also because we want to see the trend.
#      - Since it doesn't really make sense to try to extract reasoning skills from the
#        mixed trees, we won't do skill-wise analysis. Co-opt the skill-wise analysis
#        from exp1 to show facets by reasoning tree size instead.
#      - Either include tree size 0 (baseline) as a first facet, or else normalize
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
#     b. Physical position (metric: surprisal):
#        - One plot for each of A-E answer choices
#        - possibly double-colored/marked by whether correct or not as in (a)
#     c. Rank (metric: integer 1-5):
#        c.1. Of the CSQA-base label (the factual label)
#        c.2. Of the chosen answer label (the accord label)
#     d. Entropy (metric: shannon entropy):
#        - Over all the answer choices
#     e. Aggregate surprisal (metric: surprisal):
#        - Over all the answer choices


# Histogram:
#  factuality: accord label == csqa_label or not. That is precisely what LogprobData.factuality captures
#  the data: col1=Factuality, col2=specific aggregator's surprisal of specifically the forced label that matches the accord label, col3=subset
#  col1=color, col3=facet, col2=x-data
#  there is no overall, only this facet
# Variants of col2:
#  DONE - surprisal of the answer choice label/term that matches the accord label
#  DONE - rank of specifically the accord label (as well as the answer choice label/term)
#  DONE - entropy over all forced labels, all answer labels, all answer choices
#  DONE - aggregate (for matching aggregator metric) over all forced labels, all answer labels, all answer choices
#  DONE - surprisal of the source/target/collective statement terms
#  DONE - physical position (A-E) of the forced/answer/choice
# DONE: FOR ALL OF THE ABOVE THAT INVOLVE JUST LOOKING AT THE ACCORD-MATCHED LABEL, YOU CAN ALSO
# LOOK AT THE SAME METRIC FOR THE FORCED/ANSWER/CHOICE LABEL MATCHING THE *CSQA LABEL* RATHER THAN THE ACCORD LABEL
# QUESTION: Is there anything besides factuality that makes sense as a colour?
#  - LLM correctness? But how to represent when we have 1 correct and 4 incorrect?

# Scatter plot:
#  First, you can scatter any 2 metrics from the histograms against each other
#  Second, you can scatter any of those same metrics against a new one:
#   - aggregate (for matching aggregator metric) over all forced labels, all answer labels, all answer choices
#     EXCEPT either the correct one (accord label) or the base one (csqa label)
#   - actually, is there any reason this cannot simply be another histogram metric????
import os

from coma import command
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

    def get_query_df(self, llm: Nickname) -> pd.DataFrame:
        # TODO: The idea is that you process each LLM one at a time. For each, you
        #  query/manipulate the given DF to figure out which LogprobData IDs you actually
        #  need, then retrieve those to make a secondary DF.
        return pd.DataFrame(self.data_by_llm[llm.replace("/", "-")])

    def get_data_by_id(self, ids: list[int]) -> list[LogprobDataclass]:
        # TODO: question: should we return the LogprobData (raw) or make a secondary DF?
        #  - Logprob raw can be directly queried for metrics (still needs implementing)
        #  - DF is better for plotly but unknown what fields/metrics are actually needed
        return [self.data_by_id[id_] for id_ in ids]


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
            self._do_make(df, llm, metric_id)

    def make_select(self, llm: Nickname, selection: list[MetricID]) -> None:
        for metric_id in selection:
            df = self.data.make_df(llm, metric_id)
            self._do_make(df, llm, metric_id)

    def _do_make(self, df: pd.DataFrame, llm: Nickname, metric_id: MetricID) -> None:
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


class TTest:
    def __init__(
        self,
        data: DataLoader,
        analysis_dir: str,
        p_value_threshold: float,
        min_subsets_passing_threshold: int,
        binary_column_name: str,
        binary_options: tuple[str, str],
        test_type: str,  # Options are: {'independent', 'relative'}
        test_alternative: str,  # Options are: {'two-sided', 'less', 'greater'}
    ):
        self.data = data
        self.analysis_dir = analysis_dir
        self.threshold = p_value_threshold
        self.min_count = min_subsets_passing_threshold
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
        result = self.test(
            a=split_data[self.options[0]],
            b=split_data[self.options[1]],
            **self.test_kwargs,
        )
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
        self.t_tests = TTest(
            data=data,
            analysis_dir=self.analysis_dir,
            p_value_threshold=self.cfg.p_value_threshold,
            min_subsets_passing_threshold=self.cfg.min_subsets_passing_threshold,
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
