from dataclasses import dataclass, field
from enum import Enum
import os

from coma import command

from ....core import AggregatorOption
from ....io import PathConfig, load_dataclass_jsonl
from ....llms import Nickname
from ....accord import (
    AccordCaseLink,
    AccordInstance,
    AccordMetaData,
    AccordSurfaceForms,
    CsqaBase,
    AccordInstanceSurfacer,
    AccordOrderingSurfacer,
    AccordQADataSurfacer,
    AccordStatementSurfacer,
    AccordTermSurfacer,
    AccordTextSurfacer,
)


class AccordSubset(Enum):
    BASELINE = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

    def get_meta_data_file(self, path: PathConfig) -> str:
        if self == AccordSubset.BASELINE:
            return path.accord_baseline_file
        elif self == AccordSubset.ONE:
            return path.accord_tree_size_1_file
        elif self == AccordSubset.TWO:
            return path.accord_tree_size_2_file
        elif self == AccordSubset.THREE:
            return path.accord_tree_size_3_file
        elif self == AccordSubset.FOUR:
            return path.accord_tree_size_4_file
        elif self == AccordSubset.FIVE:
            return path.accord_tree_size_5_file
        else:
            raise ValueError(f"Unsupported ACCORD subset: {self}")

    def get_reductions_file(self, path: PathConfig) -> str | None:
        return None if self == AccordSubset.BASELINE else path.accord_reductions_file

    def get_prompt_key(self, base_id: str) -> str:
        return base_id + ("-baseline" if self == AccordSubset.BASELINE else "")


@dataclass
class Config:
    subset: AccordSubset = AccordSubset.BASELINE
    prompt_batch_size: int = 10
    checkpoint_frequency: float = 5 * 60  # Save every 5 minutes.
    chosen_only_logprob: bool = True
    trim_inference_logprobs: bool = True
    verbose: bool = False
    invert: bool = False  # Whether to invert the order of statements vs question/answer
    system_prompt_id: str = ""
    system_prompt: dict[str, str] = field(default_factory=dict)
    user_template_id: str = ""
    user_template: dict[str, str] = field(default_factory=dict)
    user_template_indicator: str = "Do not output anything else."
    analysis_llms: list[Nickname] = field(default_factory=list)
    aggregators: list[AggregatorOption] = field(
        default_factory=lambda: [
            AggregatorOption.FIRST,
            AggregatorOption.SUM,
            AggregatorOption.MIN,
        ]
    )
    flip_logprobs: bool = True
    label_count: int = 5

    # Together, these two determine how many of the subsets for a given metric all
    # have to have a p-value below this threshold for it to be considered a strong
    # result across the board. Because of p-hacking, it is best to set this quite
    # low (much less than 0.05). To ignore, either set threshold == 1 or min == 0.
    p_value_threshold: float = 0.001
    min_subsets_passing_threshold: int = 4

    # If greater than 0, removes outliers before t-tests. Closer to 0 means more
    # aggressive cutting.
    outlier_threshold: float = 2.0

    def get_system_prompt(self, subset: AccordSubset | None = None) -> str:
        subset = subset or self.subset
        return self.system_prompt[subset.get_prompt_key(self.system_prompt_id)]

    def get_user_template(self, subset: AccordSubset | None = None) -> str:
        subset = subset or self.subset
        return self.user_template[subset.get_prompt_key(self.user_template_id)]

    def _build_id(
        self,
        subset: bool = True,
        system_prompt: bool = True,
        user_template: bool = True,
    ) -> str:
        items = []
        if subset:
            items.append(str(self.subset.value))
        if system_prompt:
            items.append(self.system_prompt_id)
        if user_template:
            items.append(self.user_template_id)
        return "-".join(items)

    def prompts_file(self, root: str) -> str:
        b_id = self._build_id(system_prompt=False, user_template=False)
        return str(os.path.join(root, "prompts", b_id)) + ".jsonl"

    def llm_output_dir(self, root: str) -> str:
        return str(os.path.join(root, "output", self._build_id()))

    def pre_analysis_dir(self, root: str) -> str:
        return str(os.path.join(root, "pre_analysis", self._build_id(subset=False)))

    def analysis_dir(self, root: str) -> str:
        return str(os.path.join(root, "analysis", self._build_id(subset=False)))

    def plots_dir(self, root: str) -> str:
        return str(os.path.join(root, "plots", self._build_id(subset=False)))


class AccordLoader:
    """
    Loads and then preprocesses ACCORD_CSQA MetaData into ACCORD_CSQA Instances
    matching those used to generate and evaluate ACCORD_CSQA from the ACCORD paper.
    """

    def __init__(self, subset: AccordSubset, path: PathConfig, cfg: Config):
        self.data_file = subset.get_meta_data_file(path)
        self.forms_file = subset.get_reductions_file(path)
        self.csqa_file = path.accord_csqa_file
        self.instruction_prompt = cfg.get_user_template(subset)
        self.invert = cfg.invert
        self.surfacer: AccordInstanceSurfacer | None = None

    def load(self) -> list[AccordInstance]:
        self.surfacer = self._create_surfacer()
        meta_datas = load_dataclass_jsonl(self.data_file, t=AccordMetaData)
        csqa = load_dataclass_jsonl(self.csqa_file, t=CsqaBase)
        csqa = {instance.identifier: instance for instance in csqa}
        return [
            AccordInstance(self.surfacer(md), csqa[md.qa_id].correct_answer_label, md)
            for md in meta_datas
        ]

    def _load_surface_forms(self) -> AccordSurfaceForms:
        """
        Loads and registers CaseLink subsumed surface forms from a CSV-formatted file.

        Empty lines are skipped, as are lines starting with the comment string.

        Relevant headers from the CSV file are:
            "relation1_or_pairing" -> CaseLink.r1_type
            "relation2" -> CaseLink.r2_type
            "case" -> CaseLink.case
            "relation2_surface_form" -> subsumed surface form for associated CaseLink
        """
        line_comment = "#"
        with open(self.forms_file, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            lines = [
                line
                for line in f.readlines()
                if line.strip() and not line.strip().startswith(line_comment)
            ]
        forms = AccordSurfaceForms()
        for r in [dict(zip(header, line.strip().split(","))) for line in lines]:
            cl = AccordCaseLink(
                r["relation1_or_pairing"], r["relation2"], int(r["case"])
            )
            forms.register(cl, r["relation2_surface_form"])
        return forms

    def _create_surfacer(self) -> AccordInstanceSurfacer:
        """
        Creates an InstanceSurfacer with all config parameters matching those used
        to generate and evaluate ACCORD_CSQA from the ACCORD paper.
        """
        forms = None if self.forms_file is None else self._load_surface_forms()
        ordering_surfacer = None
        if forms is not None:
            ordering_surfacer = AccordOrderingSurfacer(
                prefix="Statements:\n",
                statement_separator="\n",
                statement_surfacer=AccordStatementSurfacer(
                    prefix="- ",
                    term_surfacer=AccordTermSurfacer(prefix="[", suffix="]"),
                    forms=forms,
                ),
            )
        return AccordInstanceSurfacer(
            invert=self.invert,
            prefix="",
            surfacer_separator="\n",
            prefix_surfacer=AccordTextSurfacer(
                prefix="Instructions:\n",
                text=self.instruction_prompt,
            ),
            ordering_surfacer=ordering_surfacer,
            qa_data_surfacer=AccordQADataSurfacer(
                prefix="Question:\n",
                question_answer_separator="\n",
                answer_choice_separator="    ",
                answer_choice_formatter="{}: {}",
            ),
            suffix_surfacer=AccordTextSurfacer(prefix="Answer:\n", text=""),
        )


@command(name="exp.2.test.accord.loader")
def cmd(path: PathConfig, experiment2: Config):
    baseline_data = AccordLoader(AccordSubset.BASELINE, path, experiment2).load()
    print(baseline_data[1].text)
    print("Chosen label:", baseline_data[1].meta_data.label)
    print("CSQA label:", baseline_data[1].csqa_label)
    print("Question data:", baseline_data[1].meta_data.question)
    level_1_data = AccordLoader(AccordSubset.ONE, path, experiment2).load()
    print(level_1_data[1].text)
    print("Chosen label:", level_1_data[1].meta_data.label)
    print("CSQA label:", level_1_data[1].csqa_label)
    print("Question data:", level_1_data[1].meta_data.question)
    level_5_data = AccordLoader(AccordSubset.FIVE, path, experiment2).load()
    print(level_5_data[1].text)
    print("Chosen label:", level_5_data[1].meta_data.label)
    print("CSQA label:", level_5_data[1].csqa_label)
    print("Question data:", level_5_data[1].meta_data.question)
