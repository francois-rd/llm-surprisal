import os

from coma import InvocationData, command, preload
from tqdm import tqdm
import pandas as pd

from ....core import ConceptNetFormatter, TermFormatter, Triplet
from ....inference import CheckpointedParallelInference, Inference, ParallelInference
from ....io import ConditionalPrinter, PathConfig, load_dataclass_jsonl, walk_files
from ....llms import DummyConfig, LLMsConfig, LLMImplementation, Nickname, OpenAIConfig
from ....parsing import ParserManager, ParsersConfig
from ....prompting import PromptData

from .base import Config


def pre_config_hook(data: InvocationData) -> None:
    preload(data, "llms")
    llms: LLMsConfig = data.parameters.get_config("llms").get_latest()
    if llms.implementation == LLMImplementation.MISSING:
        raise ValueError("Missing LLM implementation.")
    elif llms.implementation == LLMImplementation.DUMMY:
        llm_cfg_name = "dummy"
        drop_cfg_names = ["openai"]
    elif llms.implementation == LLMImplementation.OPENAI:
        llm_cfg_name = "openai"
        drop_cfg_names = ["dummy"]
    else:
        raise ValueError(f"Unsupported implementation: {llms.implementation}")
    preload(data, llm_cfg_name)
    config = data.parameters.get_config(llm_cfg_name)
    llm_cfg = config.as_primitive(config.get_latest_key())
    data.parameters.replace("llm_cfg_placeholder", llm_cfg)
    data.parameters.delete(*drop_cfg_names)


@command(
    name="exp.1.infer",
    pre_config_hook=pre_config_hook,
    dummy=DummyConfig,
    openai=OpenAIConfig,
)
class Experiment1Infer:
    def __init__(
        self,
        path: PathConfig,
        experiment1: Config,
        parsers: ParsersConfig,
        llms: LLMsConfig,
        llm_cfg_placeholder,
    ):
        self.path = path
        self.cfg = experiment1
        self.llms = llms
        self.print = ConditionalPrinter(self.cfg.verbose)
        out = self.cfg.llm_output_dir(self.path.experiment1_dir)
        self.out_file = os.path.join(out, f"{self.llms.llm.replace('/', '-')}.jsonl")
        self.system_prompt = self.cfg.system_prompt[self.cfg.data_format_method.value]
        self.formatter = ConceptNetFormatter(
            template=self.cfg.user_template,
            method=self.cfg.data_format_method,
            formatter=TermFormatter(language="en"),
        )
        self.infer = CheckpointedParallelInference(
            infer=ParallelInference(
                parser=ParserManager(parsers).get(self.cfg.parser_id),
                llms=llms,
                llm_cfg=llm_cfg_placeholder,
                out_dir=out,
            ),
            out_file=self.out_file,
            batch_size=self.cfg.prompt_batch_size,
            verbose=self.cfg.verbose,
            frequency=self.cfg.checkpoint_frequency,
        )
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> list[PromptData]:
        # Load prompts, skip checkpoints, and replace prompt placeholders.
        self.print("Loading data...")
        prompts = []
        prompts_file = self.cfg.prompts_file(self.path.experiment1_dir)
        for prompt_data in load_dataclass_jsonl(prompts_file, t=PromptData):
            if self.infer.skip(prompt_data):
                continue
            triplet = Triplet(**prompt_data.additional_data["triplet"])
            prompt_data.messages[0].text = self.system_prompt
            prompt_data.messages[1].text = self.formatter(triplet)
            prompts.append(prompt_data)
        self.print("Done.")
        return prompts

    def run(self):
        self.infer(self.prompts)


@command(name="exp.1.count.errors")
class CountErrors:
    def __init__(self, path: PathConfig, experiment1: Config):
        self.path, self.cfg = path, experiment1
        self.print = ConditionalPrinter(self.cfg.verbose)

    def _skip(self, test_nickname: Nickname) -> bool:
        for nickname in self.cfg.analysis_llms:
            if test_nickname == nickname.replace("/", "-"):
                return False
        return True

    @staticmethod
    def _process(inference_path: str) -> tuple[int, int]:
        errors, total = 0, 0
        for inference in load_dataclass_jsonl(inference_path, t=Inference):
            total += 1
            if inference.error_message is not None:
                errors += 1
        return errors, total

    def run(self):
        self.print("Processing...")
        data, in_dir = {}, os.path.join(self.path.experiment1_dir, "output")
        for walk in tqdm(list(walk_files(in_dir))):
            inference_path, nickname = walk.path, walk.no_ext()
            if self._skip(nickname):
                continue
            errors, total = self._process(inference_path)
            data.setdefault("Details", []).append(os.path.split(walk.root)[1])
            data.setdefault("Nickname", []).append(nickname)
            data.setdefault("Errors", []).append(errors)
            data.setdefault("Total", []).append(total)
        self.print("Done.")
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None)
        print(pd.DataFrame(data))
