import os

from coma import InvocationData, command, preload
from tqdm import tqdm
import pandas as pd

from ....inference import CheckpointedParallelInference, Inference, ParallelInference
from ....io import ConditionalPrinter, PathConfig, load_dataclass_jsonl, walk_files
from ....llms import DummyConfig, LLMsConfig, LLMImplementation, Nickname, OpenAIConfig
from ....parsing import NoOutputParser
from ....prompting import PromptData

from .base import AccordLoader, Config


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
    name="exp.2.infer",
    pre_config_hook=pre_config_hook,
    dummy=DummyConfig,
    openai=OpenAIConfig,
)
class Experiment2Infer:
    def __init__(
        self,
        path: PathConfig,
        experiment2: Config,
        llms: LLMsConfig,
        llm_cfg_placeholder,
    ):
        self.path = path
        self.cfg = experiment2
        self.llms = llms
        self.print = ConditionalPrinter(self.cfg.verbose)
        out = self.cfg.llm_output_dir(self.path.experiment2_dir)
        self.out_file = os.path.join(out, f"{self.llms.llm.replace('/', '-')}.jsonl")
        self.system_prompt = self.cfg.system_prompt[self.cfg.subset.get_prompt_key()]
        trim_indicator = None
        if self.cfg.trim_inference_logprobs:
            trim_indicator = self.cfg.user_template_indicator
        self.infer = CheckpointedParallelInference(
            infer=ParallelInference(
                parser=NoOutputParser(),
                llms=llms,
                llm_cfg=llm_cfg_placeholder,  # Passed to all LLMs' init.
                out_dir=out,  # Passed to OpenaiLLM's init.
                chosen_only=self.cfg.chosen_only_logprob,  # Passed to OpenaiLLM's init.
                trim_indicator=trim_indicator,  # Passed to OpenaiLLM's init.
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
        instances = AccordLoader(self.cfg.subset, self.path, self.cfg).load()
        instances = {instance.meta_data.id: instance.text for instance in instances}
        prompts_file = self.cfg.prompts_file(self.path.experiment2_dir)
        for prompt_data in load_dataclass_jsonl(prompts_file, t=PromptData):
            if self.infer.skip(prompt_data):
                continue
            text = instances[prompt_data.additional_data["accord_id"]]
            prompt_data.messages[0].text = self.system_prompt
            prompt_data.messages[1].text = text
            prompts.append(prompt_data)
        self.print("Done.")
        return prompts

    def run(self):
        self.infer(self.prompts, add_prompt_logprobs=True)


@command(name="exp.2.count.errors")
class CountErrors:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path, self.cfg = path, experiment2
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
        data, in_dir = {}, os.path.join(self.path.experiment2_dir, "output")
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
