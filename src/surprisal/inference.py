from dataclasses import dataclass, replace
from typing import Any
from time import time

from langchain_core.runnables import Runnable, chain
from langchain_core.runnables.base import RunnableEach
from tqdm import tqdm

from .io import ConditionalPrinter, load_dataclass_jsonl, save_dataclass_jsonl
from .llms import LLMsConfig, load_llm
from .parsing import OutputParser
from .prompting import PromptData


@dataclass
class Inference:
    prompt_data: PromptData
    output: Any | None
    derived_data: dict[str, Any] | None
    error_message: str | None


@dataclass
class PromptWrapper:
    """
    Surely, there is a better way to pass args and kwargs to each Runnable in a
    Chain, but I cannot figure it out from the documentation.
    """

    prompt_data: PromptData
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class ParallelInference:
    def __init__(
        self,
        parser: OutputParser,
        llms: LLMsConfig,
        *args,
        **kwargs,
    ):
        self.parser = parser
        self.llm = load_llm(llms, *args, **kwargs)
        self.chain = self._generate_chain()

    def __call__(self, prompts: list[PromptData], *args, **kwargs) -> list[Inference]:
        return self.chain.invoke([PromptWrapper(p, args, kwargs) for p in prompts])

    def _generate_chain(self) -> Runnable:
        @chain
        def runnable(w: PromptWrapper) -> Inference:
            llm_output = self.llm.invoke(w.prompt_data, *w.args, **w.kwargs)
            if llm_output.error_message is not None:
                return Inference(
                    prompt_data=replace(w.prompt_data, messages=None),
                    output=None,
                    derived_data=None,
                    error_message=llm_output.error_message,
                )
            parser_output = self.parser(llm_output.generated_text, *w.args, **w.kwargs)
            return Inference(
                prompt_data=replace(w.prompt_data, messages=None),
                output=parser_output,
                derived_data=llm_output.derived_data,
                error_message=None,
            )

        return RunnableEach(bound=runnable)


class CheckpointIndicator:
    def __init__(self, frequency: float):
        """
        frequency == 0 disables checkpointing altogether.
        0 < frequency < 1 is interpreted as a percentile. For example, frequency==0.1
            means checkpoint every 10th percentile of the data (or 10 times in total).
        frequency >= 1 is interpreted as an interval of seconds. For example,
            frequency == 60 means checkpoint every 60 seconds.
        """
        self.disabled = frequency == 0
        if 0 < frequency < 1:
            self.percentile = frequency
            self.interval = None
        else:
            self.percentile = None
            self.interval = frequency
        self.start_time = None
        self.modulus = None

    def start(self, total: int) -> None:
        self.start_time = None if self.interval is None else time()
        self.modulus = None if self.percentile is None else int(self.percentile * total)

    def do_checkpoint(self, iteration: int) -> bool:
        if self.disabled:
            return False
        if self.percentile is not None:
            if iteration == 0:
                return False
            return iteration % self.modulus == 0
        current_time = time()
        if self.start_time - current_time >= self.interval:
            self.start_time = current_time
            return True
        return False


class CheckpointedParallelInference:
    def __init__(
        self,
        infer: ParallelInference,
        out_file: str,
        batch_size: int,
        verbose: bool,
        frequency: int | float,
    ):
        self.infer = infer
        self.out_file = out_file
        self.batch_size = batch_size
        self.print = ConditionalPrinter(verbose)
        self.indicator = CheckpointIndicator(frequency)
        self.print("Loading checkpoint...")
        if not self.indicator.disabled:
            try:
                # Fails occur when error is *not* None, which we want to exclude from
                # the checkpoint, so that they get retried. Hence, include in checkpoint
                # only if error is None.
                self.checkpoint = [
                    (i.prompt_data.group_id, i.prompt_data.prompt_id)
                    for i in load_dataclass_jsonl(out_file, t=Inference)
                    if i.error_message is None
                ]
            except FileNotFoundError:
                self.checkpoint = []
        else:
            self.checkpoint = []
        self.print("Done.")

    def skip(self, prompt_data: PromptData) -> bool:
        return (prompt_data.group_id, prompt_data.prompt_id) in self.checkpoint

    def __call__(self, prompts: list[PromptData], *args, **kwargs) -> None:
        # Summarize checkpoint skips, if verbose.
        skipped = len(self.checkpoint)
        total = skipped + len(prompts)
        self.print(f"Skipping {skipped}/{total} prompts due to checkpointing.")

        # Make batches.
        size = self.batch_size
        if size > 0:
            prompts = [prompts[i : i + size] for i in range(0, len(prompts), size)]
        else:
            prompts = [prompts]

        self.print("Running inference...")
        inferences = []
        self.indicator.start(len(prompts))
        for i, batch in enumerate(tqdm(prompts) if self.print.condition else prompts):
            inferences.extend(self.infer(batch, *args, **kwargs))
            if self.indicator.do_checkpoint(i):
                self._save(inferences)
                inferences = []
        self._save(inferences)
        self.print("Done.")

    def _save(self, inferences: list[Inference]) -> None:
        mode = "w" if self.indicator.disabled else "a"
        save_dataclass_jsonl(self.out_file, *inferences, mode=mode, ensure_ascii=False)
