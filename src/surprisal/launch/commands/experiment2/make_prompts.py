from coma import command

from ....accord import AccordInstance
from ....core import Label
from ....prompting import IDGenerator, Message, MessageType, PromptData, PromptType
from ....io import ConditionalPrinter, PathConfig, save_dataclass_jsonl

from .base import AccordLoader, AccordSubset, Config


@command(name="exp.2.make.prompts")
class Experiment2MakePrompts:
    def __init__(self, path: PathConfig, experiment2: Config):
        self.path = path
        self.cfg = experiment2
        self.print = ConditionalPrinter(self.cfg.verbose)
        self.id_generator = IDGenerator()

    def run(self):
        for subset in AccordSubset:
            self.print("Processing subset:", subset)
            self.cfg.subset = subset
            prompts = self._make_prompts()
            file_name = self.cfg.prompts_file(self.path.experiment2_dir)
            save_dataclass_jsonl(file_name, *prompts, ensure_ascii=False)
        self.print("Done.")

    def _make_prompts(self) -> list[PromptData]:
        self.print("    Making prompts...")
        prompts = []
        for instance in AccordLoader(self.cfg.subset, self.path, self.cfg).load():
            self.id_generator.next_group_id()  # Reset for each new group.
            labels = instance.meta_data.answer_choices
            prompts.extend([self._do_make_prompt(instance, label) for label in labels])
        self.print("    Done.")
        return prompts

    def _do_make_prompt(
        self, instance: AccordInstance, forced_label: Label
    ) -> PromptData:
        return PromptData(
            messages=[
                Message(MessageType.SYSTEM, "PLACEHOLDER"),
                Message(MessageType.USER, "PLACEHOLDER"),
                Message(MessageType.ASSISTANT, forced_label),
                Message(MessageType.USER, "CAPSTONE"),
            ],
            prompt_type=PromptType.F if instance.is_factual() else PromptType.AF,
            label=forced_label,
            prompt_id=self.id_generator.next_prompt_id(),
            group_id=self.id_generator.next_group_id(no_increment=True),
            additional_data=dict(
                accord_id=instance.meta_data.id,
                csqa_label=instance.csqa_label,
            ),
        )
