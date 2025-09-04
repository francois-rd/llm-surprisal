# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Literal
from datetime import datetime
import os

from retry.api import retry_call

from .base import LLM, LLMOutput, Nickname
from ..core import Logprob, Logprobs, RankedLogprob, await_server
from ..io import init_logger
from ..prompting import PromptData, Message, MessageType


class MakeOpenAILogprobs:
    def __init__(
        self,
        nickname: Nickname,
        chosen_only: bool = False,
        trim_indicator: str | None = None,
        **_,
    ):
        self.nickname = nickname
        self.chosen_only = chosen_only
        self.indicator = trim_indicator

    def __call__(
        self, raw_logprobs, raw_format: Literal["vllm", "chat_completion", "completion"]
    ) -> Logprobs | None:
        if raw_logprobs is None:
            return None
        if raw_format == "vllm":
            return self._maybe_trim(self._from_vllm(raw_logprobs))
        elif raw_format == "completion":
            return self._maybe_trim(self._from_completion(raw_logprobs))
        elif raw_format == "chat_completion":
            return self._maybe_trim(self._from_chat(raw_logprobs))
        else:
            raise ValueError(f"Unsupported raw format for logprobs: {raw_format}")

    def _maybe_trim(self, logprobs: Logprobs) -> Logprobs:
        logprobs.maybe_trim(self.indicator)
        return logprobs

    def _from_vllm(self, raw_logprobs) -> Logprobs:
        all_logprobs = []
        for raw_logprob in raw_logprobs:
            if raw_logprob is None:
                # Typically, the first entry is None.
                # Unclear if other entries might also be None, but we skip them
                # regardless as they simply do not give us any actionable data.
                continue
            all_logprobs.append(self._process_vllm_token(raw_logprob))
        return Logprobs(sequence=all_logprobs)

    def _process_vllm_token(self, raw_logprob) -> RankedLogprob:
        chosen, others = None, {}
        for i, lp in enumerate(raw_logprob.values()):
            logprob = Logprob(
                token=self._clean_up_token(lp["decoded_token"]),
                rank=int(lp["rank"]),
                logprob=lp["logprob"],
            )
            if i == 0:
                # The first item is always the chosen one.
                chosen = logprob
            else:
                # There may or may not be extra items beyond the chosen token.
                others[logprob.rank] = logprob
        others = {} if self.chosen_only else others
        return RankedLogprob(chosen, others=others, ranking="absolute")

    def _from_completion(self, raw_logprobs) -> Logprobs | None:
        # raw_logprobs has type: openai.types.completion_choice.Logprobs
        all_logprobs = []
        if raw_logprobs.tokens is None or raw_logprobs.token_logprobs is None:
            return None
        if raw_logprobs.top_logprobs is None:
            raw_logprobs.top_logprobs = [{}] * len(raw_logprobs.tokens)
        for token, logprob, top_logprobs in zip(
            raw_logprobs.tokens, raw_logprobs.token_logprobs, raw_logprobs.top_logprobs
        ):
            ranked = self._process_completion_token(token, logprob, top_logprobs)
            all_logprobs.append(ranked)
        return Logprobs(sequence=all_logprobs)

    def _process_completion_token(
        self, chosen_token: str, chosen_logprob: float, top_logprobs: dict[str, float]
    ) -> RankedLogprob:
        all_logprobs, contains_chosen = [], False
        for other_token, other_logprob in top_logprobs.items():
            if other_token == chosen_token:
                contains_chosen = True
            all_logprobs.append((other_token, other_logprob))
        if not contains_chosen:
            all_logprobs.append((chosen_token, chosen_logprob))
        ranked_chosen, ranked_others = None, {}
        sorted_by_logprob = sorted(all_logprobs, key=lambda lp_: lp_[1], reverse=True)
        for i, (token, logprob) in enumerate(sorted_by_logprob):
            lp = Logprob(token=self._clean_up_token(token), rank=i + 1, logprob=logprob)
            if token == chosen_token:
                ranked_chosen = lp
            else:
                ranked_others[i] = lp
        ranked_others = {} if self.chosen_only else ranked_others
        return RankedLogprob(ranked_chosen, ranked_others, ranking="relative")

    def _from_chat(self, raw_logprobs) -> Logprobs | None:
        # raw_logprobs has type: open.types.chat.chat_completion.ChoiceLogprobs
        # That contains two fields: content and refusal, both optional fields of type:
        #  openai.types.chat.chat_completion_token_logprob.ChatCompletionTokenLogprob
        if raw_logprobs.content is None:
            return None
        all_logprobs = [self._process_chat_token(lp) for lp in raw_logprobs.content]
        return Logprobs(sequence=all_logprobs)

    def _process_chat_token(self, raw_logprob) -> RankedLogprob:
        all_logprobs, contains_chosen = [], False
        for other_logprob in raw_logprob.top_logprobs:
            if other_logprob.token == raw_logprob.token:
                contains_chosen = True
            all_logprobs.append((other_logprob.token, other_logprob.logprob))
        if not contains_chosen:
            all_logprobs.append((raw_logprob.token, raw_logprob.logprob))
        ranked_chosen, ranked_others = None, {}
        sorted_by_logprob = sorted(all_logprobs, key=lambda lp_: lp_[1], reverse=True)
        for i, (token, logprob) in enumerate(sorted_by_logprob):
            lp = Logprob(token=self._clean_up_token(token), rank=i + 1, logprob=logprob)
            if token == raw_logprob.token:
                ranked_chosen = lp
            else:
                ranked_others[i] = lp
        ranked_others = {} if self.chosen_only else ranked_others
        return RankedLogprob(ranked_chosen, ranked_others, ranking="relative")

    def _clean_up_token(self, token: str) -> str:
        if "llama" in self.nickname.lower():
            return token.replace("Ċ", "\n").replace("Ġ", " ")
        if "deepseek" in self.nickname.lower():
            return token.replace("Ċ", "\n").replace("Ġ", " ")
        if "qwen" in self.nickname.lower():
            return token.replace("Ċ", "\n").replace("Ġ", " ")
        if "gemma" in self.nickname.lower():
            return token.replace("▁", " ")  # Bold underscore... Not regular.
        if "mistral" in self.nickname.lower():
            # Unclear if \n becomes a literal <0x0A> or the unicode \u000A, which
            # I believe is identical to \n (i.e., "\u000A" == "\n")
            token = (
                token.replace("<0x0A>", "\n")
                .replace("<\u000A>", "\n")
                .replace("<\n>", "\n")
                .replace("0x0A", "\n")  # Must be placed AFTER <0x0A> to work properly.
            )
            return token.replace("▁", " ")  # Bold underscore... Not regular.
        return token


@dataclass
class _InvocationOutput:
    generated_text: str | None
    logprobs: Logprobs | None
    prompt_logprobs: Logprobs | None
    refusal: bool


@dataclass
class OpenAIConfig:
    # The environment variable in which the OpenAI API key is stored (or None for
    # local vLLM server requests).
    api_env_var: str | None = None

    # The SLURM job ID of the local vLLM server hosting the local LLM (or None for
    # remote OpenAI server requests).
    vllm_slurm_job_id: int | None = None

    # The base URL of the local vLLM server hosting the local LLM (or None for
    # remote OpenAI server requests).
    vllm_base_url: str | None = None

    # Whether to split out the system instructions (for LLMs that support it).
    use_system_prompt: bool = False

    # Whether to use the OpenAI Completions or OpenAI ChatCompletions protocol.
    use_chat: bool = False

    # Conversion between each MessageType and the specific role name for this LLM.
    message_type_to_role_map: dict[MessageType, str] = field(
        default_factory=lambda: {
            MessageType.SYSTEM: MessageType.SYSTEM.value.lower(),
            MessageType.USER: MessageType.USER.value.lower(),
            MessageType.ASSISTANT: MessageType.ASSISTANT.value.lower(),
        }
    )

    # Suffix for each MessageType's template when joining them all into one string.
    message_type_suffix: dict[MessageType, str] = field(
        default_factory=lambda: {
            MessageType.SYSTEM: "\n\n",
            MessageType.USER: "\n",
            MessageType.ASSISTANT: "\n\n",
        }
    )

    # See openai.OpenAI.chat.completions.create for details. Skip 'model' and
    # 'messages' which are handled specially. Other options may exist when servicing
    # requests based on VLLM servers. See:
    # - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters
    chat_completion_query_params: dict[str, Any] = field(
        default_factory=lambda: {
            # New tokens to add, not total tokens. For o-series OpenAI models, this
            # includes both visible tokens and reasoning tokens.
            "max_completion_tokens": 5,
            # Experimental feature in OpenAI. Might not be 100% deterministic.
            "seed": 314159,
            # Many truthy (1, True) or falsy values (None, 0, False) work.
            "logprobs": True,
            # This is hard-capped at 20 even in vLLM (even if given in 'extra_body').
            # Ignored if "logprobs" is falsy. If truthy, then options are:
            # - 0-20: Gives logprob of the actual prompt token (for all output tokens)
            #         in its own dedicated fields. But also contains a 'top_logprobs'
            #         fields, which is a list (possibly empty, but never None).
            #         Suppose the number is k. The 'top_logprobs' field of each entry
            #         contains k total entries. These are the top-k entries. Often, the
            #         actual prompt token is included (since it is probabilistically
            #         likely to be the top-1 token), but that isn't guaranteed. The
            #         k entries are seemingly in logprob order, but this detail isn't
            #         mentioned in the documentation. Safer to always sort entries.
            #         The list is seemingly always empty for k=0.
            "top_logprobs": 2,
            "extra_body": {
                # Options:
                # - 0: This gives exactly 1 logprob per token (that of the actual
                #      prompt token).
                # - True: This gives the actual prompt token first. Then, if that token
                #         isn't rank 1, it also gives the rank 1 token. So each item
                #         in the return list contains 1 or 2 sub-entries.
                # - 1: This behaves exactly like True.
                # - 2-20: Supposed the number is k. This always gives as many entries as
                #         need to show rank k. The actual prompt token is always first.
                #         The remaining start at rank 1 and go to k, skipping the actual
                #         prompt's rank if it happens to be high enough in the rankings
                #         to have otherwise made the list.
                #         Examples:
                #         - k=2. Actual rank=2. Result: actual, then rank 1
                #         - k=2. Actual rank=7. Result: actual, then rank 1, then rank 2
                #         - k=3. Actual rank=2. Result: actual, then rank 1, then rank 3
                #         Summary: Rank k always has to be in the result. If actual
                #         rank<=k, there are k total entries. If actual rank>k, there
                #         are k+1 total entries. Actual is always first. The rest are in
                #         rank order.
                # NOTE: Contains control tokens, such as "<|start_header_id|>".
                "prompt_logprobs": 1,
            },
        },
    )

    # See openai.OpenAI.completions.create for details. Skip 'model' and 'prompt' which
    # are handled specially. Other options may exist when servicing requests based on
    # VLLM servers. See: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters
    completion_query_params: dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 5,
            # Experimental feature in OpenAI.
            "seed": 314159,
            # Integer or None. This is hard-capped at 5 even in vLLM (even if given in
            # 'extra_body'). NOTE: None != 0. None omits while 0 gives the top value.
            # Unlike, for ChatCompletions, which returns a list of logprob data, one
            # for each token, here the return is a single object which contains parallel
            # lists containing the data for each token:
            # - 'tokens' is a list of the actual tokens (in order)
            # - 'token_logprobs' is a list of the actual tokens' logprobs (in order)
            # - 'top_logprobs' is a list of dictionaries (in order), where each dict
            # contains a token key and a logprob value. The number of entries depends
            # on the value of logprobs. Let's let k be the value. Then:
            # - k=0: This seems to give 'top_logprobs' containing the actual tokens.
            #        But it's also possible that it is simply giving the rank 1 token
            #        and these happen to be the actual tokens in the test runs.
            # - k=True: This seems to behave like k=1.
            # - k=[1-5]: This seems to behave like 'prompt_logprobs' (except perhaps
            #            for the ordering): Rank k always has to be in the result. If
            #            actual rank<=k, there are k total entries. If actual rank>k,
            #            there are k+1 total entries. The ordering of entries is
            #            seemingly based of logprob values, but that is not mentioned
            #            in the documentation. Safer to always sort.
            "logprobs": 5,
            "extra_body": {
                # Identical behaviour to "prompt_logprobs" for ChatCompletions, but
                # without the control tokens, such as "<|start_header_id|>".
                "prompt_logprobs": 1,
            },
        },
    )


class OpenAILLM(LLM):
    """
    Interface for all LLMs using the OpenAI API. Supports both requests to OpenAI
    servers (remote) and requests to a VLLM server (local).
    """

    def __init__(
        self,
        nickname: Nickname,
        llm_cfg: OpenAIConfig,
        *args,
        **kwargs,
    ):
        from openai import OpenAI  # Delayed import.

        super().__init__(nickname, *args, **kwargs)
        self.cfg = llm_cfg
        api_key, base_url = "EMPTY", None
        api_none = self.cfg.api_env_var is None
        job_none = self.cfg.vllm_slurm_job_id is None
        url_none = self.cfg.vllm_base_url is None
        if not api_none and job_none and url_none:
            api_key = os.environ[self.cfg.api_env_var]
        elif not job_none and api_none and url_none:
            out_dir: str = kwargs.get("out_dir", ".")
            log_file = f"{nickname.replace('/', '-')}-{datetime.now().isoformat()}.log"
            init_logger(log_file, filename=str(os.path.join(out_dir, log_file)))
            base_url = await_server(self.cfg.vllm_slurm_job_id, logger_name=log_file)
        elif not url_none and api_none and job_none:
            base_url = self.cfg.vllm_base_url
        else:
            raise ValueError(
                "Configs 'api_env_var', 'vllm_slurm_job_id', and 'vllm_base_url' are "
                "mutually exclusive. Exactly 1 must be non-None."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.make_logprobs = MakeOpenAILogprobs(nickname, **kwargs)

    def invoke(self, prompt_data: PromptData, *args, **kwargs) -> LLMOutput:
        from openai import OpenAIError

        data = prompt_data.additional_data.get("derived_data", None)
        try:
            f_args = (prompt_data.messages,)
            out = retry_call(self._do_invoke, fargs=f_args, fkwargs=kwargs, delay=10)
        except OpenAIError as e:
            return LLMOutput(
                generated_text=None, error_message=str(e), derived_data=data
            )
        if out.refusal:
            return LLMOutput(
                generated_text=None, error_message=out.generated_text, derived_data=data
            )
        data = data or {}
        if out.logprobs is not None:
            data["logprobs"] = out.logprobs
        if out.prompt_logprobs is not None:
            data["prompt_logprobs"] = out.prompt_logprobs
        return LLMOutput(
            generated_text=out.generated_text, error_message=None, derived_data=data
        )

    def _do_invoke(
        self,
        messages: list[Message],
        add_logprobs: bool = False,
        add_prompt_logprobs: bool = False,
        **_,
    ) -> _InvocationOutput:
        roles, suffix = self.cfg.message_type_to_role_map, self.cfg.message_type_suffix
        new_messages = [
            {"role": roles[m.message_type], "content": m.text} for m in messages
        ]
        text = "".join([f"{m.text}{suffix[m.message_type]}" for m in messages]).strip()
        if self.cfg.use_chat:
            if self.cfg.use_system_prompt:
                messages = new_messages
            else:
                messages = [{"role": "user", "content": text}]
            response = self.client.chat.completions.create(
                model=self.nickname,
                messages=messages,
                **self.cfg.chat_completion_query_params,
            )

            # This is always a string, or None if refusal is not None.
            generated_text = response.choices[0].message.content

            # This is always ChoiceLogprobs or None. None definitely occurs if the
            # logprobs where not requested. Can it also be None in other circumstances?
            # Unclear. Supposedly, if refusal is true, ChoiceLogprobs contains the
            # logprobs of the refusal message.

            # Logprobs contains both a "content" and a "refusal" attribute. Supposedly,
            # one will always be non-None. Whichever it is, they have the same format:
            #   a list of ChatCompletionTokenLogprob objects.
            if add_logprobs:
                logprobs = response.choices[0].logprobs
                logprobs = self.make_logprobs(logprobs, "chat_completion")
            else:
                logprobs = None

            # This attribute only exists if the OpenAI API is calling a local vLLM
            # server. If not requested, it is then None. It has a vLLM specific format
            # when it is non-None. It is unclear whether refusal results in None or
            # attribute omission. vLLM puts it under response for ChatCompletion.
            prompt_logprobs = getattr(response, "prompt_logprobs", None)

            # This is presumably non-None when a refusal happens.
            refusal = response.choices[0].message.refusal is not None
        else:
            response = self.client.completions.create(
                model=self.nickname,
                prompt=text,
                **self.cfg.completion_query_params,
            )
            # This is always a string. If content filtering removed it, then it might
            # be an empty string. Who knows. It is not specified in the documentation.
            generated_text = response.choices[0].text

            # This is always a Logprobs object or None. None definitely occurs if the
            # logprobs where not requested. Can it also occur if content filtering
            # removes the generated text? Unclear.
            if add_logprobs:
                logprobs = response.choices[0].logprobs
                logprobs = self.make_logprobs(logprobs, "completion")
            else:
                logprobs = None

            # This attribute only exists if the OpenAI API is calling a local vLLM
            # server. If not requested, it is then None. It has a vLLM specific format
            # when it is non-None. It is unclear whether refusal results in None or
            # attribute omission. vLLM puts it under choices[0] for Completion.
            prompt_logprobs = getattr(response.choices[0], "prompt_logprobs", None)

            # This is presumably the only time a refusal happens.
            refusal = response.choices[0].finish_reason == "content_filter"
        if add_prompt_logprobs:
            prompt_logprobs = self.make_logprobs(prompt_logprobs, "vllm")
        else:
            prompt_logprobs = None
        return _InvocationOutput(generated_text, logprobs, prompt_logprobs, refusal)
