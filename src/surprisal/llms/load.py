from .base import LLM, LLMsConfig, LLMImplementation, MISSING_NICKNAME
from .dummy import DummyLLM
from .openai import OpenAILLM


def load_llm(cfg: LLMsConfig, *args, **kwargs) -> LLM:
    if cfg.llm == MISSING_NICKNAME:
        raise ValueError(f"Missing runtime LLM: {cfg.llm}")
    if cfg.implementation == LLMImplementation.MISSING:
        raise ValueError(f"Missing implementation type for LLM: {cfg.llm}")
    elif cfg.implementation == LLMImplementation.DUMMY:
        return DummyLLM(cfg.llm, *args, **kwargs)
    elif cfg.implementation == LLMImplementation.OPENAI:
        return OpenAILLM(cfg.llm, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM implementation type: {cfg.implementation}")
