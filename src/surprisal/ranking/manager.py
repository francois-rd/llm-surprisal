from dataclasses import dataclass
from typing import Any

from .rankers import ConceptNetHierarchyRanker, Ranker, RandomRanker, RankingMethod


RankerID = str


@dataclass
class RankerInfo:
    method: RankingMethod
    data: dict[str, Any]


@dataclass
class RankersConfig:
    rankers: dict[RankerID, RankerInfo]


def ranker_factory(method: RankingMethod, **data) -> Ranker:
    if method == RankingMethod.RANDOM:
        return RandomRanker(**data)
    elif method == RankingMethod.CNET_HIERARCHY:
        return ConceptNetHierarchyRanker(**data)
    else:
        raise ValueError(f"Unsupported ranking method: {method}")


class RankerManager:
    def __init__(self, cfg: RankersConfig):
        self.rankers = {}
        for ranker_id, info in cfg.rankers.items():
            self.rankers[ranker_id] = ranker_factory(info.method, **info.data)

    def get(self, ranker_id: RankerID) -> Ranker:
        try:
            return self.rankers[ranker_id]
        except KeyError:
            raise ValueError(f"Missing ranker data for: '{ranker_id}'.")
