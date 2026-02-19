from dataclasses import dataclass
from typing import Literal
from enum import Enum

from ruptures.exceptions import BadSegmentationParameters
from ruptures import KernelCPD
from hampel import hampel
import numpy as np


KernelOptions = Literal["linear", "rbf", "cosine"]


@dataclass
class SerialHyperparameters:
    cpd_kernel: str = "linear"  # OmegaConf cannot serialize Literal type.
    cpd_penalty: float | None = 2.0
    hampel_window_size: int = 10
    hampel_n_sigma: float = 3.0


@dataclass
class CollectiveSerialHyperparameters:
    instance: SerialHyperparameters = SerialHyperparameters()
    question: SerialHyperparameters = SerialHyperparameters()
    answer_portion: SerialHyperparameters = SerialHyperparameters()
    choices: SerialHyperparameters = SerialHyperparameters()
    forced: SerialHyperparameters = SerialHyperparameters()


class SerialOption(Enum):
    # Change point-based metrics.
    CHANGE_COUNT = "CHANGE_COUNT"  # Normalized as count / total tokens.
    CHANGE_FIRST_INDEX = "CHANGE_FIRST_INDEX"  # Normalized as index / total tokens.
    CHANGE_FIRST_SIZE = "CHANGE_FIRST_SIZE"  # ABS_Diff(first CP, prior token)
    CHANGE_MEAN_SIZE = "CHANGE_MEAN_SIZE"  # mean(ABS_Diff(CP_i, token_{i-1}))
    CHANGE_MEDIAN_SIZE = "CHANGE_MEDIAN_SIZE"  # median(ABS_Diff(CP_i, token_{i-1}))
    CHANGE_MIN_SPACING = "CHANGE_MIN_SPACING"  # min space b/w consecutive change pts.
    CHANGE_MEAN_SPACING = "CHANGE_MEAN_SPACING"  # mean space b/w consecutive CPs.
    CHANGE_MEDIAN_SPACING = "CHANGE_MEDIAN_SPACING"  # md space b/w consecutive CPs.
    # Outlier-based metrics.
    OUTLIER_COUNT = "OUTLIER_COUNT"  # Normalized as count / total tokens.
    OUTLIER_FIRST_INDEX = "OUTLIER_FIRST_INDEX"  # Normalized as index / total tokens.
    OUTLIER_FIRST_SIGMA = "OUTLIER_FIRST_SIGMA"  # MAD std of first outlier.
    OUTLIER_MEAN_SIGMA = "OUTLIER_MEAN_SIGMA"  # mean(MAD std) over all outliers.
    OUTLIER_MEDIAN_SIGMA = "OUTLIER_MEDIAN_SIGMA"  # md(MAD std) over all outliers.
    OUTLIER_MIN_SPACING = "OUTLIER_MIN_SPACING"  # min space b/w consecutive outliers.
    OUTLIER_MEAN_SPACING = "OUTLIER_MEAN_SPACING"  # mean space b/w consecutive outliers
    OUTLIER_MEDIAN_SPACING = "OUTLIER_MEDIAN_SPACING"  # md b/w consecutive outliers.

    def as_attribute(self) -> str:
        return self.value.lower()


@dataclass
class SerialMetrics:
    change_count: float | None
    change_first_index: float | None
    change_first_size: float | None
    change_mean_size: float | None
    change_median_size: float | None
    change_min_spacing: float | None
    change_mean_spacing: float | None
    change_median_spacing: float | None
    outlier_count: float | None
    outlier_first_index: float | None
    outlier_first_sigma: float | None
    outlier_mean_sigma: float | None
    outlier_median_sigma: float | None
    outlier_min_spacing: float | None
    outlier_mean_spacing: float | None
    outlier_median_spacing: float | None

    logprobs: list[float] | None
    change_indices: list[int] | None
    outlier_indices: list[int] | None

    cumsum: list[float] | None

    @staticmethod
    def build_from(lps: list[float], hps: SerialHyperparameters) -> "SerialMetrics":
        return SerialMetrics(
            **_cpd_analysis(lps, hps.cpd_kernel, hps.cpd_penalty),
            **_hampel_analysis(lps, hps.hampel_window_size, hps.hampel_n_sigma),
            logprobs=lps,
            cumsum=np.cumsum(lps).tolist(),
        )

    def get(self, option: SerialOption) -> float | None:
        return getattr(self, option.as_attribute())

    def relative_to(self, other: "SerialMetrics") -> "SerialMetrics":
        slp, olp = self.logprobs, other.logprobs
        if slp is None or olp is None or len(slp) != len(olp):
            lp, cs = None, None
        else:
            lp = [slp[i] - olp[i] for i in range(len(slp))]
            cs = np.cumsum(lp).tolist()
        kw = dict(logprobs=lp, change_indices=None, outlier_indices=None, cumsum=cs)
        # TODO: Unclear if taking the difference of medians makes sense, statistically.
        for option in SerialOption:
            s, o = self.get(option), other.get(option)
            kw[option.as_attribute()] = None if (s is None or o is None) else s - o
        return SerialMetrics(**kw)


class SerialComponent(Enum):
    INSTANCE = "INSTANCE"
    QUESTION = "QUESTION"
    ANSWER_PORTION = "ANSWER_PORTION"
    CHOICE_MATCHING_ACCORD = "CHOICE_MATCHING_ACCORD"
    CHOICE_MATCHING_CSQA = "CHOICE_MATCHING_CSQA"
    FORCED_MATCHING_ACCORD = "FORCED_MATCHING_ACCORD"
    FORCED_MATCHING_CSQA = "FORCED_MATCHING_CSQA"

    def as_attribute(self) -> str:
        return self.value.lower()


@dataclass
class CollectiveSerialMetrics:
    instance: SerialMetrics
    question: SerialMetrics
    answer_portion: SerialMetrics
    choice_matching_accord: SerialMetrics
    choice_matching_csqa: SerialMetrics
    forced_matching_accord: SerialMetrics | None = None  # This gets filled post-init.
    forced_matching_csqa: SerialMetrics | None = None  # This gets filled post-init.

    def get(self, c: SerialComponent) -> SerialMetrics:
        return getattr(self, c.as_attribute())

    def relative_to(
        self, other: "CollectiveSerialMetrics"
    ) -> "CollectiveSerialMetrics":
        return CollectiveSerialMetrics(
            instance=self.instance.relative_to(other.instance),
            question=self.question.relative_to(other.question),
            answer_portion=self.answer_portion.relative_to(other.answer_portion),
            choice_matching_accord=self.choice_matching_accord.relative_to(
                other.choice_matching_accord
            ),
            choice_matching_csqa=self.choice_matching_csqa.relative_to(
                other.choice_matching_csqa
            ),
            forced_matching_accord=self.forced_matching_accord.relative_to(
                other.forced_matching_accord
            ),
            forced_matching_csqa=self.forced_matching_csqa.relative_to(
                other.forced_matching_csqa
            ),
        )


def _cpd_analysis(
    logprobs: list[float], kernel: str, penalty: float | None
) -> dict[str, float]:
    try:
        indices = KernelCPD(kernel=kernel).fit_predict(
            signal=np.array(logprobs),  # noqa
            pen=penalty,
        )

        # There seems to be a bug where it always returns the very last index as a CP.
        indices = np.array(indices[:-1])
    except BadSegmentationParameters:
        # This happens on the short time series (CHOICE and FORCED).
        # Proceed as if no change points are detected.
        indices = np.array([])

    if indices.size > 0:
        diffs = np.ediff1d(logprobs)[indices - 1]
        change_indices = indices.tolist()
        change_count = float(indices.size / len(logprobs))
        change_first_index = float(indices[0] / len(logprobs))
        change_first_size = float(diffs[0])
        change_mean_size = float(diffs.mean())
        change_median_size = float(np.median(diffs))
    else:
        change_indices = []
        change_count = None
        change_first_index = None
        change_first_size = None
        change_mean_size = None
        change_median_size = None
    if len(indices) > 1:
        diffs = np.ediff1d(indices)
        change_min_spacing = float(diffs.min())
        change_mean_spacing = float(diffs.mean())
        change_median_spacing = float(np.median(diffs))
    else:
        change_min_spacing = None
        change_mean_spacing = None
        change_median_spacing = None
    return dict(
        change_indices=change_indices,
        change_count=change_count,
        change_first_index=change_first_index,
        change_first_size=change_first_size,
        change_mean_size=change_mean_size,
        change_median_size=change_median_size,
        change_min_spacing=change_min_spacing,
        change_mean_spacing=change_mean_spacing,
        change_median_spacing=change_median_spacing,
    )


def _hampel_analysis(
    logprobs: list[float],
    window_size: int,
    n_sigma: float,
) -> dict[str, float]:
    k = 1.4826  # Hampel Gaussian scale factor. See
    # https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d/
    # https://stats.stackexchange.com/questions/355943/how-to-estimate-the-scale-factor-for-mad-for-a-non-normal-distribution

    # Get outlier indices from hampel filter.
    data = np.array(logprobs)
    r = hampel(data, window_size=window_size, n_sigma=n_sigma)
    indices = r.outlier_indices

    # This is the actual sigma values from hampel. Any values > 'hampel_n_sigma'
    # are outliers returned by the library. We can index this array at the position
    # of the outliers to compute the actual number of sigmas, rather than just a
    # binary "did it exceed the threshold?".
    sigmas = np.abs(data - r.medians) / (r.median_absolute_deviations * k + 0.0001)
    if indices.size > 0:
        outlier_indices = indices.tolist()
        outlier_count = float(indices.size / data.size)
        outlier_first_index = float(indices[0] / data.size)
        outlier_first_sigma = float(sigmas[indices[0]])
        outlier_mean_sigma = float(sigmas[indices].mean())
        outlier_median_sigma = float(np.median(sigmas[indices]))
    else:
        outlier_indices = []
        outlier_count = None
        outlier_first_index = None
        outlier_first_sigma = None
        outlier_mean_sigma = None
        outlier_median_sigma = None
    if indices.size > 1:
        diffs = np.ediff1d(indices)
        outlier_min_spacing = float(diffs.min())
        outlier_mean_spacing = float(diffs.mean())
        outlier_median_spacing = float(np.median(diffs))
    else:
        outlier_min_spacing = None
        outlier_mean_spacing = None
        outlier_median_spacing = None
    return dict(
        outlier_indices=outlier_indices,
        outlier_count=outlier_count,
        outlier_first_index=outlier_first_index,
        outlier_first_sigma=outlier_first_sigma,
        outlier_mean_sigma=outlier_mean_sigma,
        outlier_median_sigma=outlier_median_sigma,
        outlier_min_spacing=outlier_min_spacing,
        outlier_mean_spacing=outlier_mean_spacing,
        outlier_median_spacing=outlier_median_spacing,
    )
