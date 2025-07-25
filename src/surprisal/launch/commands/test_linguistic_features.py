from coma import command

import pandas as pd

from ...core import (
    AccordAnalyzer,
    ConceptNetFormatter,
    FormatMethod,
    LinguisticFeatures,
    TermFormatter,
    Triplet,
)


def test_default(data: dict[str, list]) -> None:
    analyzer = AccordAnalyzer(
        features=LinguisticFeatures(),
        formatter=ConceptNetFormatter(
            template="{data}",
            method=FormatMethod.ACCORD,
            formatter=TermFormatter(language="en"),
        ),
    )

    source, relation, f_target = "/c/en/acorn/n", "AtLocation", "/c/en/oak_tree"
    f_analysis = analyzer.analyze_factual(Triplet(source, relation, f_target))
    if f_analysis is None:
        raise ValueError("Bug in analyze factual.")

    af_targets = {
        "/c/en/office_supply_store/n": True,
        "/c/en/office_supply_stores/n": False,  # Because tag of stores is NNS not NN
        "/c/en/become_accountant": False,
    }
    for t, expected_result in af_targets.items():
        res = analyzer.is_valid_anti_factual(Triplet(source, relation, t), f_analysis)
        if res != expected_result:
            data.setdefault("Test", []).append("default")
            data.setdefault("Target", []).append(t)
            data.setdefault("Expected", []).append(expected_result)
            data.setdefault("Actual", []).append(res)


def test_children_basic(data: dict[str, list]) -> None:
    analyzer = AccordAnalyzer(
        features=LinguisticFeatures(children=True),
        formatter=ConceptNetFormatter(
            template="{data}",
            method=FormatMethod.ACCORD,
            formatter=TermFormatter(language="en"),
        ),
    )

    source, relation, f_target = "/c/en/acorn/n", "AtLocation", "/c/en/oak_tree"
    f_analysis = analyzer.analyze_factual(Triplet(source, relation, f_target))
    if f_analysis is None:
        raise ValueError("Bug in analyze factual.")

    af_targets = {
        "/c/en/supply_store/n": True,  # Tree => store; oak => supply are matches
        "/c/en/office_supply_store/n": False,  # Because too many children
        "/c/en/office_supply_stores/n": False,  # Because tag of stores is NNS not NN
        "/c/en/become_accountant": False,
    }
    for t, expected_result in af_targets.items():
        res = analyzer.is_valid_anti_factual(Triplet(source, relation, t), f_analysis)
        if res != expected_result:
            data.setdefault("Test", []).append("basic-children")
            data.setdefault("Target", []).append(t)
            data.setdefault("Expected", []).append(expected_result)
            data.setdefault("Actual", []).append(res)


def test_children_advanced(data: dict[str, list]) -> None:
    analyzer = AccordAnalyzer(
        features=LinguisticFeatures(children=True),
        formatter=ConceptNetFormatter(
            template="{data}",
            method=FormatMethod.ACCORD,
            formatter=TermFormatter(language="en"),
        ),
    )

    source, relation = "/c/en/acorn/n", "AtLocation"
    f_target = "/c/en/very_young_american_oak_tree"
    f_analysis = analyzer.analyze_factual(Triplet(source, relation, f_target))
    if f_analysis is None:
        raise ValueError("Bug in analyze factual.")

    af_targets = {
        # Very difficult to find an alternative long term that is internally consistent.
        "/c/en/very_young_american_oak_tree/n": True,
        # Because office is child of supply (compound), rather than store directly.
        "/c/en/very_red_office_supply_store/n": False,
    }
    for t, expected_result in af_targets.items():
        res = analyzer.is_valid_anti_factual(Triplet(source, relation, t), f_analysis)
        if res != expected_result:
            data.setdefault("Test", []).append("advanced-children")
            data.setdefault("Target", []).append(t)
            data.setdefault("Expected", []).append(expected_result)
            data.setdefault("Actual", []).append(res)


def test_pos_only(data: dict[str, list]) -> None:
    analyzer = AccordAnalyzer(
        features=LinguisticFeatures(pos=True, tag=False, dep=False, full_morph=False),
        formatter=ConceptNetFormatter(
            template="{data}",
            method=FormatMethod.ACCORD,
            formatter=TermFormatter(language="en"),
        ),
    )

    source, relation, f_target = "/c/en/acorn/n", "AtLocation", "/c/en/oak_tree"
    f_analysis = analyzer.analyze_factual(Triplet(source, relation, f_target))
    if f_analysis is None:
        raise ValueError("Bug in analyze factual.")

    af_targets = {
        "/c/en/office_supply_store/n": True,
        "/c/en/office_supply_stores/n": True,
        "/c/en/become_accountant": False,  # Because pos is verb instead of noun
    }
    for t, expected_result in af_targets.items():
        res = analyzer.is_valid_anti_factual(Triplet(source, relation, t), f_analysis)
        if res != expected_result:
            data.setdefault("Test", []).append("pos_only")
            data.setdefault("Target", []).append(t)
            data.setdefault("Expected", []).append(expected_result)
            data.setdefault("Actual", []).append(res)


@command(name="test.linguistic.features")
def cmd():
    data = {}
    test_default(data)
    test_children_basic(data)
    test_children_advanced(data)
    test_pos_only(data)
    if not data:
        print("All tests passed!")
    else:
        print("Failed tests:")
        print(pd.DataFrame(data))
