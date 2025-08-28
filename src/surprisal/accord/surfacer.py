import re

from .base import AccordCaseLink, AccordMetaData, AccordSurfaceForms


class AccordSurfacer:
    """A Surfacer converts data to a text string, optionally preceded with a prefix."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, meta_data: AccordMetaData, *args, **kwargs) -> str:
        raise NotImplementedError


class AccordTextSurfacer(AccordSurfacer):
    """Surfaces the given text preceded with the given prefix."""

    def __init__(self, prefix: str, text: str):
        super().__init__(prefix)
        self.text = text

    def __call__(self, *args, **kwargs) -> str:
        return self.prefix + self.text


class AccordTermSurfacer(AccordSurfacer):
    """Surfaces a term preceded and succeeded with the given prefix and suffix."""

    def __init__(self, prefix: str, suffix: str):
        super().__init__(prefix)
        self.suffix = suffix

    def __call__(self, *args, **kwargs) -> str:
        """NOTE: "kwargs" should contain a "term"."""
        if "term" not in kwargs:
            raise KeyError("No Term provided for surfacing.")
        return self.prefix + kwargs["term"] + self.suffix


class AccordStatementSurfacer(AccordSurfacer):
    """
    Surfaces a complete Statement using the given term surfacer and surface form
    registry preceded with the given prefix.
    """

    def __init__(
        self, prefix: str, term_surfacer: AccordSurfacer, forms: AccordSurfaceForms
    ):
        super().__init__(prefix)
        self.surfacer = term_surfacer
        self.forms = forms
        self.pos_neg_pattern = re.compile(r"(\[\[(.*?)\|(.*?)]])")

    def __call__(self, meta_data: AccordMetaData, *args, **kwargs) -> str:
        """NOTE: "kwargs" should contain an "ordering" for the Statement to process."""
        # Grab the statement based on the ordering.
        if "ordering" not in kwargs:
            raise KeyError("No ordering provided for surfacing.")
        tree_label, statement_id = kwargs["ordering"]
        statement = meta_data.get_statement(kwargs["ordering"])
        reduction_cases = meta_data.reduction_cases

        # Surface the source and target terms.
        source_term = self.surfacer(*args, term=statement.source_term, **kwargs)
        target_term = self.surfacer(*args, term=statement.target_term, **kwargs)

        # If the statement is part of the reduction cases, update the surface form.
        if reduction_cases is not None and statement_id in reduction_cases:
            case_link = AccordCaseLink(
                r1_type=meta_data.pairing.relation_type,
                r2_type=statement.relation_type,
                case=reduction_cases[statement_id],
            )
            surface_form = self.forms.get(case_link)
            in_reduction = True
        else:
            surface_form = statement.surface_form
            in_reduction = False

        # Format the surface form using the surfaced source and target terms.
        text = surface_form.format(source_term, target_term)

        # Find all positive/negative variations in the surface form of the statement.
        match = self.pos_neg_pattern.findall(text)

        # If the statement is part of the reduction cases or is the pairing statement,
        # format according to positive/negative variations.
        if in_reduction or statement_id == meta_data.pairing.id:
            if statement_id == meta_data.pairing.id and not match:
                raise ValueError(
                    "Pairing statement must contain positive/negative variations."
                )

            # Replace pos/neg variations based on label match with the chosen answer.
            is_positive = meta_data.label == tree_label
            if meta_data.pairing.flip_negative:
                is_positive = not is_positive
            for group, positive, negative in match:
                text = text.replace(group, positive if is_positive else negative, 1)
        elif match:
            # The statement is not a pairing statement, so it CANNOT contain variations.
            raise ValueError(
                "Non-pairing statement cannot contain positive/negative variations."
            )
        return self.prefix + text


class AccordOrderingSurfacer(AccordSurfacer):
    """
    Surfaces all Statements based on the ordering in the MetaData using the given
    statement surfacer. Each statement is separated with the given statement separator
    and the whole sequence is preceded with the given prefix.
    """

    def __init__(
        self,
        prefix: str,
        statement_separator: str,
        statement_surfacer: AccordSurfacer,
    ):
        super().__init__(prefix)
        self.surfacer = statement_surfacer
        self.separator = statement_separator

    def __call__(self, meta_data: AccordMetaData, *args, **kwargs) -> str:
        return self.prefix + self.separator.join(
            [
                self.surfacer(meta_data, *args, **kwargs, ordering=ordering)
                for ordering in meta_data.statements_ordering
            ]
        )


class AccordQADataSurfacer(AccordSurfacer):
    """
    Surfaces all QA instance data. The question and answer choices are separated with
    the given question answer separator. Each answer choice is separated with the given
    answer choice separator. Each answer choice (label, term) is formatted using the
    given answer choice formatter. The whole text is preceded with the given prefix.
    """

    def __init__(
        self,
        prefix: str,
        question_answer_separator: str,
        answer_choice_separator: str,
        answer_choice_formatter: str,
    ):
        super().__init__(prefix)
        self.question_answer_separator = question_answer_separator
        self.answer_choice_separator = answer_choice_separator
        self.answer_choice_formatter = answer_choice_formatter

    def __call__(self, meta_data: AccordMetaData, *args, **kwargs) -> str:
        answer_choices = [
            self.answer_choice_formatter.format(label, term)
            for label, term in meta_data.answer_choices.items()
        ]
        return (
            self.prefix
            + meta_data.question
            + self.question_answer_separator
            + self.answer_choice_separator.join(answer_choices)
        )


class AccordInstanceSurfacer(AccordSurfacer):
    """
    Surfaces an Instance using all its MetaData. The prefix surfacer can be used to
    prefix the main instance data with some text (for example, with an instruction
    prompt). The ordering surfacer can be used to surface the Statements in the correct
    order. The QA data surfacer can be used to surface the QA instance data. The suffix
    surfacer can be used to append the preceding text with more text (for example, with
    an answer prompt). Each surfacer is separated with the given surfacer separator.
    Each surfacer can be None, in which case it is ignored. The whole text is preceded
    with the given prefix.
    """

    def __init__(
        self,
        prefix: str,
        surfacer_separator: str,
        prefix_surfacer: AccordSurfacer | None,
        ordering_surfacer: AccordSurfacer | None,
        qa_data_surfacer: AccordSurfacer | None,
        suffix_surfacer: AccordSurfacer | None,
    ):
        super().__init__(prefix)
        self.surfacer_separator = surfacer_separator
        self.prefix_surfacer = prefix_surfacer
        self.ordering_surfacer = ordering_surfacer
        self.qa_data_surfacer = qa_data_surfacer
        self.suffix_surfacer = suffix_surfacer
        self.surfacers = [
            prefix_surfacer,
            ordering_surfacer,
            qa_data_surfacer,
            suffix_surfacer,
        ]

    def __call__(self, meta_data: AccordMetaData, *args, **kwargs) -> str:
        def fn_caller(fn):
            return fn(meta_data, *args, **kwargs)

        return self.prefix + self.surfacer_separator.join(
            [fn_caller(f) for f in self.surfacers if f is not None],
        )
