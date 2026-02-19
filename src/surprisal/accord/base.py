from dataclasses import dataclass


AccordLabel = str
AccordRelation = str
AccordTerm = str
AccordID = str
AccordCaseID = int
AccordStatementID = str
AccordStatementKey = tuple[AccordTerm, AccordTerm, AccordRelation]
CsqaID = str


@dataclass
class AccordPairing:
    """
    Dataclass to store pairing data.

    Fields:
        id: Identifier of the pairing statement (which has the same ID in each tree)
        flip_negative: Whether the positive/negative variation logic needs to be flipped
        relation_type: The skill type of the relation of the pairing statement.
    """

    id: AccordStatementID
    flip_negative: bool
    relation_type: AccordRelation


@dataclass
class AccordStatement:
    """
    Dataclass to store statement data.

    Fields:
        surface_form: Format string for the surface (i.e., text) form of the statement
        source_term: String value of the source term
        target_term: String value of the target term
        relation_type: The skill type of the relation of the statement
    """

    surface_form: str
    source_term: AccordTerm
    target_term: AccordTerm
    relation_type: AccordRelation

    def to_key(self) -> AccordStatementKey:
        return self.source_term, self.target_term, self.relation_type


@dataclass
class AccordMetaData:
    """
    Dataclass to store all instance meta-data.

    Fields:
        id: Identifier for this instance within the data subset
        qa_id: Identifier of the CSQA instance from the base dataset
        question: Question text of the CSQA instance
        answer_choices: Answer choices of the CSQA instance; form: {label: term}
        label: Chosen answer label (which can be different from the CSQA instance label)
        pairing: Pairing data
        reduction_cases: Indicates which statements reduce to which cases wrt the
            pairing statement; form {statement_id: case_id}
        statements: All Statement data for each statement in each tree; form:
            {answer_label/tree_label: {statement_id: Statement}}
        statements_order: Ordered list of all Statements from all trees; form:
            [(answer_label/tree_label, statement_id)]
    """

    id: AccordID
    qa_id: CsqaID
    question: str
    answer_choices: dict[AccordLabel, AccordTerm]
    label: AccordLabel
    pairing: AccordPairing | None
    reduction_cases: dict[AccordStatementID, AccordCaseID] | None
    statements: dict[AccordLabel, dict[AccordStatementID, AccordStatement]]
    statements_ordering: list[list]

    def get_statement(
        self, ordering: tuple[AccordLabel, AccordStatementID]
    ) -> AccordStatement:
        """Returns a specific statement based on an ordering tuple entry."""
        return self.statements[ordering[0]][ordering[1]]


@dataclass
class AccordInstance:
    """
    Dataclass to store all ACCORD instance data.

    Fields:
        text: Fully surfaced text string of the instance data for LLM consumption
        csqa_label: Ground truth label of the base CSQA instance from which this ACCORD
            derives (which can be different from the ACCORD's chosen answer label)
        meta_data: Metadata for this ACCORD instance
    """

    text: str
    csqa_label: AccordLabel
    meta_data: AccordMetaData

    def is_factual(self) -> bool:
        return self.csqa_label == self.meta_data.label


@dataclass
class AccordCaseLink:
    """
    A permutation case linking two statements based on their relations' skill type.
    Typically, one statement will dominate over the other.

    The five permutation cases are:
        Case 0: A relation1 B; C relation2 D (no linking relation)
        Case 1: A relation1 B; B relation2 C (relation1.target == relation2.source)
        Case 2: A relation1 B; C relation2 B (relation1.target == relation2.target)
        Case 3: B relation1 A; B relation2 C (relation1.source == relation2.source)
        Case 4: B relation1 A; C relation2 B (relation1.source == relation2.target)

    NOTE: The cases are equivalent with respect to permutation of the order of the
    statements unless the relation types are the same (where the surface form of the
    subsumed statement can sometimes be different). In general, however:
        Case 1 permutes to case 4
        Case 4 permutes to case 1
        All other cases permute to themselves

    Fields:
        r1_type: Relation skill type of the dominant statement
        r2_type: Relation skill type of the subsumed statement
        case: Permutation case linking the two statements
    """

    r1_type: AccordRelation
    r2_type: AccordRelation
    case: AccordCaseID

    def equivalent(self) -> "AccordCaseLink":
        if self.case == 1:
            return AccordCaseLink(self.r2_type, self.r1_type, 4)
        elif self.case == 0 or self.case == 2 or self.case == 3:
            return AccordCaseLink(self.r2_type, self.r1_type, self.case)
        elif self.case == 4:
            return AccordCaseLink(self.r2_type, self.r1_type, 1)
        else:
            raise TypeError(f"Unsupported permutation case: {self.case}")

    def as_tuple(self) -> tuple[AccordRelation, AccordRelation, AccordCaseID]:
        return self.r1_type, self.r2_type, self.case


class AccordSurfaceForms:
    """
    Keeps track of surface form variations of subsumed relations in a CaseLink.
    """

    def __init__(self):
        self.permutations = {}

    def register(self, case_link: AccordCaseLink, subsumed_surface_form: str):
        """Registers a CaseLink and its associated subsumed surface form."""
        self.permutations[case_link.as_tuple()] = subsumed_surface_form

    def get(self, case_link: AccordCaseLink) -> str | None:
        """
        Returns the surface form for a CaseLink. Returns the surface form of the
        equivalent CaseLink under case permutation if no surface form is registered
        for the given CaseLink. Returns None if neither the given nor the equivalent
        CaseLink have registered surface forms.
        """
        equiv = case_link.equivalent().as_tuple()
        if case_link.as_tuple() in self.permutations:
            return self.permutations[case_link.as_tuple()]
        elif equiv in self.permutations:
            return self.permutations[equiv]
        else:
            return None


@dataclass
class CsqaBase:
    """
    Dataclass to store all data for an instance of base CSQA.

    Fields:
        identifier: Identifier of the CSQA instance from the base dataset
        question: Question text of the CSQA instance
        correct_answer_label: Ground truth answer label if the CSQA instance
        answer_choices: Answer choices of the CSQA instance; form: {label: term}
        pairing_templates: ignored
    """

    identifier: CsqaID
    question: str
    correct_answer_label: AccordLabel
    answer_choices: dict[AccordLabel, AccordTerm]
    pairing_templates: list[dict]


@dataclass
class ComponentBoundaries:
    statements: tuple[int, int] | None
    question: tuple[int, int]
    answer_choices: tuple[int, int]
    instance: tuple[int, int]
