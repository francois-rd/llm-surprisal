from coma import command

from ...core import Logprob, Logprobs, RankedLogprob


@command(name="test.logprob.alignment")
def cmd():
    text = (
        "This is a test [sentence] with     arbitrary\t\n"
        "whitespace and repeat of the word 'sentence'."
    )
    tokens = [
        "This ",
        "is",
        " a ",
        "test ",
        "[",
        "sen",
        "tence",
        "] ",
        "with",
        "     arbitrary",
        "\t\nwhitespace ",
        "and ",
        "repeat",
        " of ",
        "the ",
        "word ",
        "'",
        "sen",
        "tence",
        "'",
        ".",
    ]
    assert "".join(tokens) == text
    sequence = []
    for token in tokens:
        logprob = Logprob(token=token, rank=1, logprob=-1)
        sequence.append(RankedLogprob(chosen=logprob, others={}, ranking="relative"))
    logprobs = Logprobs(sequence=sequence)
    result = list(logprobs.indices_of("sentence"))
    assert len(result) == 2
    assert result[0].indices == {5: "", 6: ""}
    assert result[0].to_text() == "sentence"
    assert result[1].to_text() == "sentence"
    assert result[1].indices == {17: "", 18: ""}
    result = list(logprobs.indices_of("[sentence]"))
    assert len(result) == 1
    assert result[0].indices == {4: "", 5: "", 6: "", 7: ""}
    assert result[0].to_text() == "[sentence]"
    result = list(logprobs.indices_of("is a test"))
    assert len(result) == 1
    assert result[0].indices == {1: "", 2: " ", 3: " "}
    assert result[0].to_text() == "is a test"
    result = list(logprobs.indices_of("with     arbitrary\t\nwhitespace"))
    assert len(result) == 1
    assert result[0].indices == {8: "", 9: "     ", 10: "\t\n"}
    assert result[0].to_text() == "with     arbitrary\t\nwhitespace"
    print("All passed!")
