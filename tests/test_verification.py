import asyncio

from app.agent.llm import ChatResult
from app.agent.verification import (
    Claim, check_negation, check_numbers, needs_scrutiny, verify_claims,
)


def test_wording_differences_are_not_fabrications():
    assert check_negation("Employees must submit the form.", ["Employees shall submit the form."]).ok
    assert check_negation("Employees cannot share data.", ["Employees may not share data."]).ok
    assert check_negation("Only managers approve.", ["Managers approve requests."]).ok


def test_fabricated_negation_is_a_hard_fail():
    r = check_negation("Employees cannot work remotely.", ["Employees may work remotely."])
    assert not r.ok


def test_invented_number_is_a_hard_fail():
    assert not check_numbers("Leave is 99 days", ["Leave is 10 days."]).ok
    assert check_numbers("Leave is 3,5 days", ["Leave is 3.5 days."]).ok


def test_dropped_negation_and_added_absolute_trigger_scrutiny():
    assert needs_scrutiny("Employees may work remotely.", ["Employees may not work remotely."])
    assert needs_scrutiny("Only managers approve.", ["Managers approve requests."])
    assert not needs_scrutiny("Sick leave is 10 days.", ["Sick leave is 10 days per year."])


class _NLI:
    configured = True

    def __init__(self, score):
        self.score = score

    def entailment_score(self, premise, hypothesis):
        return self.score


class _Judge:
    configured = True

    def __init__(self, label="supported"):
        self.label, self.calls = label, 0

    async def complete_json(self, system, user, **kw):
        self.calls += 1
        return {"label": self.label, "reason": "r"}, ChatResult("", 1, 1, 0.0)


CTX = {"s1": {"sentence_id": "s1", "document_name": "d", "text": "Managers approve requests."}}


def _run(claim, nli, llm):
    return asyncio.run(
        verify_claims([claim], context_by_id=CTX, nli=nli, llm=llm, nli_pass=0.8, nli_fail=0.2)
    )[0]


def test_confident_nli_skips_the_judge():
    llm = _Judge()
    v = _run(Claim("Managers approve requests.", ["s1"]), _NLI(0.95), llm)
    assert v.ok and v.stage_decided == "nli" and llm.calls == 0


def test_confident_nli_is_not_trusted_when_claim_adds_an_absolute():
    llm = _Judge("unsupported")
    v = _run(Claim("Only managers approve requests.", ["s1"]), _NLI(0.95), llm)
    assert not v.ok and llm.calls == 1


def test_judge_outage_fails_closed():
    class Boom(_Judge):
        async def complete_json(self, *a, **k):
            raise RuntimeError("503")

    v = _run(Claim("Managers approve requests.", ["s1"]), _NLI(0.5), Boom())
    assert not v.ok and v.stage_decided == "judge_unavailable"


def test_framing_is_kept_only_when_it_carries_no_information():
    from app.agent.verification import check_framing

    sources = ["Remote work is allowed two days per week.", "Leave must be approved by the line manager."]
    for ok in ("Dear Manager,", "Kind regards", "Here is what the policy says about remote work:"):
        assert check_framing(ok, sources) == [], ok
    assert check_framing("Remote work is allowed.", sources)            # restates the documents
    assert check_framing("You get 25 days.", sources)                   # a number
    assert check_framing("You cannot do that.", sources)                # a negation
    assert check_framing(" ".join(["word"] * 16), sources)              # too long


PLANETS = {
    "v": {"sentence_id": "v", "document_name": "d", "text": "Venus has a surface temperature of about 465 °C."},
    "m": {"sentence_id": "m", "document_name": "d", "text": "Mercury reaches 430 °C by day and -180 °C at night."},
}


def test_a_number_repeated_from_another_cited_source_goes_to_the_judge_with_that_source():
    llm = _Judge()
    claims = [
        Claim("Venus is about 465 °C at the surface.", ["v"]),
        Claim("Mercury swings from 430 °C to -180 °C, so it averages far below Venus's 465 °C.", ["m"]),
        Claim("Mercury averages 200 °C.", ["m"]),  # 200 is in no source of the answer
    ]
    venus, compare, invented = asyncio.run(
        verify_claims(claims, context_by_id=PLANETS, nli=_NLI(0.95), llm=llm, nli_pass=0.8, nli_fail=0.2)
    )
    assert venus.ok and venus.stage_decided == "nli"
    # kept by the judge, not NLI, and now also cites the source the 465 came from
    assert compare.ok and compare.stage_decided == "judge" and compare.claim.citations == ["m", "v"]
    assert llm.calls == 1
    assert not invented.ok and invented.stage_decided == "rules"
