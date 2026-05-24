"""Offline unit tests for the knowledge_qa plugin.

Validates plugin invariants without any live LLM:
  - source-provenance (add_triple rejects unregistered sources)
  - excerpt-in-source (add_triple rejects fabricated excerpts)
  - value-in-evidence (subject AND object literal substrings)
  - predicate-support (LLM judge — substituted with a deterministic stub here)
  - ClaimEvidenceAlignmentVerifier accepts/rejects per the contract
  - coherence_check rejects unknown verifiers and subjective+evidence pairs

The real wiki_tool isn't exercised here (network); atomic_qa_demo covers that.

Run:
  python -m task_runtime.tests.test_knowledge_qa
  pytest task_runtime/tests/test_knowledge_qa.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_runtime.core import (  # noqa: E402
    Accept, IncoherentContract, RejectWithRepairHint, TaskContract, TaskResult,
)
from task_runtime.plugins.knowledge_qa import (  # noqa: E402
    KnowledgeQAPlugin, disabled_claim_judge,
)


def _seed_source(p: KnowledgeQAPlugin, source_id: str, body: str) -> None:
    """Pretend a wiki_read happened — populate the source registry directly."""
    p.sources[source_id] = body


# -----------------------------------------------------------------------------
# add_triple: source provenance
# -----------------------------------------------------------------------------

def test_add_triple_rejects_unknown_source() -> None:
    """A model that fabricates a source_id must be rejected."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    out = p._add_triple(
        subject="Alice", predicate="founded", object="Example Institute",
        source="wiki:Some_Page",  # never registered
        excerpt="Alice founded the Example Institute in 1985.",
    )
    assert "error" in out
    assert "unknown source" in out["error"].lower()
    assert len(p.triples) == 0


def test_add_triple_rejects_excerpt_not_in_source() -> None:
    """A model that quotes text not actually in the source body is rejected."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College is a private liberal arts college in Memphis.")
    out = p._add_triple(
        subject="Marcus Pohlmann", predicate="founded",
        object="Rhodes College mock trial team",
        source="wiki:Rhodes_College",
        excerpt="Marcus Pohlmann founded the Rhodes College mock trial team.",
    )
    assert "error" in out
    assert "not a substring" in out["error"]
    assert len(p.triples) == 0


def test_add_triple_excerpt_whitespace_normalized() -> None:
    """Multi-line/reflowed excerpts still match the source body."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College is a private liberal arts college in Memphis, Tennessee.")
    # Excerpt has different whitespace from source.
    out = p._add_triple(
        subject="Rhodes College", predicate="located_in", object="Memphis",
        source="wiki:Rhodes_College",
        excerpt="Rhodes  College  is a  private  liberal arts college\nin Memphis, Tennessee.",
    )
    assert "added" in out, out
    assert out["graph_size"] == 1


# -----------------------------------------------------------------------------
# add_triple: value-in-evidence
# -----------------------------------------------------------------------------

def test_add_triple_accepts_subject_and_object_in_excerpt() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College is a private liberal arts college in Memphis, Tennessee.")
    out = p._add_triple(
        subject="Rhodes College", predicate="located_in", object="Memphis",
        source="wiki:Rhodes_College",
        excerpt="Rhodes College is a private liberal arts college in Memphis, Tennessee.",
    )
    assert "added" in out, out
    assert out["added"]["id"] == "t-0000"
    assert out["graph_size"] == 1


def test_add_triple_rejects_when_subject_missing_from_excerpt() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_source(p, "wiki:Brett_Kavanaugh",
                 "Brett Kavanaugh is a justice of the United States Supreme Court.")
    out = p._add_triple(
        subject="Marcus Pohlmann", predicate="founded", object="Mock Trial Program",
        source="wiki:Brett_Kavanaugh",
        excerpt="Brett Kavanaugh is a justice of the United States Supreme Court.",
    )
    assert "error" in out
    assert "Marcus Pohlmann" in out["error"]
    assert len(p.triples) == 0


def test_add_triple_case_insensitive_value_check() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College is a private liberal arts college in Memphis, Tennessee.")
    out = p._add_triple(
        subject="RHODES COLLEGE", predicate="located_in", object="memphis",
        source="wiki:Rhodes_College",
        excerpt="Rhodes College is a private liberal arts college in Memphis, Tennessee.",
    )
    assert "added" in out, out


# -----------------------------------------------------------------------------
# add_triple: predicate support (judge)
# -----------------------------------------------------------------------------

def test_add_triple_rejects_unsupported_predicate() -> None:
    """The `provides` vs `founded` mismatch from the live atomic_qa_demo."""
    def reject_founded(subject, predicate, obj, excerpt):
        if predicate == "founded" and "provides" in excerpt.lower():
            return False, "excerpt says 'provides', not 'founded'"
        return True, ""

    p = KnowledgeQAPlugin(claim_judge=reject_founded)
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College provides an undergraduate mock trial program.")
    out = p._add_triple(
        subject="Rhodes College", predicate="founded", object="mock trial program",
        source="wiki:Rhodes_College",
        excerpt="Rhodes College provides an undergraduate mock trial program.",
    )
    assert "error" in out
    assert "does not support" in out["error"]
    assert "provides" in out["error"]
    assert len(p.triples) == 0


def test_add_triple_records_judge_reason_on_success() -> None:
    def approve_with_reason(subject, predicate, obj, excerpt):
        return True, "explicit match"

    p = KnowledgeQAPlugin(claim_judge=approve_with_reason)
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College is in Memphis.")
    out = p._add_triple(
        subject="Rhodes College", predicate="located_in", object="Memphis",
        source="wiki:Rhodes_College",
        excerpt="Rhodes College is in Memphis.",
    )
    assert "added" in out
    assert out["judge_reason"] == "explicit match"
    assert p.triples[0].judge_reason == "explicit match"


# -----------------------------------------------------------------------------
# query_graph
# -----------------------------------------------------------------------------

def test_query_graph_filters_by_subject_and_predicate() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College is in Memphis. Rhodes College was founded in 1848.")
    _seed_source(p, "wiki:Brown_University",
                 "Brown University is in Providence, Rhode Island.")
    p._add_triple(subject="Rhodes College", predicate="located_in", object="Memphis",
                  source="wiki:Rhodes_College",
                  excerpt="Rhodes College is in Memphis.")
    p._add_triple(subject="Rhodes College", predicate="founded_in", object="1848",
                  source="wiki:Rhodes_College",
                  excerpt="Rhodes College was founded in 1848.")
    p._add_triple(subject="Brown University", predicate="located_in", object="Providence",
                  source="wiki:Brown_University",
                  excerpt="Brown University is in Providence, Rhode Island.")

    assert len(p._query_graph(subject="Rhodes")["matches"]) == 2
    assert len(p._query_graph(predicate="located_in")["matches"]) == 2
    assert len(p._query_graph(subject="Rhodes", predicate="founded")["matches"]) == 1
    assert p._query_graph()["total_in_graph"] == 3


# -----------------------------------------------------------------------------
# ClaimEvidenceAlignmentVerifier
# -----------------------------------------------------------------------------

def _seed_one_triple(p: KnowledgeQAPlugin) -> str:
    _seed_source(p, "wiki:Rhodes_College",
                 "Rhodes College is a private liberal arts college in Memphis, Tennessee.")
    p._add_triple(
        subject="Rhodes College", predicate="located_in", object="Memphis",
        source="wiki:Rhodes_College",
        excerpt="Rhodes College is a private liberal arts college in Memphis, Tennessee.",
    )
    return p.triples[0].id


def test_verifier_accepts_when_cited_triples_exist() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    tid = _seed_one_triple(p)
    v = p.verifiers()["claim_evidence_alignment"]
    verdict = v.check(
        TaskContract(goal="x", verifier="claim_evidence_alignment"),
        TaskResult(status="success",
                   output={"answer": "Memphis", "supporting_triple_ids": [tid]}),
        workspace={},
    )
    assert isinstance(verdict, Accept), verdict
    assert verdict.record["cited_triple_ids"] == [tid]
    assert verdict.record["cited_triples"][0]["subject"] == "Rhodes College"
    assert verdict.record["sources_registered"] == ["wiki:Rhodes_College"]


def test_verifier_rejects_when_no_supporting_triples() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_one_triple(p)
    v = p.verifiers()["claim_evidence_alignment"]
    verdict = v.check(
        TaskContract(goal="x", verifier="claim_evidence_alignment"),
        TaskResult(status="success", output={"answer": "Memphis", "supporting_triple_ids": []}),
        workspace={},
    )
    assert isinstance(verdict, RejectWithRepairHint), verdict


def test_verifier_rejects_unknown_triple_ids() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    _seed_one_triple(p)
    v = p.verifiers()["claim_evidence_alignment"]
    verdict = v.check(
        TaskContract(goal="x", verifier="claim_evidence_alignment"),
        TaskResult(status="success",
                   output={"answer": "Memphis", "supporting_triple_ids": ["t-9999"]}),
        workspace={},
    )
    assert isinstance(verdict, RejectWithRepairHint), verdict
    assert "t-9999" in verdict.reason


def test_verifier_rejects_non_object_output() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    v = p.verifiers()["claim_evidence_alignment"]
    verdict = v.check(
        TaskContract(goal="x", verifier="claim_evidence_alignment"),
        TaskResult(status="success", output="just a string answer"),
        workspace={},
    )
    assert isinstance(verdict, RejectWithRepairHint), verdict


# -----------------------------------------------------------------------------
# coherence_check
# -----------------------------------------------------------------------------

def test_coherence_unknown_verifier_rejected() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    v = p.coherence_check(TaskContract(goal="anything", verifier="nope"))
    assert isinstance(v, IncoherentContract), v
    assert "nope" in v.reason


def test_coherence_subjective_goal_with_evidence_verifier_rejected() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    v = p.coherence_check(TaskContract(
        goal="What is the purpose of life?",
        verifier="claim_evidence_alignment",
    ))
    assert isinstance(v, IncoherentContract), v


def test_coherence_factual_goal_with_evidence_verifier_passes() -> None:
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    v = p.coherence_check(TaskContract(
        goal="Who founded the mock trial program at Rhodes College?",
        verifier="claim_evidence_alignment",
    ))
    assert v is None


# -----------------------------------------------------------------------------
# propose_triples + evidence-bound mention resolution
# -----------------------------------------------------------------------------

def test_propose_triples_resolves_surname_subject_mention() -> None:
    """Canonical 'Brett Kavanaugh', excerpt only says 'Kavanaugh' → accepted,
    with the graph storing the canonical and recording the surface mention.

    Object surname 'Trump' for canonical 'Donald Trump' now requires an
    explicit alias under the role-sensitive policy (object surnames are
    too ambiguous by default — could be Donald Trump, Ivanka Trump, etc.)."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Brett_Kavanaugh"] = (
        "Kavanaugh was nominated by President Trump in 2018."
    )
    out = p._propose_triples([{
        "subject": "Brett Kavanaugh",
        "predicate": "appointed by",
        # Object aliases supplied explicitly per the role-sensitive policy.
        "object": {"canonical": "Donald Trump",
                   "aliases": ["President Trump", "Trump"]},
        "source": "wiki:Brett_Kavanaugh",
        "excerpt": "Kavanaugh was nominated by President Trump in 2018.",
    }])
    assert len(out["accepted"]) == 1, out
    t = p.triples[0]
    # Canonical stays full; mention reflects surface form.
    assert t.subject == "Brett Kavanaugh"
    assert t.subject_mention == "Kavanaugh"
    assert t.object == "Donald Trump"
    # Object mention came from explicit alias (longest match: "President Trump").
    assert t.object_mention in ("President Trump", "Trump"), t.object_mention


def test_propose_triples_with_explicit_alias() -> None:
    """Aliases take priority over derived candidates — useful for honorifics."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Trump"] = "President Trump nominated three SCOTUS justices."
    out = p._propose_triples([{
        "subject": {"canonical": "Donald Trump", "aliases": ["President Trump"]},
        "predicate": "nominated",
        "object": {"canonical": "SCOTUS justices", "aliases": ["SCOTUS justices"]},
        "source": "wiki:Trump",
        "excerpt": "President Trump nominated three SCOTUS justices.",
    }])
    assert len(out["accepted"]) == 1, out
    assert p.triples[0].subject_mention == "President Trump"


def test_propose_triples_rejects_when_no_mention_found() -> None:
    """Canonical entity not findable in the excerpt → rejected before judge."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Gorsuch"] = "Gorsuch was nominated by Trump in 2017."
    out = p._propose_triples([{
        "subject": "Brett Kavanaugh",   # not in excerpt under any form
        "predicate": "appointed by",
        "object": "Donald Trump",
        "source": "wiki:Gorsuch",
        "excerpt": "Gorsuch was nominated by Trump in 2017.",
    }])
    assert len(out["accepted"]) == 0
    assert len(out["rejected"]) == 1
    err = out["rejected"][0]["error"]
    assert "no mention" in err.lower()
    assert "Brett Kavanaugh" in err


def test_propose_triples_first_token_not_a_default_candidate() -> None:
    """'Michael' is NOT a default mention candidate for 'Michael Coney' —
    a generic 'Michael ...' excerpt must not falsely certify the entity.
    Must be supplied via explicit aliases if desired."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Random"] = "Michael went to the store yesterday."
    out = p._propose_triples([{
        "subject": "Michael Coney",     # excerpt has "Michael" alone — should NOT count
        "predicate": "went",
        "object": "the store",
        "source": "wiki:Random",
        "excerpt": "Michael went to the store yesterday.",
    }])
    assert len(out["accepted"]) == 0
    assert "no mention" in out["rejected"][0]["error"].lower()


def test_propose_triples_stored_triple_preserves_canonical_identity() -> None:
    """The graph stores the canonical 'Amy Coney Barrett' even when the
    excerpt only contains 'Barrett' — slot certifiers can then match on
    the canonical via _has_overlap."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Barrett"] = "Barrett's father is Michael Coney."
    out = p._propose_triples([{
        "subject": "Amy Coney Barrett",
        "predicate": "father",
        "object": "Michael Coney",
        "source": "wiki:Barrett",
        "excerpt": "Barrett's father is Michael Coney.",
    }])
    assert len(out["accepted"]) == 1, out
    t = p.triples[0]
    # The graph node remembers the canonical entity, not the surname.
    assert t.subject == "Amy Coney Barrett"
    assert t.subject_mention == "Barrett"
    # object happened to match in full.
    assert t.object == "Michael Coney"
    assert t.object_mention == "Michael Coney"
    # Slot-overlap on the canonical works:
    from task_runtime.plugins.knowledge_qa import _has_overlap
    assert _has_overlap("Amy Coney Barrett", t.subject)


# -----------------------------------------------------------------------------
# ClaimSupportVerifierV2: structured witness + mechanical validation
# -----------------------------------------------------------------------------

from task_runtime.plugins.knowledge_qa import (  # noqa: E402
    _validate_witness, _schema_for_predicate, make_witness_claim_judge,
)


KAVANAUGH_AMBIGUOUS_EXCERPT = (
    "Brett Kavanaugh is of Irish Catholic descent on both sides of his family. "
    "His paternal great-great-grandfather Michael Murphy and his wife immigrated "
    "from Ireland to New Jersey. Kavanaugh's father was a lawyer."
)


def test_witness_rejects_kavanaugh_father_michael_false_positive() -> None:
    """The canonical adjacent-text failure: a witness that tries to claim
    (Brett Kavanaugh, father, Michael) from this excerpt must be rejected
    because 'great-great-grandfather' immediately precedes 'Michael'."""
    schema = _schema_for_predicate("father")
    # Best possible witness for the bad claim: cover both clauses.
    witness = {
        "supported": True,
        "support_span": (
            "His paternal great-great-grandfather Michael Murphy "
            "and his wife immigrated from Ireland to New Jersey. "
            "Kavanaugh's father was a lawyer."
        ),
        "subject_mention": "Kavanaugh",
        "predicate_cue": "father",
        "object_mention": "Michael",
        "binding_explanation": "father binds Kavanaugh to Michael (incorrectly)",
        "rejection_reason": None,
    }
    ok, reason = _validate_witness(
        witness, "Brett Kavanaugh", "father", "Michael",
        KAVANAUGH_AMBIGUOUS_EXCERPT, schema,
    )
    assert not ok, f"should reject but accepted: {reason}"
    assert "great-great-grandfather" in reason or "disallowed" in reason.lower()


def test_witness_accepts_kavanaugh_father_everett() -> None:
    """The TRUE father claim — (Brett Kavanaugh, father, Everett Edward
    Kavanaugh Jr.) — with a clean witness span must be accepted."""
    schema = _schema_for_predicate("father")
    excerpt = (
        "Kavanaugh was born on February 12, 1965, in Washington, D.C., "
        "the son of Martha Gamble and Everett Edward Kavanaugh Jr."
    )
    witness = {
        "supported": True,
        "support_span": (
            "the son of Martha Gamble and Everett Edward Kavanaugh Jr."
        ),
        "subject_mention": "Kavanaugh",
        "predicate_cue": "son of",
        "object_mention": "Everett Edward Kavanaugh Jr.",
        "binding_explanation": "'son of ... Everett Edward Kavanaugh Jr.' binds father",
        "rejection_reason": None,
    }
    # NB: subject_mention "Kavanaugh" isn't in this span; the span starts
    # mid-sentence. Validator will check subject_mention IS in span.
    # Let's give a span that includes Kavanaugh.
    witness["support_span"] = excerpt
    ok, reason = _validate_witness(
        witness, "Brett Kavanaugh", "father", "Everett Edward Kavanaugh Jr.",
        excerpt, schema,
    )
    assert ok, f"should accept but rejected: {reason}"


def test_witness_accepts_barrett_father_michael_coney() -> None:
    """The Barrett correct case — full father name in span, 'father' is an
    allowed cue, no disallowed cue precedes the object."""
    schema = _schema_for_predicate("father")
    excerpt = "Amy Coney Barrett's father is Michael Coney, a former lawyer."
    witness = {
        "supported": True,
        "support_span": "Amy Coney Barrett's father is Michael Coney",
        "subject_mention": "Amy Coney Barrett",
        "predicate_cue": "father",
        "object_mention": "Michael Coney",
        "binding_explanation": "'father is Michael Coney' identifies the father",
        "rejection_reason": None,
    }
    ok, reason = _validate_witness(
        witness, "Amy Coney Barrett", "father", "Michael Coney",
        excerpt, schema,
    )
    assert ok, f"should accept but rejected: {reason}"


def test_witness_rejects_michael_alone_as_father_object() -> None:
    """A bare 'Michael' as object — without the full 'Michael Coney' — must
    not certify father. The canonical object 'Michael Coney' doesn't
    substring-overlap 'Michael' alone via _has_overlap (no shared >=3-letter
    token besides 'Michael' itself, but the canonical's other token 'Coney'
    isn't matched). Actually 'Michael' is shared, so _has_overlap returns
    True. So this test instead verifies that the validator catches the
    case via disallowed-cue precedence."""
    schema = _schema_for_predicate("father")
    # The witness claims 'Michael' as object for father(Kavanaugh, ?).
    # 'great-great-grandfather' precedes 'Michael' → disallowed cue
    # binding catches it.
    excerpt = "His paternal great-great-grandfather Michael Murphy immigrated."
    witness = {
        "supported": True,
        "support_span": excerpt,
        "subject_mention": "His",  # not great, but span check still applies
        "predicate_cue": "father",
        "object_mention": "Michael",
        "binding_explanation": "n/a",
        "rejection_reason": None,
    }
    ok, reason = _validate_witness(
        witness, "Brett Kavanaugh", "father", "Michael", excerpt, schema,
    )
    assert not ok, f"should reject but accepted: {reason}"


def test_witness_accepts_nominated_by_trump_as_appointed_by() -> None:
    """The relation schema for appointed_by allows 'nominated by' as a cue,
    and 'Trump' / 'President Trump' as object aliases for 'Donald Trump'."""
    schema = _schema_for_predicate("appointed_by")
    excerpt = "Amy Coney Barrett was nominated by President Trump in 2020."
    witness = {
        "supported": True,
        "support_span": excerpt,
        "subject_mention": "Amy Coney Barrett",
        "predicate_cue": "nominated by",
        "object_mention": "President Trump",
        "binding_explanation": "'nominated by President Trump' = appointed_by Donald Trump",
        "rejection_reason": None,
    }
    ok, reason = _validate_witness(
        witness, "Amy Coney Barrett", "appointed_by", "Donald Trump",
        excerpt, schema,
    )
    assert ok, f"should accept but rejected: {reason}"


def test_witness_rejects_overlong_span_used_to_fuse_distant_terms() -> None:
    """If the witness picks a span > 250 chars to make subject and object
    coexist, the validator rejects on length."""
    schema = _schema_for_predicate("father")
    long_excerpt = (
        "Kavanaugh's father was a lawyer. " + "Filler. " * 50 + "Michael Murphy"
    )
    witness = {
        "supported": True,
        "support_span": long_excerpt,
        "subject_mention": "Kavanaugh",
        "predicate_cue": "father",
        "object_mention": "Michael",
        "binding_explanation": "n/a",
        "rejection_reason": None,
    }
    ok, reason = _validate_witness(
        witness, "Brett Kavanaugh", "father", "Michael",
        long_excerpt, schema,
    )
    assert not ok
    assert "too long" in reason.lower()


def test_witness_judge_factory_works_with_stub_extractor() -> None:
    """make_witness_claim_judge accepts a stub extractor — useful for offline
    use in propose_triples / add_triple flows."""
    def stub(subject, predicate, obj, excerpt, schema):
        return {
            "supported": True,
            "support_span": excerpt,
            "subject_mention": subject,
            "predicate_cue": "is",
            "object_mention": str(obj),
            "binding_explanation": "stub",
            "rejection_reason": None,
        }

    judge = make_witness_claim_judge(witness_extractor=stub)
    ok, reason = judge("X", "rel", "Y", "X is Y.")
    assert ok, reason


# -----------------------------------------------------------------------------
# Deterministic witness-candidate generation (V3)
# -----------------------------------------------------------------------------

from task_runtime.plugins.knowledge_qa import (  # noqa: E402
    generate_witness_candidates, make_classifier_claim_judge,
)


def test_candidate_generator_rejects_kavanaugh_father_michael() -> None:
    """The deterministic generator filters out spans where 'great-grandfather'
    (a disallowed cue) precedes 'Michael' near the candidate object.
    Result: no candidate offers a false-positive binding to the classifier."""
    schema = _schema_for_predicate("father")
    body = (
        "His paternal great-great-grandfather Michael Murphy and his wife "
        "immigrated from Ireland to New Jersey. Kavanaugh's father was a lawyer."
    )
    candidates = generate_witness_candidates(
        canonical_subject="Brett Kavanaugh", predicate="father",
        canonical_object="Michael", body=body, schema=schema,
    )
    # Either no candidates, or none that bind 'father' to 'Michael' through
    # the disallowed great-grandfather span.
    for c in candidates:
        span_l = c["span"].lower()
        # If 'Michael' appears, 'great-great-grandfather' must NOT precede it
        # within 60 chars.
        m_idx = span_l.find("michael")
        g_idx = span_l.find("great-great-grandfather")
        assert not (0 <= g_idx < m_idx <= g_idx + len("great-great-grandfather") + 60), \
            f"generator emitted a span where great-grandfather precedes Michael: {c['span']!r}"


def test_candidate_generator_finds_barrett_father() -> None:
    """The Barrett-father claim succeeds when the source contains a clean
    binding. The generator should produce at least one candidate whose
    span, subject_mention, predicate_cue, object_mention all align."""
    schema = _schema_for_predicate("father")
    body = (
        "Amy Coney Barrett's father is Michael Coney, a former lawyer. "
        "She grew up in New Orleans."
    )
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="father",
        canonical_object="Michael Coney", body=body, schema=schema,
    )
    assert candidates, "expected at least one candidate"
    top = candidates[0]
    # The chosen subject mention overlaps the canonical (Barrett shares the
    # word "Coney" with the canonical).
    assert "barrett" in top["subject_mention"].lower() or \
           "coney" in top["subject_mention"].lower()
    # The cue is in allowed_cues.
    assert top["predicate_cue"].lower() in [c.lower() for c in schema["allowed_cues"]]
    # The object_mention is or overlaps "Michael Coney".
    assert "michael" in top["object_mention"].lower()


def test_generated_witness_span_is_substring_of_body() -> None:
    """Every candidate span must be a verbatim substring of the body —
    invariant the mechanical validator relies on."""
    schema = _schema_for_predicate("appointed_by")
    body = "Amy Coney Barrett was nominated by President Trump in 2020."
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="appointed_by",
        canonical_object="Donald Trump", body=body, schema=schema,
    )
    assert candidates
    for c in candidates:
        assert c["span"] in body, f"generated span not in body: {c['span']!r}"


def test_candidate_includes_subject_object_cue() -> None:
    """The top candidate must contain the subject mention, object mention,
    and predicate cue — all three — within its span."""
    schema = _schema_for_predicate("appointed_by")
    body = "Amy Coney Barrett was nominated by President Trump in 2020."
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="appointed_by",
        canonical_object="Donald Trump", body=body, schema=schema,
    )
    assert candidates
    top = candidates[0]
    span_l = top["span"].lower()
    assert top["subject_mention"].lower() in span_l
    assert top["predicate_cue"].lower() in span_l
    assert top["object_mention"].lower() in span_l


def test_great_grandfather_pattern_filtered_at_generation() -> None:
    """Negative: a span containing 'great-great-grandfather Michael Murphy'
    cannot make it onto the candidate list for father(Kavanaugh, Michael)."""
    schema = _schema_for_predicate("father")
    body = "Kavanaugh's great-great-grandfather Michael Murphy immigrated."
    candidates = generate_witness_candidates(
        canonical_subject="Brett Kavanaugh", predicate="father",
        canonical_object="Michael", body=body, schema=schema,
    )
    # Either no candidates, or none that include the disallowed pattern.
    for c in candidates:
        assert "great-great-grandfather" not in c["span"].lower(), \
            f"generator should have filtered this span: {c['span']!r}"


def test_classifier_judge_cannot_invent_spans() -> None:
    """If the classifier returns a chosen_span_id outside the candidate
    list, the judge rejects. The LLM cannot smuggle a hallucinated span
    past validation."""
    def evil_classifier(subject, predicate, obj, excerpt, candidates, schema):
        return {
            "supported": True,
            "chosen_span_id": 999,  # not a valid candidate index
            "binding_explanation": "fake",
            "rejection_reason": None,
        }
    judge = make_classifier_claim_judge(witness_classifier=evil_classifier)
    body = "Amy Coney Barrett's father is Michael Coney."
    ok, reason = judge("Amy Coney Barrett", "father", "Michael Coney", body)
    assert not ok
    assert "chosen_span_id" in reason.lower() or "invalid" in reason.lower()


def test_classifier_judge_end_to_end_with_stub_classifier() -> None:
    """The classifier-judge factory composes correctly with a stub
    classifier that picks candidate 0."""
    def picker(subject, predicate, obj, excerpt, candidates, schema):
        return {"supported": True, "chosen_span_id": 0,
                "binding_explanation": "stub", "rejection_reason": None}
    judge = make_classifier_claim_judge(witness_classifier=picker)
    body = "Amy Coney Barrett's father is Michael Coney."
    ok, reason = judge("Amy Coney Barrett", "father", "Michael Coney", body)
    assert ok, reason


# -----------------------------------------------------------------------------
# Source-local alias extraction (V4)
# -----------------------------------------------------------------------------

from task_runtime.plugins.knowledge_qa import (  # noqa: E402
    extract_source_local_aliases,
)


def test_source_local_birth_name_registered() -> None:
    """When wiki_read encounters a 'X was born' pattern in the opening, the
    extracted name is registered as a source-local alias for the page subject.

    Note: the extractor filters out aliases that don't actually appear in
    the body, so 'Barrett' / 'Coney Barrett' (canonical-derived) won't be
    in the returned set unless the body actually mentions them."""
    barrett_body = (
        "Amy Coney Barrett (née Coney; born January 28, 1972) is an associate "
        "justice. Amy Vivian Coney was born in 1972 in New Orleans, Louisiana, "
        "to Linda (née Vath) and Michael Coney."
    )
    aliases = extract_source_local_aliases(barrett_body, "Amy Coney Barrett")
    # Birth name from 'X was born' pattern.
    assert "Amy Vivian Coney" in aliases, aliases
    # Canonical-derived forms that ALSO appear in the body.
    assert "Amy Coney Barrett" in aliases
    assert "Barrett" in aliases or "Coney Barrett" in aliases


def test_witness_generation_uses_source_local_alias() -> None:
    """End-to-end: a Barrett-father claim grounded by an excerpt using her
    birth name 'Amy Vivian Coney' is accepted when source-local aliases are
    threaded through to the candidate generator. The body must contain a
    literal allowed cue ('father'/'son of'/etc.) for candidates to exist."""
    barrett_body = (
        "Amy Vivian Coney was born in 1972 to Linda and Michael Coney. "
        "Amy Vivian Coney's father is Michael Coney, a lawyer."
    )
    schema = _schema_for_predicate("father")
    local_aliases = list(extract_source_local_aliases(barrett_body, "Amy Coney Barrett"))
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="father",
        canonical_object="Michael Coney", body=barrett_body, schema=schema,
        subject_aliases=local_aliases,
    )
    assert candidates, (
        f"expected at least one candidate; local_aliases={local_aliases}"
    )
    top = candidates[0]
    # Subject mention should be a source-local alias since canonical
    # 'Amy Coney Barrett' doesn't appear in this body.
    assert top["subject_mention"] in local_aliases, (
        f"top subject_mention {top['subject_mention']!r} should be a "
        f"source-local alias from {local_aliases}"
    )


def test_source_local_alias_does_not_leak_globally() -> None:
    """The plugin registers source-local aliases scoped to a source_id.
    They must not be visible when grounding a triple against a DIFFERENT
    source. This protects the canonical-identity invariant."""
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    barrett_body = "Amy Vivian Coney was born in 1972 to Michael Coney."
    p.sources["wiki:Amy Coney Barrett"] = barrett_body
    p.source_aliases["wiki:Amy Coney Barrett"] = {
        "Amy Coney Barrett": {"Amy Vivian Coney"}
    }
    # Asked for Barrett aliases inside her own page → returned.
    assert "Amy Vivian Coney" in p._subject_aliases_for(
        "Amy Coney Barrett", "wiki:Amy Coney Barrett"
    )
    # Asked for Barrett aliases inside ANOTHER page → empty (no leak).
    assert p._subject_aliases_for(
        "Amy Coney Barrett", "wiki:Brett Kavanaugh"
    ) == []
    # Asked for a totally different canonical → empty.
    assert p._subject_aliases_for(
        "Some Other Person", "wiki:Amy Coney Barrett"
    ) == []


def test_first_name_only_alias_still_rejected() -> None:
    """Source-local extraction must NOT promote 'Michael' alone as an alias
    for 'Michael Coney' or any other multi-word entity, even if the source
    mentions the first name."""
    body = "Michael went to the store. He bought milk."
    aliases = extract_source_local_aliases(body, "Michael Coney")
    # 'Michael' alone is a single token that isn't the SURNAME of canonical
    # 'Michael Coney' (surname is 'Coney'), so it must NOT be promoted.
    assert "Michael" not in aliases, aliases


def test_kavanaugh_false_positive_remains_rejected_with_aliases() -> None:
    """Source-local aliasing must not resurrect the Kavanaugh-father-Michael
    false positive. Even if 'Brett Michael Kavanaugh' is extracted as a
    source-local alias, the disallowed-cue-precedes-object filter at
    candidate generation still blocks the bad binding."""
    kav_body = (
        "Brett Michael Kavanaugh was born on February 12, 1965. "
        "His paternal great-great-grandfather Michael Murphy immigrated "
        "from Ireland. Kavanaugh's father was a lawyer."
    )
    schema = _schema_for_predicate("father")
    local_aliases = list(extract_source_local_aliases(kav_body, "Brett Kavanaugh"))
    # Whatever aliases get extracted, the candidate generator must NOT
    # produce a span that lets 'father' bind 'Kavanaugh' to 'Michael'
    # through the great-grandfather clause.
    candidates = generate_witness_candidates(
        canonical_subject="Brett Kavanaugh", predicate="father",
        canonical_object="Michael", body=kav_body, schema=schema,
        subject_aliases=local_aliases,
    )
    for c in candidates:
        span_l = c["span"].lower()
        g_idx = span_l.find("great-great-grandfather")
        m_idx = span_l.find("michael")
        assert not (0 <= g_idx < m_idx <= g_idx + 60), (
            f"source-local aliases must not weaken disallowed-cue filter; "
            f"got bad span: {c['span']!r}"
        )


# -----------------------------------------------------------------------------
# Relation-targeted witness search (V5)
# -----------------------------------------------------------------------------

def test_appointed_by_accepts_nominated_by_president_donald_trump() -> None:
    """The expanded appointed_by schema accepts 'nominated by' as a cue and
    'President Donald Trump' as an object alias."""
    schema = _schema_for_predicate("appointed_by")
    body = "Amy Coney Barrett was nominated by President Donald Trump in 2020."
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="appointed_by",
        canonical_object="Donald Trump", body=body, schema=schema,
    )
    assert candidates, f"expected a candidate; schema={schema}"
    top = candidates[0]
    assert "nominated" in top["predicate_cue"].lower()
    assert "trump" in top["object_mention"].lower()


def test_appointed_by_accepts_trump_nominated_her_with_subject_alias() -> None:
    """When the source says 'Trump nominated Barrett...' the candidate
    generator finds 'Barrett' as a source-local-style alias and Trump as
    object."""
    schema = _schema_for_predicate("appointed_by")
    body = "President Trump nominated Barrett to the Supreme Court."
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="appointed_by",
        canonical_object="Donald Trump", body=body, schema=schema,
        subject_aliases=["Barrett"],
    )
    assert candidates, "expected a candidate using 'Barrett' alias"


def test_senate_confirmed_alone_does_not_satisfy_appointed_by() -> None:
    """'The Senate confirmed Barrett' alone has no Trump cue/mention, so no
    candidate should be generated for appointed_by Trump."""
    schema = _schema_for_predicate("appointed_by")
    body = "The Senate confirmed Barrett in October 2020."
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="appointed_by",
        canonical_object="Donald Trump", body=body, schema=schema,
        subject_aliases=["Barrett"],
    )
    # Either no candidates, OR none whose object_mention overlaps Trump.
    for c in candidates:
        assert "trump" in c["object_mention"].lower(), (
            f"Senate-only excerpt should not produce a Trump-binding candidate; "
            f"got {c}"
        )
    # In practice, the body has no Trump term at all, so 0 candidates.
    assert not candidates, candidates


def test_trump_criticized_does_not_satisfy_appointed_by() -> None:
    """'Trump criticized Barrett' contains a Trump mention but no
    appointment cue — and 'criticized by' is in disallowed_cues."""
    schema = _schema_for_predicate("appointed_by")
    body = "Trump criticized Barrett's dissent in 2024."
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="appointed_by",
        canonical_object="Donald Trump", body=body, schema=schema,
        subject_aliases=["Barrett"],
    )
    # No appointment/nomination cue present in this body.
    assert not candidates, f"criticized excerpt should not produce a candidate; got {candidates}"


def test_trump_nominated_kavanaugh_does_not_satisfy_barrett() -> None:
    """A clearly-about-someone-else excerpt should not produce a Barrett
    candidate."""
    schema = _schema_for_predicate("appointed_by")
    body = "Trump nominated Kavanaugh in 2018."
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="appointed_by",
        canonical_object="Donald Trump", body=body, schema=schema,
        subject_aliases=["Barrett", "Amy Vivian Coney"],
    )
    # No subject mention of Barrett — generator finds nothing.
    assert not candidates, f"Kavanaugh excerpt should not produce a Barrett candidate; got {candidates}"


def test_source_local_alias_still_works_for_father_after_appointment_changes() -> None:
    """Regression: the Barrett-father case from the previous round still
    works with the source-local alias 'Amy Vivian Coney' after the
    appointed_by schema changes."""
    schema = _schema_for_predicate("father")
    body = (
        "Amy Vivian Coney was born in 1972 to Linda and Michael Coney. "
        "Amy Vivian Coney's father is Michael Coney."
    )
    candidates = generate_witness_candidates(
        canonical_subject="Amy Coney Barrett", predicate="father",
        canonical_object="Michael Coney", body=body, schema=schema,
        subject_aliases=["Amy Vivian Coney"],
    )
    assert candidates, candidates


# -----------------------------------------------------------------------------
# EntityConstraintResolutionExecutor
# -----------------------------------------------------------------------------

def test_executor_certifies_correct_candidate_offline() -> None:
    """The executor mechanically enumerates → reads bios → propose_triples →
    candidate_table → selects the candidate satisfying all obligations.
    Stubs wiki_read so this is fully offline."""
    from task_runtime.plugins.knowledge_qa import resolve_entity_constraint_slot

    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    # Stub wiki sources for each candidate's "bio page".
    p.sources["wiki:Neil Gorsuch"] = (
        "Neil Gorsuch was appointed by Donald Trump in 2017. "
        "Gorsuch's father is David Gorsuch."
    )
    p.sources["wiki:Brett Kavanaugh"] = (
        "Brett Kavanaugh was appointed by Donald Trump in 2018. "
        "Brett Kavanaugh's father is Everett Edward Kavanaugh."
    )
    p.sources["wiki:Amy Coney Barrett"] = (
        "Amy Coney Barrett was appointed by Donald Trump in 2020. "
        "Amy Coney Barrett's father is Michael Coney."
    )
    # Stub the wiki_read tool so the executor's bio-reads are no-ops on the
    # ALREADY-registered sources (it'd otherwise try to fetch from network).
    p._wiki_read = lambda title: {
        "title": title, "source_id": f"wiki:{title}",
        "found": f"wiki:{title}" in p.sources,
        "body": p.sources.get(f"wiki:{title}", ""),
    }

    slot_spec = {
        "name": "justice",
        "proof_obligations": [
            {"predicate_contains": "appointed", "object_contains": "Trump"},
            {"predicate_contains": "father", "object_contains": "Michael"},
        ],
        "disallowed_predicates": ["middle name"],
        "preferred_method": "enumerate_filter",
    }
    result = resolve_entity_constraint_slot(
        plugin=p, slot_name="justice", slot_spec=slot_spec,
        candidates=["Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett"],
    )

    # The executor selected Barrett (the only one whose father is named Michael).
    assert "Barrett" in result["value"], result
    # Candidate table covers all 3 candidates.
    assert [r["candidate"] for r in result["candidate_table"]] == [
        "Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett",
    ]
    # Only Barrett satisfies.
    sat_flags = [r["satisfies"] for r in result["candidate_table"]]
    assert sat_flags == [False, False, True], sat_flags
    # Cited obligation triples are populated for the selected row.
    assert len(result["cited_obligation_triples"]) == 2


def test_executor_uses_canonical_mention_split() -> None:
    """When the bio uses a shorter form (e.g. only 'Kavanaugh' not 'Brett
    Kavanaugh'), the executor still records canonical='Brett Kavanaugh'
    in the graph; mention reflects the surface form."""
    from task_runtime.plugins.knowledge_qa import resolve_entity_constraint_slot

    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    # Bio uses surname only — the canonical-vs-mention split should handle it.
    p.sources["wiki:Brett Kavanaugh"] = "Kavanaugh was appointed by Trump in 2018."
    p._wiki_read = lambda title: {
        "title": title, "source_id": f"wiki:{title}",
        "found": f"wiki:{title}" in p.sources,
        "body": p.sources.get(f"wiki:{title}", ""),
    }

    slot_spec = {
        "name": "justice",
        "proof_obligations": [
            {"predicate_contains": "appointed", "object_contains": "Trump"},
        ],
        "disallowed_predicates": [],
        "preferred_method": "enumerate_filter",
    }
    result = resolve_entity_constraint_slot(
        plugin=p, slot_name="justice", slot_spec=slot_spec,
        candidates=["Brett Kavanaugh"],
    )
    # Triple was accepted and is in the graph with canonical preserved.
    assert len(p.triples) >= 1
    t = p.triples[0]
    assert t.subject == "Brett Kavanaugh"          # canonical
    assert t.subject_mention == "Kavanaugh"        # surface mention


def test_executor_does_not_use_add_triple_for_disallowed_paths() -> None:
    """The executor relies exclusively on propose_triples, not add_triple.
    Sanity-check by observing that all triples accepted by the executor
    carry a recorded subject_mention (the artifact of mention resolution)."""
    from task_runtime.plugins.knowledge_qa import resolve_entity_constraint_slot

    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Amy Coney Barrett"] = (
        "Barrett's father is Michael Coney."
    )
    p._wiki_read = lambda title: {
        "title": title, "source_id": f"wiki:{title}",
        "found": f"wiki:{title}" in p.sources,
        "body": p.sources.get(f"wiki:{title}", ""),
    }
    slot_spec = {
        "proof_obligations": [
            {"predicate_contains": "father", "object_contains": "Michael"},
        ],
        "disallowed_predicates": [],
        "preferred_method": "enumerate_filter",
    }
    resolve_entity_constraint_slot(
        plugin=p, slot_name="justice", slot_spec=slot_spec,
        candidates=["Amy Coney Barrett"],
    )
    # Every accepted triple must carry a mention distinct-or-equal-to canonical
    # (i.e., the propose_triples path filled it in, not the strict add_triple
    # path which leaves them implicitly equal).
    for t in p.triples:
        assert t.subject_mention, t
        # And the canonical is preserved.
        assert t.subject == "Amy Coney Barrett", t


def test_extract_person_name_starting_with() -> None:
    """Full-name extraction: finds person names like 'Michael Coney' but
    rejects bare 'Michael'."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        _extract_person_name_starting_with,
    )
    # Positive cases
    assert _extract_person_name_starting_with(
        "Amy Vivian Coney was born to Linda and Michael Coney.", "Michael"
    ) == "Michael Coney"
    assert _extract_person_name_starting_with(
        "Brett Kavanaugh's father was Everett Edward Kavanaugh Jr.", "Everett"
    ) in ("Everett Edward Kavanaugh Jr", "Everett Edward Kavanaugh Jr.")
    # Negative: bare first name with no surname → None
    assert _extract_person_name_starting_with(
        "Michael went to the store.", "Michael"
    ) is None
    # Negative: wrong first word
    assert _extract_person_name_starting_with(
        "His father is David Gorsuch.", "Michael"
    ) is None


def test_uniqueness_invariant_rejects_ambiguous_executor_result() -> None:
    """If multiple candidates satisfy all obligations, the executor must
    NOT select arbitrarily — it must escalate as ambiguous."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_entity_constraint_slot, make_classifier_claim_judge,
    )

    def stub_classifier(s, p, o, e, candidates, schema):
        if not candidates:
            return {"supported": False, "rejection_reason": "no candidates"}
        return {"supported": True, "chosen_span_id": 0,
                "binding_explanation": "stub", "rejection_reason": None}

    judge = make_classifier_claim_judge(witness_classifier=stub_classifier)
    p = KnowledgeQAPlugin(claim_judge=judge)
    # Deliberately constructed: BOTH candidates have triples that satisfy
    # 'appointed by Trump'.
    p.sources["wiki:A"] = "A was appointed by Donald Trump in 2017."
    p.sources["wiki:B"] = "B was appointed by Donald Trump in 2018."
    p._wiki_read = lambda title: {
        "title": title, "source_id": f"wiki:{title}",
        "found": f"wiki:{title}" in p.sources,
        "body": p.sources.get(f"wiki:{title}", ""),
    }
    slot_spec = {
        "proof_obligations": [
            {"predicate_contains": "appointed", "object_contains": "Trump"},
        ],
        "disallowed_predicates": [],
        "preferred_method": "enumerate_filter",
    }
    result = resolve_entity_constraint_slot(
        plugin=p, slot_name="test_slot", slot_spec=slot_spec,
        candidates=["A", "B"],
    )
    # Both candidates satisfy → uniqueness invariant rejects selection.
    assert result["value"] == "", (
        f"expected no selection due to ambiguity; got {result['value']!r}"
    )
    assert result["ambiguous"] is True
    assert result["satisfying_count"] == 2
    assert "AMBIGUOUS" in result["selection_reason"]


def test_full_name_extraction_blocks_bare_michael_false_positive() -> None:
    """REGRESSION: a Kavanaugh-style bio where the only 'Michael' is in
    'Michael Murphy' (great-great-grandfather) must NOT produce a false
    father-Michael triple, because the full-name extractor can't find a
    Michael-X name that the disallowed-cue filter would accept."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_entity_constraint_slot, make_classifier_claim_judge,
        extract_source_local_aliases,
    )

    def stub_classifier(s, p, o, e, candidates, schema):
        if not candidates:
            return {"supported": False, "rejection_reason": "no candidates"}
        return {"supported": True, "chosen_span_id": 0,
                "binding_explanation": "stub", "rejection_reason": None}

    judge = make_classifier_claim_judge(witness_classifier=stub_classifier)
    p = KnowledgeQAPlugin(claim_judge=judge)
    # The only 'Michael' in this bio is a great-great-grandfather, which
    # the disallowed-cue filter must block. With object_first_word='Michael'
    # the executor will try to extract 'Michael Murphy' as the father —
    # but the candidate generator filters out spans where great-great-
    # grandfather precedes Michael.
    p.sources["wiki:Brett Kavanaugh"] = (
        "Brett Kavanaugh's paternal great-great-grandfather was Michael Murphy. "
        "Brett Kavanaugh's father was Everett Edward Kavanaugh, a lawyer."
    )
    p._wiki_read = lambda title: {
        "title": title, "source_id": f"wiki:{title}",
        "found": f"wiki:{title}" in p.sources,
        "body": p.sources.get(f"wiki:{title}", ""),
    }
    aliases = extract_source_local_aliases(
        p.sources["wiki:Brett Kavanaugh"], "Brett Kavanaugh"
    )
    if aliases:
        p.source_aliases["wiki:Brett Kavanaugh"] = {"Brett Kavanaugh": aliases}

    slot_spec = {
        "proof_obligations": [
            {"predicate_contains": "father", "object_first_word": "Michael"},
        ],
        "disallowed_predicates": ["middle name"],
        "preferred_method": "enumerate_filter",
    }
    result = resolve_entity_constraint_slot(
        plugin=p, slot_name="justice", slot_spec=slot_spec,
        candidates=["Brett Kavanaugh"],
    )
    # Kavanaugh's row must NOT satisfy — the only Michael-X name available
    # is Michael Murphy, which the disallowed-cue filter blocks.
    row = result["candidate_table"][0]
    assert row["satisfies"] is False, (
        f"Kavanaugh should NOT satisfy father obligation; "
        f"row={row}, graph={[(t.subject, t.predicate, t.object) for t in p.triples]}"
    )


def test_relation_follow_certifies_from_existing_graph() -> None:
    """Graph-backed relation_follow: an accepted triple linking the
    consumed slot to a downstream entity certifies the slot directly."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_relation_follow_slot,
    )
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Amy Coney Barrett"] = (
        "Amy Coney Barrett attended Rhodes College."
    )
    p._add_triple(
        subject="Amy Coney Barrett", predicate="attended", object="Rhodes College",
        source="wiki:Amy Coney Barrett",
        excerpt="Amy Coney Barrett attended Rhodes College.",
    )
    slot_spec = {
        "subject_slot": "justice",
        "allowed_predicates": ["attended", "graduated from"],
        "disallowed_predicates": ["law school"],
        "disallowed_objects": ["Notre Dame Law School"],
        "uniqueness": "exactly_one",
    }
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="college", slot_spec=slot_spec,
        consumed_slots={"justice": "Amy Coney Barrett"},
    )
    assert result["value"] == "Rhodes College", result
    assert result["method"] == "relation_follow"
    assert result["supporting_triple_ids"] == ["t-0000"]


def test_relation_follow_rejects_law_school_via_disallowed_object() -> None:
    """A graph triple where the object is a disallowed entity (Notre Dame
    Law School) must NOT certify the college slot."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_relation_follow_slot,
    )
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Amy Coney Barrett"] = (
        "Amy Coney Barrett attended Notre Dame Law School."
    )
    p._add_triple(
        subject="Amy Coney Barrett", predicate="attended", object="Notre Dame Law School",
        source="wiki:Amy Coney Barrett",
        excerpt="Amy Coney Barrett attended Notre Dame Law School.",
    )
    slot_spec = {
        "subject_slot": "justice",
        "allowed_predicates": ["attended"],
        "disallowed_objects": ["Notre Dame Law School", "law school"],
        "uniqueness": "exactly_one",
    }
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="college", slot_spec=slot_spec,
        consumed_slots={"justice": "Amy Coney Barrett"},
    )
    assert result["value"] == "", result
    assert "no accepted graph triple" in result["selection_reason"]


def test_relation_follow_ambiguous_rejects() -> None:
    """Two graph triples both satisfying the relation pattern → uniqueness
    invariant rejects (won't arbitrarily pick one)."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_relation_follow_slot,
    )
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:X"] = "X attended A College. X attended B College."
    p._add_triple(subject="X", predicate="attended", object="A College",
                  source="wiki:X", excerpt="X attended A College.")
    p._add_triple(subject="X", predicate="attended", object="B College",
                  source="wiki:X", excerpt="X attended B College.")
    slot_spec = {
        "subject_slot": "justice",
        "allowed_predicates": ["attended"],
        "uniqueness": "exactly_one",
    }
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="college", slot_spec=slot_spec,
        consumed_slots={"justice": "X"},
    )
    assert result["value"] == ""
    assert result["ambiguous"] is True


def test_relation_follow_subject_aliases_for_founder_slot() -> None:
    """The founder slot consumes 'Rhodes College' but the relevant graph
    triple's subject is 'mock trial program at Rhodes College' — subject_aliases
    let the executor still match via 'mock trial'."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_relation_follow_slot,
    )
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Rhodes College"] = (
        "The Rhodes College mock trial program was founded by Marcus Pohlmann."
    )
    p._add_triple(
        subject="Rhodes College mock trial program",
        predicate="founded by", object="Marcus Pohlmann",
        source="wiki:Rhodes College",
        excerpt="The Rhodes College mock trial program was founded by Marcus Pohlmann.",
    )
    slot_spec = {
        "subject_slot": "college",
        "subject_aliases": ["mock trial program", "mock trial"],
        "allowed_predicates": ["founded by", "founded"],
        "uniqueness": "exactly_one",
    }
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="founder", slot_spec=slot_spec,
        consumed_slots={"college": "Rhodes College"},
    )
    assert result["value"] == "Marcus Pohlmann", result


def test_executor_justice_slot_regression() -> None:
    """REGRESSION: protect the justice-slot live success against future code
    changes. Encodes the four bugs we just fought through:

      - Kavanaugh middle-name confusion
      - Kavanaugh great-great-grandfather adjacency
      - Barrett birth-name / source-local alias requirement
      - Barrett nominated-by-Trump relation cue

    Uses the REAL V3 pipeline (candidate generator + mechanical validator)
    with a stub classifier that picks the first available candidate. This
    way the disallowed-cue filter, source-local aliases, and mechanical
    span checks all exercise — only the LLM classifier is stubbed."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_entity_constraint_slot, extract_source_local_aliases,
        make_classifier_claim_judge,
    )

    def stub_classifier(subject, predicate, obj, excerpt, candidates, schema):
        """Accept whichever candidate the generator put first (if any)."""
        if not candidates:
            return {"supported": False, "rejection_reason": "no candidates"}
        return {"supported": True, "chosen_span_id": 0,
                "binding_explanation": "stub picks first candidate",
                "rejection_reason": None}

    judge = make_classifier_claim_judge(witness_classifier=stub_classifier)
    p = KnowledgeQAPlugin(claim_judge=judge)
    p.sources["wiki:Neil Gorsuch"] = (
        "Neil Gorsuch was nominated by President Donald Trump in 2017. "
        "Gorsuch was born to David Gorsuch and Anne Gorsuch."
    )
    # Deliberately use 'Brett Kavanaugh' (no middle name) so the only
    # 'Michael' in the bio is the great-great-grandfather — that one is
    # what the disallowed-cue filter must block.
    p.sources["wiki:Brett Kavanaugh"] = (
        "Brett Kavanaugh was nominated by President Donald Trump in 2018. "
        "His paternal great-great-grandfather Michael Murphy immigrated. "
        "Brett Kavanaugh's father was Everett Edward Kavanaugh."
    )
    p.sources["wiki:Amy Coney Barrett"] = (
        "Amy Coney Barrett was nominated by President Donald Trump in 2020. "
        "Amy Vivian Coney was born in 1972 to Linda and Michael Coney."
    )
    # Stub the read method so the executor doesn't try to hit network.
    p._wiki_read = lambda title: {
        "title": title, "source_id": f"wiki:{title}",
        "found": f"wiki:{title}" in p.sources,
        "body": p.sources.get(f"wiki:{title}", ""),
    }
    # Populate source-local aliases (would normally happen inside _wiki_read).
    for source_id, body in p.sources.items():
        title = source_id.removeprefix("wiki:")
        aliases = extract_source_local_aliases(body, title)
        if aliases:
            p.source_aliases.setdefault(source_id, {})[title] = aliases

    slot_spec = {
        "name": "justice",
        "proof_obligations": [
            {"predicate_contains": "appointed", "object_contains": "Trump"},
            # Use object_first_word (full-name extraction): the executor
            # extracts a person name like 'Michael Coney' from the excerpt
            # rather than accepting bare 'Michael'.
            {"predicate_contains": "father", "object_first_word": "Michael"},
        ],
        "disallowed_predicates": ["middle name", "first name of"],
        "preferred_method": "enumerate_filter",
    }
    result = resolve_entity_constraint_slot(
        plugin=p, slot_name="justice", slot_spec=slot_spec,
        candidates=["Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett"],
    )

    # Barrett selected.
    assert "Barrett" in result["value"], f"expected Barrett; got {result['value']!r}"

    # Candidate table covers all three.
    cands = [r["candidate"] for r in result["candidate_table"]]
    assert cands == ["Neil Gorsuch", "Brett Kavanaugh", "Amy Coney Barrett"], cands

    # Kavanaugh does NOT satisfy father-is-Michael — the disallowed-cue
    # filter (great-great-grandfather → Michael Murphy) blocks the bad span.
    kav_row = next(r for r in result["candidate_table"]
                   if r["candidate"] == "Brett Kavanaugh")
    assert kav_row["obligation_status"]["1"]["satisfied"] is False, kav_row
    assert kav_row["satisfies"] is False

    # Gorsuch satisfies appointed_by-Trump but not father-is-Michael.
    gor_row = next(r for r in result["candidate_table"]
                   if r["candidate"] == "Neil Gorsuch")
    assert gor_row["obligation_status"]["0"]["satisfied"] is True
    assert gor_row["obligation_status"]["1"]["satisfied"] is False

    # Barrett satisfies BOTH obligations.
    bar_row = next(r for r in result["candidate_table"]
                   if r["candidate"] == "Amy Coney Barrett")
    assert bar_row["obligation_status"]["0"]["satisfied"] is True
    assert bar_row["obligation_status"]["1"]["satisfied"] is True


# -----------------------------------------------------------------------------
# mock_trial_v1 baseline lockdown — regression tests for the frozen success.
# -----------------------------------------------------------------------------
#
# These tests assert that every failure class fixed across the 10-round
# development arc remains structurally precluded. Future changes must
# preserve these invariants.

def test_baseline_father_triple_stores_full_name_not_bare_first() -> None:
    """REGRESSION (round V6 / full-name extraction):
    Once a father obligation uses object_first_word, the triple stored in
    the graph is the FULL person name ('Michael Coney'), not the bare
    first name ('Michael'). This is what makes the canonical graph entity
    re-usable downstream and prevents the bare-Michael attachment class."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_entity_constraint_slot, make_classifier_claim_judge,
        extract_source_local_aliases,
    )
    def stub(s, p, o, e, candidates, schema):
        if not candidates:
            return {"supported": False, "rejection_reason": "no candidates"}
        return {"supported": True, "chosen_span_id": 0,
                "binding_explanation": "stub", "rejection_reason": None}
    p = KnowledgeQAPlugin(claim_judge=make_classifier_claim_judge(witness_classifier=stub))
    p.sources["wiki:Amy Coney Barrett"] = (
        "Amy Coney Barrett (née Coney; born 1972) is an associate justice. "
        "Amy Vivian Coney was born in 1972 to Linda and Michael Coney."
    )
    p._wiki_read = lambda title: {
        "title": title, "source_id": f"wiki:{title}",
        "found": f"wiki:{title}" in p.sources,
        "body": p.sources[f"wiki:{title}"],
    }
    aliases = extract_source_local_aliases(
        p.sources["wiki:Amy Coney Barrett"], "Amy Coney Barrett"
    )
    if aliases:
        p.source_aliases["wiki:Amy Coney Barrett"] = {"Amy Coney Barrett": aliases}

    slot_spec = {
        "proof_obligations": [
            {"predicate_contains": "father", "object_first_word": "Michael"},
        ],
        "disallowed_predicates": ["middle name"],
        "preferred_method": "enumerate_filter",
    }
    result = resolve_entity_constraint_slot(
        plugin=p, slot_name="justice", slot_spec=slot_spec,
        candidates=["Amy Coney Barrett"],
    )
    # The accepted triple must store the FULL father name.
    father_triples = [t for t in p.triples if "father" in t.predicate.lower()]
    assert father_triples, "expected at least one father triple"
    for t in father_triples:
        assert t.object != "Michael", (
            f"REGRESSION: father triple stored bare 'Michael' instead of "
            f"full name: {(t.subject, t.predicate, t.object)}"
        )
        # The full name should contain Michael as a prefix.
        assert t.object.split()[0].lower() == "michael", t.object


def test_baseline_relation_follow_dedupes_by_object_value() -> None:
    """REGRESSION (round V8 / object-dedup): the chain executor wrote
    (Barrett, attended, Rhodes College) twice during one run. The
    relation_follow uniqueness invariant must NOT reject this as
    ambiguous — same object value = same slot value."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_relation_follow_slot,
    )
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:X"] = "X attended Rhodes College. X attended Rhodes College."
    p._add_triple(subject="X", predicate="attended", object="Rhodes College",
                  source="wiki:X", excerpt="X attended Rhodes College.")
    p._add_triple(subject="X", predicate="attended", object="Rhodes College",
                  source="wiki:X", excerpt="X attended Rhodes College.")
    slot_spec = {
        "subject_slot": "justice",
        "allowed_predicates": ["attended"],
        "uniqueness": "exactly_one",
    }
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="college", slot_spec=slot_spec,
        consumed_slots={"justice": "X"},
    )
    assert result["value"] == "Rhodes College", (
        f"duplicate triples with same object must resolve to one slot value; "
        f"got {result}"
    )
    assert not result.get("ambiguous"), result


def test_baseline_college_slot_rejects_notre_dame_law_school() -> None:
    """REGRESSION (round V7 / disallowed_objects): a triple with
    'Notre Dame Law School' as object must NOT certify the college slot."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_relation_follow_slot,
    )
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    p.sources["wiki:Barrett"] = "Barrett attended Notre Dame Law School."
    p._add_triple(subject="Amy Coney Barrett", predicate="attended",
                  object="Notre Dame Law School",
                  source="wiki:Barrett",
                  excerpt="Barrett attended Notre Dame Law School.")
    slot_spec = {
        "subject_slot": "justice",
        "allowed_predicates": ["attended"],
        "disallowed_objects": ["Notre Dame Law School", "law school"],
        "uniqueness": "exactly_one",
    }
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="college", slot_spec=slot_spec,
        consumed_slots={"justice": "Amy Coney Barrett"},
    )
    assert result["value"] == "", (
        f"college slot must reject Notre Dame Law School; got {result['value']!r}"
    )


def test_baseline_graph_salvage_certifies_from_existing_facts() -> None:
    """REGRESSION (round V8 / graph-salvage):
    A relation_follow slot must be certifiable directly from an already-
    accepted graph triple, even if no LLM acquisition was run."""
    from task_runtime.plugins.knowledge_qa import (  # noqa: PLC0415
        resolve_relation_follow_slot,
    )
    p = KnowledgeQAPlugin(claim_judge=disabled_claim_judge)
    # Pre-seed the graph: an LLM agent (in a prior turn) might have
    # added this, then failed its own packaging.
    p.sources["wiki:Barrett"] = "Amy Coney Barrett attended Rhodes College."
    p._add_triple(subject="Amy Coney Barrett", predicate="attended",
                  object="Rhodes College",
                  source="wiki:Barrett",
                  excerpt="Amy Coney Barrett attended Rhodes College.")
    slot_spec = {
        "subject_slot": "justice",
        "allowed_predicates": ["attended", "graduated from"],
        "disallowed_objects": ["Notre Dame Law School"],
        "uniqueness": "exactly_one",
    }
    result = resolve_relation_follow_slot(
        plugin=p, slot_name="college", slot_spec=slot_spec,
        consumed_slots={"justice": "Amy Coney Barrett"},
    )
    assert result["value"] == "Rhodes College"
    assert result["method"] == "relation_follow"
    # The cited triple must point back to an existing graph entry.
    cited_ids = list(result["cited_obligation_triples"].values())
    assert cited_ids[0] in {t.id for t in p.triples}


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                print(f"  FAIL {name}: {e}")
                return 1
    print("all knowledge_qa tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
