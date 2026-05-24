"""Knowledge-QA plugin: evidence-backed factual claims via Wikipedia.

A port of the universal pattern from `decompose_graph_agent.py`, hardened
through review:

  - add_triple writes (subject, predicate, object) facts to a graph
  - every add_triple cites a SOURCE that was previously registered by
    wiki_read (source-provenance check — invented sources are rejected)
  - the cited EXCERPT must literally be a substring of the source body
    (with whitespace normalization)
  - the subject AND object must literally appear in the excerpt
    (value-in-evidence invariant)
  - an LLM judge checks the predicate is actually supported by the
    excerpt (catches the `provides` vs `founded` mismatch class)
  - claim_evidence_alignment verifier accepts iff the final answer
    cites only triples that survived all of the above at write time

Deliberately omitted from v1 (QA refinements, not runtime/plugin invariants):
  - predicate canonicalization (graduated_from ≡ educated_at)
  - inference engine (1-hop instance_of derivation)
  - entity registry with fuzzy alias resolution
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

# wiki_tool is bundled inside the package (task_runtime/wiki_tool.py); imported
# lazily via relative imports from inside the functions that need it, so the
# package can still be inspected without paying the import cost up front.

from ..core import (  # noqa: E402
    Accept, Escalate, IncoherentContract, RejectWithRepairHint,
    TaskContract, TaskResult, Verdict,
)


_SUBJECTIVE_MARKERS = (
    "purpose of life",
    "meaning of life",
    "moving poem",
    "beautiful essay",
    "best possible",
    "should i pursue",
    "what should i",
    "decide for me",
    "your opinion",
)


# A judge takes (subject, predicate, object, excerpt) and returns
# (supported, reason). Pluggable so tests can substitute a deterministic stub.
ClaimJudge = Callable[[str, str, str, str], tuple[bool, str]]


def _normalize_ws(s: str) -> str:
    """Collapse all runs of whitespace to a single space, strip ends.

    Used so that a multi-line excerpt the model copied with reflowed
    whitespace still matches the source body. The semantic content is
    preserved; only formatting is normalized."""
    return " ".join(s.split())


_PERSON_HONORIFICS = (
    "Professor", "Prof.", "Prof",
    "Dr.", "Dr",
    "Mr.", "Ms.", "Mrs.", "Mr", "Ms", "Mrs",
    "Sir",
    "President", "Justice", "Judge", "Senator", "Representative",
    "Captain", "Colonel", "General", "Lieutenant",
    "Rev.", "Reverend",
)


def normalize_person_value(s: str) -> str:
    """Strip leading honorifics from a person-value string.
    Used for slot-value uniqueness comparisons so 'Marcus Pohlmann' and
    'Professor Marcus Pohlmann' resolve to the same canonical person.

    The original surface form is preserved in the triple — only the
    uniqueness key is normalized."""
    if not s:
        return s
    cleaned = s.strip()
    # Strip a leading honorific (one pass — handles "Dr. Smith", not
    # "Dr. Mr. Smith"; nested honorifics aren't expected in practice).
    for h in _PERSON_HONORIFICS:
        prefix = h + " "
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned


def _has_overlap(a: str, b: str) -> bool:
    """True iff `a` and `b` share a substring or a >=3-letter word token.

    Used by both the answer/cited-triple overlap check and the chain
    verifier's edge-matching pass. Lenient on purpose: 'Marcus Pohlmann'
    should match a triple about 'Pohlmann'; 'Amy Coney Barrett' should
    match a triple about 'Coney'."""
    if not a or not b:
        return False
    a_l, b_l = a.lower(), b.lower()
    if a_l in b_l or b_l in a_l:
        return True
    punct = ".,;:'\"()-"
    a_tokens = {w.strip(punct) for w in a_l.split()
                if len(w.strip(punct)) >= 3}
    b_tokens = {w.strip(punct) for w in b_l.split()
                if len(w.strip(punct)) >= 3}
    return bool(a_tokens & b_tokens)


# -----------------------------------------------------------------------------
# Relation schemas — per-predicate cue policies for the witness judge.
# -----------------------------------------------------------------------------

RELATION_SCHEMAS: dict[str, dict] = {
    "father": {
        "allowed_cues": ["father", "son of", "daughter of", "born to",
                         "parents", "father is", "father was",
                         # 'born' as a cue catches biographical openers like
                         # 'X was born in YEAR in PLACE, to PARENT and PARENT'
                         # where 'born to' isn't a literal substring. The
                         # disallowed-cue filter still blocks the
                         # 'great-great-grandfather Michael' false positive.
                         "born"],
        "disallowed_cues": ["grandfather", "great-grandfather",
                            "great-great-grandfather", "middle name",
                            "stepfather", "godfather", "father-in-law"],
        "object_kind": "person_name",
    },
    "appointed_by": {
        "allowed_cues": [
            "appointed by", "nominated by",
            "appointed", "nominated",
            "Trump-appointed", "Trump nominated",
            "nominee",          # works in "Trump's nominee" / "his nominee"
        ],
        "disallowed_cues": [
            # Senate confirmation isn't the appointment relation; the
            # appointment relation requires a Trump cue.
            "confirmed by the Senate", "Senate confirmed",
            # Adversarial uses of the same name shouldn't satisfy appointed_by.
            "criticized by", "opposed by", "denounced by",
        ],
        "object_aliases": [
            "Donald Trump", "Donald J. Trump", "President Trump",
            "President Donald Trump", "Trump",
        ],
    },
    "attended": {
        "allowed_cues": ["attended", "graduated from", "alma mater",
                         "studied at", "educated at"],
        "disallowed_cues": ["law school", "graduate school"],
    },
    "founded": {
        "allowed_cues": ["founded", "founded by", "established", "started"],
        "disallowed_cues": ["provides", "offers", "hosts"],
        # Compound-entity object aliases: programs/organizations are often
        # referenced with partial forms ("mock trial program" → "mock trial",
        # "the program"). The role-sensitive object policy excludes these by
        # default; this schema-driven list authorizes them only when the
        # context is right. Keys are canonical compound entities; values
        # are surface forms the source may use.
        "compound_object_aliases": {
            "mock trial program": [
                "mock trial program", "mock trial", "the program", "program",
            ],
            "Rhodes College mock trial program": [
                "Rhodes College mock trial program",
                "mock trial program",
                "the program",
            ],
        },
    },
}


def _schema_for_predicate(predicate: str) -> dict | None:
    """Look up a relation schema by predicate. Substring-matches in either
    direction so 'father_first_name' and 'father' both find the father schema."""
    p_l = (predicate or "").lower().strip()
    for key, schema in RELATION_SCHEMAS.items():
        if key in p_l or p_l in key:
            return schema
    return None


# -----------------------------------------------------------------------------
# Deterministic witness-candidate generation
# -----------------------------------------------------------------------------
#
# Instead of asking the LLM to FIND a support span (which is brittle —
# witness-extractor recall on true claims was the new failure mode after
# the V2 judge fixed precision), generate candidate spans MECHANICALLY
# from the source/excerpt by looking up subject aliases, object aliases,
# and predicate cues. The LLM is then a classifier: it picks one of the
# pre-computed candidates or says "unsupported." This raises recall
# without weakening the validator that catches the Kavanaugh-class
# false positives.

def _all_occurrences(body_l: str, terms: list[str]) -> list[tuple[int, str]]:
    """Return [(start_index, matched_term)] for every occurrence of any
    term in body_l. body_l should already be lowercased."""
    out: list[tuple[int, str]] = []
    for term in terms:
        if not term:
            continue
        t_l = term.lower()
        start = 0
        while True:
            idx = body_l.find(t_l, start)
            if idx == -1:
                break
            out.append((idx, term))
            start = idx + max(1, len(t_l))
    return out


def generate_witness_candidates(
    canonical_subject: str,
    predicate: str,
    canonical_object: str,
    body: str,
    schema: dict | None = None,
    subject_aliases: list[str] | None = None,
    max_candidates: int = 5,
) -> list[dict]:
    """Mechanically generate candidate witness spans.

    Each candidate is a dict {span, subject_mention, predicate_cue,
    object_mention, score} where the span is a verbatim substring of
    `body`. Disallowed-cue-preceding-object patterns are filtered out
    before scoring, so the Kavanaugh-class false positives never make
    it onto the candidate list at all.

    The LLM-classifier step (separate) will pick one of these by id or
    reject the whole claim.
    """
    if not body:
        return []

    # Build term lists. Subject uses the looser surname-allowed policy
    # (subjects often appear in prose as bare surnames). Object uses the
    # stricter policy that excludes bare last-words — schema.object_aliases
    # is the supported way to add surname-only object matching when it's
    # semantically intended (e.g. 'Trump' as alias for 'Donald Trump'
    # is explicit in the appointed_by schema).
    subj_terms = list(subject_aliases or []) + _mention_candidates(
        canonical_subject, role="subject",
    )
    obj_terms = _mention_candidates(canonical_object, role="object")
    if schema:
        obj_terms.extend(schema.get("object_aliases", []) or [])
        cues = list(schema.get("allowed_cues", []) or [])
        disallowed = list(schema.get("disallowed_cues", []) or [])
    else:
        cues = [predicate] if predicate else []
        disallowed = []

    # Dedupe.
    subj_terms = list(dict.fromkeys(t for t in subj_terms if t))
    obj_terms = list(dict.fromkeys(t for t in obj_terms if t))
    cues = list(dict.fromkeys(c for c in cues if c))

    body_l = body.lower()
    subj_positions = _all_occurrences(body_l, subj_terms)
    obj_positions = _all_occurrences(body_l, obj_terms)
    cue_positions = _all_occurrences(body_l, cues)
    disallowed_positions = _all_occurrences(body_l, disallowed)

    if not subj_positions or not obj_positions or not cue_positions:
        return []

    candidates: list[dict] = []

    for s_pos, s_term in subj_positions:
        for o_pos, o_term in obj_positions:
            # Skip pairs too far apart — likely separate topics.
            if abs(s_pos - o_pos) > 250:
                continue
            # Look for a cue near the pair.
            pair_lo = min(s_pos, o_pos)
            pair_hi = max(s_pos + len(s_term), o_pos + len(o_term))
            best_cue = None
            for c_pos, c_term in cue_positions:
                # Cue should be within ~30 chars of the subject/object pair,
                # or sandwiched between them.
                if pair_lo - 30 <= c_pos <= pair_hi + 30:
                    if best_cue is None or abs(c_pos - (pair_lo + pair_hi) / 2) < \
                       abs(best_cue[0] - (pair_lo + pair_hi) / 2):
                        best_cue = (c_pos, c_term)
            if best_cue is None:
                continue
            c_pos, c_term = best_cue

            # Construct the span: start at the prior sentence boundary,
            # end at the next sentence boundary after the trio.
            span_lo = min(pair_lo, c_pos)
            span_hi = max(pair_hi, c_pos + len(c_term))
            back = body.rfind(". ", 0, span_lo)
            if back >= 0 and span_lo - back < 200:
                span_lo = back + 2
            else:
                span_lo = max(0, span_lo - 20)
            fwd = body.find(". ", span_hi)
            if fwd >= 0 and fwd - span_hi < 200:
                span_hi = fwd + 1
            else:
                span_hi = min(len(body), span_hi + 20)
            span = body[span_lo:span_hi].strip()
            if not span or len(span) > 250:
                continue

            # Filter out spans where a disallowed cue precedes the object
            # mention. CRITICAL: check in BODY coordinates, not span
            # coordinates, because the span-snap logic may truncate the
            # start of the disallowed cue (e.g. span starts at "dfather"
            # instead of "great-great-grandfather"), causing in-span
            # substring search to miss the cue.
            #
            # Equally critical: use THIS candidate's specific object
            # position (o_pos), not the first occurrence after span_lo.
            # Otherwise an o_term like 'Marsh' that appears multiple times
            # (in 'Raymond Marsh' AND 'Samuel Marsh') would be checked
            # against the earliest match instead of the one this candidate
            # is actually built around — and the disallowed-cue check
            # silently passes the bad candidate.
            span_l = span.lower()
            disallowed_blocks = False
            # Check disallowed-cue precedence against THIS pairing's o_pos
            # AND against the canonical object's position in the body
            # (which may differ when o_term is a short alias like 'Marsh'
            # while the canonical is 'Samuel Marsh'). Without the
            # canonical-position check, a candidate built around
            # (subject="Elena Marsh", obj_alias="Marsh" at the wrong pos)
            # bypasses the great-grandfather filter that should have
            # blocked "Samuel Marsh".
            positions_to_check = {o_pos}
            canonical_obj_pos = body_l.find(canonical_object.lower())
            if canonical_obj_pos >= 0 and span_lo <= canonical_obj_pos < span_hi:
                positions_to_check.add(canonical_obj_pos)
            for o_body_pos in positions_to_check:
                if disallowed_blocks:
                    break
                for d_pos, d_term in disallowed_positions:
                    # Disallowed cue precedes the object within 60 chars
                    # AND is reasonably close to the span we're considering
                    # (within 80 chars of the span start, to allow for snap).
                    if (d_pos < o_body_pos
                            and 0 <= o_body_pos - (d_pos + len(d_term)) <= 60
                            and d_pos >= span_lo - 80):
                        disallowed_blocks = True
                        break
            if disallowed_blocks:
                continue

            # Score: shorter is better; cue between subject & object is better;
            # everything-in-one-sentence is better.
            score = 200 - len(span)
            if min(s_pos, o_pos) <= c_pos <= max(s_pos, o_pos):
                score += 50
            # Penalize if subject & object are in separate sentences.
            mid_periods = sum(1 for i in range(min(s_pos, o_pos),
                                              max(s_pos + len(s_term), o_pos + len(o_term)))
                              if i < len(body) and body[i] == ".")
            score -= 30 * mid_periods

            candidates.append({
                "span": span,
                "subject_mention": s_term,
                "predicate_cue": c_term,
                "object_mention": o_term,
                "score": score,
            })

    # Dedupe by span and return top-N.
    seen: set[str] = set()
    ranked: list[dict] = []
    for c in sorted(candidates, key=lambda x: -x["score"]):
        if c["span"] in seen:
            continue
        seen.add(c["span"])
        ranked.append(c)
        if len(ranked) >= max_candidates:
            break
    return ranked


# -----------------------------------------------------------------------------
# Claim-support judge: structured witness extractor + mechanical validation
# -----------------------------------------------------------------------------
#
# Replaces the prior yes/no LLM judge. The LLM is now asked to *show its
# work* — return an exact support_span, a subject_mention, a predicate_cue,
# an object_mention, and a binding_explanation. Then mechanical checks accept
# or reject based on:
#   1. support_span is a substring of the excerpt (verbatim)
#   2. support_span is short (cap at ~250 chars) — prevents fusing distant
#      sentences into a false-positive binding
#   3. subject_mention / predicate_cue / object_mention all appear in the span
#   4. predicate_cue isn't in the schema's disallowed_cues
#   5. no disallowed_cue precedes the object_mention in the span
#      (catches "great-great-grandfather Michael Murphy" claiming to be 'father')
#   6. subject_mention overlaps the canonical subject (via _has_overlap)
#   7. object_mention overlaps the canonical object OR a schema-defined alias
#
# Together these catch the Kavanaugh-father-Michael false positive: no
# *tight* span exists where 'father' binds 'Kavanaugh' to 'Michael'; the only
# spans containing both have 'great-great-grandfather' (disallowed cue)
# preceding 'Michael'.

def _llm_witness_extractor(subject: str, predicate: str, obj: str,
                           excerpt: str, schema: dict | None) -> dict:
    """Call an LLM to extract a structured witness for the claim."""
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "witness extractor requires the `openai` package. "
            "Install it, or pass claim_judge=disabled_claim_judge to "
            "KnowledgeQAPlugin() for offline use."
        ) from e
    client = OpenAI()

    schema_hint = ""
    if schema:
        allowed = ", ".join(schema.get("allowed_cues", []))
        disallowed = ", ".join(schema.get("disallowed_cues", []))
        obj_aliases = ", ".join(schema.get("object_aliases", []))
        bits = []
        if allowed:
            bits.append(f"ALLOWED cues for predicate {predicate!r}: {allowed}")
        if disallowed:
            bits.append(f"DISALLOWED cues (must NOT be the predicate_cue): {disallowed}")
        if obj_aliases:
            bits.append(f"Acceptable object surface forms: {obj_aliases}")
        if bits:
            schema_hint = "\n\nRelation schema:\n  " + "\n  ".join(bits)

    prompt = (
        "You are extracting a PROOF WITNESS for whether the EXCERPT "
        "directly supports the TRIPLE. Be strict: do not infer from "
        "adjacent unrelated mentions.\n\n"
        f"TRIPLE: subject={subject!r}, predicate={predicate!r}, object={obj!r}\n\n"
        f"EXCERPT (verbatim):\n{excerpt!r}"
        f"{schema_hint}\n\n"
        "Return JSON ONLY with these fields:\n"
        '  "supported": true|false\n'
        '  "support_span": EXACT verbatim substring of the excerpt (<= 200 chars) '
        'that proves the claim. The span MUST contain a subject mention, a predicate '
        'cue, and an object mention, in that order or close together.\n'
        '  "subject_mention": surface form referring to the subject (appears in span)\n'
        '  "predicate_cue": word/phrase in span expressing the relation (appears in span)\n'
        '  "object_mention": surface form referring to the object (appears in span)\n'
        '  "binding_explanation": brief: how the cue binds subject to object\n'
        '  "rejection_reason": null if supported, else why not\n\n'
        "Crucial: if the excerpt only mentions the object in an UNRELATED clause "
        "(e.g. naming a great-grandfather or middle name), set supported=false. "
        "Do NOT pick a long span just to make all three terms appear."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        return {"supported": False, "rejection_reason": f"malformed witness JSON: {e}"}


def _validate_witness(
    witness: dict,
    canonical_subject: str, predicate: str, canonical_object: str,
    excerpt: str, schema: dict | None,
) -> tuple[bool, str]:
    """Mechanical validation of an extracted witness. Pure — no LLM."""
    if not isinstance(witness, dict):
        return False, "witness is not a dict"
    if not witness.get("supported"):
        return False, "witness extractor said not supported: " + str(
            witness.get("rejection_reason", "(no reason)")
        )

    span = str(witness.get("support_span", "") or "").strip()
    if not span:
        return False, "witness missing support_span"
    # Span must be short — prevents fusing distant sentences. 250 leaves a
    # bit of slack over the 200-char requested cap.
    if len(span) > 250:
        return False, f"support_span too long ({len(span)} chars; cap 250)"

    excerpt_norm = _normalize_ws(excerpt)
    span_norm = _normalize_ws(span)
    if span_norm.lower() not in excerpt_norm.lower():
        return False, "support_span is not a verbatim substring of excerpt"

    subj_m = str(witness.get("subject_mention", "") or "").strip()
    obj_m = str(witness.get("object_mention", "") or "").strip()
    cue = str(witness.get("predicate_cue", "") or "").strip()
    if not (subj_m and obj_m and cue):
        return False, "witness missing subject_mention/object_mention/predicate_cue"

    span_l = span_norm.lower()
    if subj_m.lower() not in span_l:
        return False, f"subject_mention {subj_m!r} not in support_span"
    if obj_m.lower() not in span_l:
        return False, f"object_mention {obj_m!r} not in support_span"
    if cue.lower() not in span_l:
        return False, f"predicate_cue {cue!r} not in support_span"

    # The witness's mentions must actually refer to the canonical entities
    # (lenient: substring or shared-token overlap).
    if not _has_overlap(subj_m, canonical_subject):
        return False, (
            f"subject_mention {subj_m!r} doesn't overlap canonical subject "
            f"{canonical_subject!r}"
        )
    # For the object: either overlap with canonical, OR match a schema alias.
    obj_aliases = []
    if schema:
        obj_aliases = [a.lower() for a in schema.get("object_aliases", []) or []]
    canonical_obj_ok = _has_overlap(obj_m, canonical_object)
    alias_ok = any(_has_overlap(obj_m, a) for a in obj_aliases) if obj_aliases else False
    if not (canonical_obj_ok or alias_ok):
        return False, (
            f"object_mention {obj_m!r} doesn't overlap canonical object "
            f"{canonical_object!r} or any schema alias {obj_aliases}"
        )

    # Schema-based cue checks.
    if schema:
        disallowed = [c.lower() for c in schema.get("disallowed_cues", []) or []]
        for d in disallowed:
            if d in cue.lower():
                return False, (
                    f"predicate_cue {cue!r} matches disallowed cue {d!r} for "
                    f"predicate {predicate!r}"
                )
        # Disallowed cue appearing BETWEEN subject and object in the span
        # (or immediately preceding the object) is the canonical false-
        # positive pattern: "great-great-grandfather Michael Murphy" with
        # 'father' falsely claimed as the cue.
        for d in disallowed:
            idx = span_l.find(d)
            if idx < 0:
                continue
            obj_idx = span_l.find(obj_m.lower())
            # If the disallowed cue precedes the object_mention within ~60
            # chars, the binding is ambiguous — the object is more likely
            # to refer to the disallowed relation than to our predicate.
            if 0 <= idx < obj_idx <= idx + len(d) + 60:
                return False, (
                    f"disallowed cue {d!r} precedes object_mention {obj_m!r} in "
                    f"the support_span; binding is ambiguous"
                )

    return True, f"witness span: {span[:120]}"


def make_witness_claim_judge(witness_extractor=None):
    """Factory for a witness-based claim judge. Pass a custom extractor for
    offline testing; default uses the OpenAI LLM."""
    if witness_extractor is None:
        witness_extractor = _llm_witness_extractor

    def judge(subject: str, predicate: str, obj: str, excerpt: str,
              **kwargs) -> tuple[bool, str]:
        schema = _schema_for_predicate(predicate)
        try:
            witness = witness_extractor(subject, predicate, obj, excerpt, schema)
        except Exception as e:  # noqa: BLE001
            return False, f"witness extractor error: {type(e).__name__}: {e}"
        ok, reason = _validate_witness(
            witness, subject, predicate, str(obj), excerpt, schema,
        )
        return ok, reason

    return judge


def _llm_witness_classifier(subject: str, predicate: str, obj: str,
                            excerpt: str, candidates: list[dict],
                            schema: dict | None) -> dict:
    """Ask the LLM to pick one of the mechanically-generated candidates,
    or say unsupported. The LLM cannot invent a new span."""
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "witness classifier requires the `openai` package."
        ) from e
    client = OpenAI()
    candidate_lines = "\n".join(
        f"  [{i}] span: {c['span']!r}\n"
        f"       subject_mention={c['subject_mention']!r}  "
        f"predicate_cue={c['predicate_cue']!r}  object_mention={c['object_mention']!r}"
        for i, c in enumerate(candidates)
    )
    schema_hint = ""
    if schema:
        allowed = schema.get("allowed_cues", []) or []
        disallowed = schema.get("disallowed_cues", []) or []
        if allowed:
            schema_hint += f"\nALLOWED predicate cues for {predicate!r}: {allowed}"
        if disallowed:
            schema_hint += f"\nDISALLOWED cues (binding through these is wrong): {disallowed}"
    prompt = (
        "You are judging whether any candidate span directly supports the TRIPLE.\n\n"
        f"TRIPLE: subject={subject!r}, predicate={predicate!r}, object={obj!r}\n"
        f"{schema_hint}\n\n"
        f"CANDIDATES (you must choose one by id or say unsupported):\n{candidate_lines}\n\n"
        "Reply JSON ONLY:\n"
        '  "supported": true|false\n'
        '  "chosen_span_id": integer id of the chosen candidate (or null if unsupported)\n'
        '  "binding_explanation": brief: how the cue binds subject to object in the chosen span\n'
        '  "rejection_reason": null if supported, else why no candidate works\n\n'
        "Important: you may NOT invent a span. Pick from the candidate list, or say "
        "unsupported. Do not pick a span where the predicate cue binds subject to a "
        "different entity than the object."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        return {"supported": False, "rejection_reason": f"malformed JSON: {e}"}


def make_classifier_claim_judge(witness_classifier=None,
                                candidate_generator=None):
    """Factory: deterministic candidate generation + LLM classification.

    The plugin generates candidate witness spans mechanically from the
    excerpt (and ideally the full source body — for now, the excerpt is
    used). The LLM then chooses one or rejects. Mechanical validation
    runs on the chosen candidate.
    """
    if witness_classifier is None:
        witness_classifier = _llm_witness_classifier
    if candidate_generator is None:
        candidate_generator = generate_witness_candidates

    def judge(subject: str, predicate: str, obj: str, excerpt: str,
              *, subject_aliases: list[str] | None = None) -> tuple[bool, str]:
        schema = _schema_for_predicate(predicate)
        candidates = candidate_generator(
            subject, predicate, str(obj), excerpt, schema,
            subject_aliases=subject_aliases,
        )
        if not candidates:
            return False, (
                "no candidate witness spans could be generated from the excerpt "
                "(missing subject alias, object alias, or allowed predicate cue)"
            )
        try:
            choice = witness_classifier(subject, predicate, obj, excerpt,
                                        candidates, schema)
        except Exception as e:  # noqa: BLE001
            return False, f"classifier error: {type(e).__name__}: {e}"
        if not choice.get("supported"):
            return False, "classifier rejected: " + str(
                choice.get("rejection_reason") or "(no reason)"
            )
        chosen_id = choice.get("chosen_span_id")
        if not isinstance(chosen_id, int) or not (0 <= chosen_id < len(candidates)):
            return False, f"classifier returned invalid chosen_span_id={chosen_id!r}"
        chosen = candidates[chosen_id]
        # Re-validate mechanically. The chosen candidate already passed the
        # disallowed-precedes-object filter at generation time, so this
        # double-checks the LLM didn't lie about supported status or pick a
        # span whose validity the classifier shouldn't have endorsed.
        witness = {
            "supported": True,
            "support_span": chosen["span"],
            "subject_mention": chosen["subject_mention"],
            "predicate_cue": chosen["predicate_cue"],
            "object_mention": chosen["object_mention"],
            "binding_explanation": choice.get("binding_explanation", ""),
            "rejection_reason": None,
        }
        ok, reason = _validate_witness(
            witness, subject, predicate, str(obj), excerpt, schema,
        )
        return ok, reason

    return judge


def _default_claim_judge(subject: str, predicate: str, obj: str, excerpt: str,
                         **kwargs) -> tuple[bool, str]:
    """Witness-classifier claim judge (v3). Mechanical candidate generation
    + LLM classification + mechanical validation. Accepts subject_aliases
    via kwargs so the runtime can pass source-local aliases through to the
    candidate generator (without those, e.g. canonical 'Amy Coney Barrett'
    can't be grounded in a bio that uses her birth name 'Amy Vivian Coney')."""
    return make_classifier_claim_judge()(
        subject, predicate, obj, excerpt, **kwargs,
    )


def disabled_claim_judge(subject: str, predicate: str, obj: str, excerpt: str,
                         **kwargs) -> tuple[bool, str]:
    """A no-op judge that approves everything. Use for offline tests where
    the value-in-evidence check is what's under test. Accepts **kwargs to
    swallow any contextual extras (e.g. subject_aliases) the runtime threads."""
    return True, "claim judge disabled"


@dataclass
class _Triple:
    id: str
    subject: str           # canonical — what the graph stores as the entity
    predicate: str
    object: str            # canonical
    source: str
    excerpt: str
    judge_reason: str = ""
    # Surface forms that actually appear in the excerpt. Default to the
    # canonical (back-compat for the strict add_triple path); the
    # propose_triples path resolves these from aliases/heuristics so the
    # graph isn't forced to store surname-only entities.
    subject_mention: str = ""
    object_mention: str = ""

    def __post_init__(self) -> None:
        if not self.subject_mention:
            self.subject_mention = self.subject
        if not self.object_mention:
            self.object_mention = self.object

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source,
            "excerpt": self.excerpt,
            "judge_reason": self.judge_reason,
        }
        # Only surface mention fields when they differ from canonical, to
        # avoid noise in the common (strict) case where they're identical.
        if self.subject_mention and self.subject_mention != self.subject:
            d["subject_mention"] = self.subject_mention
        if self.object_mention and self.object_mention != self.object:
            d["object_mention"] = self.object_mention
        return d


# -----------------------------------------------------------------------------
# Mention resolution: canonical entity ↔ surface form found in the excerpt
# -----------------------------------------------------------------------------

def _mention_candidates(value: str, role: str = "subject") -> list[str]:
    """Role-sensitive surface-form candidates derived from a canonical value.

    For role='subject' (the default; preserves backward compatibility):
      - full canonical
      - last word ('Kavanaugh' from 'Brett Kavanaugh')
      - last two words (for 3+ word names: 'Coney Barrett' from 'Amy Coney Barrett')
    Subject surname-only aliases are common in legitimate prose
    ("Kavanaugh was appointed..."), and the source-local alias registry
    provides additional disambiguation where needed.

    For role='object':
      - full canonical
      - last two words (for 3+ word names — still useful for compound entity names)
      - NO bare last-word — too ambiguous (canonical 'Samuel Marsh'
        shouldn't be grounded by a span about 'Raymond Marsh' just because
        both share 'Marsh'). Callers that need surname-only object matching
        must supply explicit aliases via the entity spec or relation schema.

    Deliberately never includes the first word alone — generic 'Michael'
    for 'Michael Coney' would let unrelated sentences ground the entity.
    """
    if not value:
        return []
    cleaned = value.strip().replace(".", "")
    parts = [p for p in cleaned.split() if p]
    out = [value.strip()]
    if role == "subject":
        if len(parts) >= 2:
            out.append(parts[-1])                # surname
        if len(parts) >= 3:
            out.append(" ".join(parts[-2:]))     # last two
    else:  # role == "object"
        # Objects: only the last-two-words multi-token form, no bare surname.
        if len(parts) >= 3:
            out.append(" ".join(parts[-2:]))
    # Strip common honorific prefixes.
    HONORIFICS = ("Dr.", "Dr", "Professor", "President", "Justice", "Judge",
                  "Mr.", "Ms.", "Mrs.", "Sir", "Senator", "Mr", "Ms", "Mrs")
    for h in HONORIFICS:
        prefix = h + " "
        if value.startswith(prefix):
            out.append(value[len(prefix):])
    # Dedupe preserving longest-first ordering for matching.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    return sorted(deduped, key=len, reverse=True)


def _find_mention(value: str, excerpt: str,
                  aliases: list[str] | None = None,
                  role: str = "subject") -> str | None:
    """Return the longest mention (alias OR derived candidate) that
    actually appears in the excerpt, or None if no candidate matches.

    Whitespace is normalized for the comparison so reflowed excerpts still
    work. Caller-supplied aliases take priority over derived candidates.
    `role` controls whether bare-last-word aliases are included in the
    derived set — see _mention_candidates.
    """
    candidates: list[str] = []
    if aliases:
        candidates.extend(a for a in aliases if a)
    candidates.extend(_mention_candidates(value, role=role))
    # Dedupe + sort longest-first so we prefer the most specific match.
    seen: set[str] = set()
    ordered: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    ordered.sort(key=len, reverse=True)
    excerpt_norm_l = _normalize_ws(excerpt).lower()
    for cand in ordered:
        if _normalize_ws(cand).lower() in excerpt_norm_l:
            return cand
    return None


def extract_source_local_aliases(body: str, canonical: str) -> set[str]:
    """Extract source-local aliases for the page's main subject from its body.

    Conservative: looks at the first ~3000 chars (lead section), uses
    high-precision name-introduction patterns, and filters out single-token
    aliases that aren't the surname (so generic 'Michael' from a biography
    doesn't become an alias for that page's subject).

    The aliases returned are intended to be SOURCE-LOCAL: valid inside the
    given source_id only. The caller (KnowledgeQAPlugin._wiki_read) stores
    them under that source_id, and witness-candidate generation consults
    them only when grounding a triple cited against that source.
    """
    if not body or not canonical:
        return set()

    lead = body[:3000]
    aliases: set[str] = set()

    # Always include the canonical title + default mention candidates.
    aliases.add(canonical)
    canon_tokens = canonical.replace(".", "").split()
    if len(canon_tokens) >= 2:
        aliases.add(canon_tokens[-1])               # surname
        aliases.add(" ".join(canon_tokens[-2:]))    # last two
    surname_lower = canon_tokens[-1].lower() if canon_tokens else ""

    # Name pattern: TitleCase tokens, 2–5 of them, allows hyphens/apostrophes.
    name_re = r"[A-Z][a-z]+(?:[\s'-][A-Z][A-Za-z.'-]+){1,4}"

    # Pattern 1: "<Name> was born" — common bio opener for birth names.
    for m in re.finditer(rf"\b({name_re})\s+was born\b", lead):
        aliases.add(m.group(1).strip())
    # Pattern 2: "(born <Name>" — parenthetical birth name.
    for m in re.finditer(rf"\(born\s+({name_re})", lead):
        aliases.add(m.group(1).strip())
    # Pattern 3: "born as <Name>" — explicit birth name.
    for m in re.finditer(rf"\bborn as\s+({name_re})", lead):
        aliases.add(m.group(1).strip())
    # Pattern 4: "<Name>, née <Maiden>" — maiden name introduction.
    for m in re.finditer(rf"\bnée\s+({name_re})", lead):
        aliases.add(m.group(1).strip())

    # Filter: keep multi-token aliases; keep single-token only if surname.
    filtered: set[str] = set()
    for a in aliases:
        a_clean = a.strip()
        if not a_clean:
            continue
        toks = a_clean.split()
        if len(toks) >= 2:
            filtered.add(a_clean)
        elif len(toks) == 1 and toks[0].lower() == surname_lower:
            filtered.add(a_clean)

    # Aliases must actually appear somewhere in the body — useless otherwise.
    body_l = body.lower()
    return {a for a in filtered if a.lower() in body_l}


def _is_truncated_at_boundary(text: str, match_end: int,
                              source_text: str | None) -> bool:
    """Boundary-truncation guard: True iff the match ends at the text edge
    AND the source has more name-like content immediately past that edge.

    This catches the 'Professor Ma' class of false positive surfaced by A0
    baseline characterization: when `wiki_windows_around` truncates the
    window mid-name, an extracted name like 'Professor Ma' would otherwise
    be accepted even though the source continues with 'Marcus Pohlmann.'.

    Without source access, we conservatively assume non-truncation (the LLM
    judge has to catch it). With source access, we check whether the
    character(s) immediately after the matched name in the source are
    name-like (uppercase letter or apostrophe), and if so the match is
    truncated.
    """
    # The match must touch the right edge of the text (no whitespace/
    # punctuation between match end and text end).
    tail = text[match_end:]
    if tail.strip():
        # There's already content after the match — not at the boundary.
        return False
    if source_text is None:
        # No source to check; let the match through. The LLM judge / down-
        # stream validators have to catch it.
        return False
    matched_substr = text[max(0, match_end - 60):match_end]
    # Find where this substring appears in the source so we can look ahead.
    src_pos = source_text.find(matched_substr)
    if src_pos < 0:
        return False
    after_src = src_pos + len(matched_substr)
    # Look at the next few chars in the source.
    ahead = source_text[after_src:after_src + 6]
    if not ahead:
        return False
    # If the source continues with whitespace + uppercase letter (typical
    # for a name continuation like " Pohlmann."), OR continues directly
    # with a lowercase letter (truncated mid-token like "Ma" -> "Marcus"),
    # treat the match as truncated.
    if re.match(r"^[A-Za-z]", ahead):
        return True
    return False


def _extract_person_name_starting_with(
    text: str, first_word: str,
    source_text: str | None = None,
) -> str | None:
    """Find a multi-word person name in `text` whose first word equals
    `first_word` (case-insensitive). Returns the full name (typically 2–4
    capitalized words including the first), or None if no match.

    If `source_text` is provided, applies the boundary-truncation guard:
    matches at the text edge are rejected when the source continues with
    name-like content. Used by the entity-constraint executor to prevent
    the 'Professor Ma' class of false positive surfaced by A0 baseline
    characterization."""
    if not text or not first_word:
        return None
    # Person name: a TitleCase token equal to first_word, followed by 1-3
    # more capitalized name-like tokens. Allow apostrophes/hyphens, an
    # optional 'Jr.'/'Sr.'/'II'/'III' suffix.
    # Case-sensitive on subsequent tokens — only properly-capitalized words
    # (real names) can extend the match. Otherwise 'Michael went to the' would
    # match because lowercase words look like 'tokens' under IGNORECASE.
    # Inner token class deliberately EXCLUDES period — otherwise 'Murphy.'
    # matches as one token and the regex bridges the sentence boundary into
    # the next name (extracting 'Michael Murphy. Brett Kavanaugh's'). The
    # Jr./Sr. suffixes are matched via the explicit optional group.
    pattern = (
        rf"\b({re.escape(first_word)}"
        rf"(?:\s+[A-Z][A-Za-z'-]+){{1,3}}"
        rf"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?)\b"
    )
    for m in re.finditer(pattern, text):
        name = m.group(1).strip()
        if not (3 < len(name) < 60):
            continue
        if _is_truncated_at_boundary(text, m.end(1), source_text):
            continue
        return name
    return None


def _extract_person_name_avoiding_self(
    text: str, first_word: str, exclude_alias_spans: list[str],
    source_text: str | None = None,
) -> str | None:
    """Like _extract_person_name_starting_with, but rejects matches whose
    position falls inside any of the given alias spans in the text. Used
    to prevent extracting 'Michael Kavanaugh' from inside the subject's
    own canonical name 'Brett Michael Kavanaugh' and proposing it as
    that subject's father."""
    if not text or not first_word:
        return None
    pattern = (
        rf"\b({re.escape(first_word)}"
        rf"(?:\s+[A-Z][A-Za-z'-]+){{1,3}}"
        rf"(?:\s+(?:Jr\.?|Sr\.?|II|III|IV))?)\b"
    )
    excluded_ranges: list[tuple[int, int]] = []
    text_l = text.lower()
    for alias in (exclude_alias_spans or []):
        if not alias:
            continue
        a_l = alias.lower()
        cursor = 0
        while True:
            i = text_l.find(a_l, cursor)
            if i < 0:
                break
            excluded_ranges.append((i, i + len(alias)))
            cursor = i + max(1, len(a_l))
    for m in re.finditer(pattern, text):
        name = m.group(1).strip()
        if not (3 < len(name) < 60):
            continue
        m_start, m_end = m.span(1)
        if any(r_start <= m_start < r_end for r_start, r_end in excluded_ranges):
            continue
        if _is_truncated_at_boundary(text, m_end, source_text):
            continue
        return name
    return None


def _parse_entity_spec(spec) -> tuple[str, list[str]]:
    """Accept either a plain canonical string OR {canonical, aliases?}.
    Returns (canonical, aliases). Raises ValueError on malformed input."""
    if isinstance(spec, str):
        return spec.strip(), []
    if isinstance(spec, dict):
        canon = str(spec.get("canonical", "") or "").strip()
        if not canon:
            raise ValueError("entity spec missing 'canonical'")
        aliases = spec.get("aliases") or []
        if not isinstance(aliases, list):
            raise ValueError("entity spec 'aliases' must be a list")
        return canon, [str(a) for a in aliases]
    raise ValueError(
        f"entity spec must be a string or {{canonical, aliases}}; "
        f"got {type(spec).__name__}"
    )


class KnowledgeQAPlugin:
    name = "knowledge_qa"

    def __init__(self, claim_judge: ClaimJudge | None = None) -> None:
        self.triples: list[_Triple] = []
        # source_id (str) -> registered body text. Populated by wiki_read.
        # add_triple's excerpt must be a substring of one of these.
        self.sources: dict[str, str] = {}
        # Source-local alias registry: source_id -> canonical_entity ->
        # set of surface forms that may refer to that entity INSIDE that
        # source. Populated by _wiki_read using extract_source_local_aliases.
        # Crucial invariant: aliases here NEVER leak to other source_ids.
        self.source_aliases: dict[str, dict[str, set[str]]] = {}
        # Research ledger: every rejected add_triple attempt is recorded so
        # render_contract_context can surface them into repair attempts.
        self.rejected_attempts: list[dict] = []
        self._claim_judge: ClaimJudge = (
            claim_judge if claim_judge is not None else _default_claim_judge
        )

    def _subject_aliases_for(self, canonical: str, source: str) -> list[str]:
        """Return source-local aliases for `canonical` registered against
        `source`, or [] if none. SOURCE-LOCAL means: not valid globally;
        only used when the triple cites this exact source."""
        if not source or source not in self.source_aliases:
            return []
        return sorted(self.source_aliases[source].get(canonical, set()))

    # ---- tool: add_triple --------------------------------------------------

    def _record_rejection(self, subject, predicate, object, source, excerpt, error):
        """Append to the rejection ledger (capped). Surfaced into repair context."""
        if len(self.rejected_attempts) < 30:
            self.rejected_attempts.append({
                "subject": str(subject)[:80],
                "predicate": str(predicate)[:60],
                "object": str(object)[:80],
                "source": str(source)[:60],
                "excerpt": str(excerpt)[:160],
                "error": str(error)[:200],
            })

    def _add_triple_with_mentions(
        self, subject: str, predicate: str, object: str,
        source: str, excerpt: str,
        subject_mention: str, object_mention: str,
    ) -> dict:
        """Internal: validate-then-write with explicit surface mentions.

        Stage 1 (source provenance), Stage 2 (excerpt-in-source), and
        Stage 4 (predicate judge) work on the canonical subject/predicate/
        object. Stage 3 (value-in-evidence) uses the MENTIONS — what
        actually appears in the excerpt — so a triple about the canonical
        entity "Brett Kavanaugh" can be grounded by an excerpt that only
        says "Kavanaugh".

        Called both by `_add_triple` (strict: mention = canonical) and by
        `_propose_triples` (after mention resolution).
        """
        def _err(msg: str) -> dict:
            self._record_rejection(subject, predicate, object, source, excerpt, msg)
            return {"error": msg}

        if not subject or not predicate or not object:
            return _err("subject, predicate, and object are all required")
        if not source or not excerpt:
            return _err("source and excerpt are required for every triple")
        if not subject_mention or not object_mention:
            return _err("internal: subject_mention and object_mention required")

        # Stage 1: source provenance.
        if source not in self.sources:
            known = sorted(self.sources)
            return _err(
                f"unknown source {source!r}. Sources must be registered by "
                f"a prior wiki_read call. Currently registered: "
                f"{known if known else '(none)'}"
            )

        # Stage 2: excerpt-in-source.
        body_norm = _normalize_ws(self.sources[source])
        excerpt_norm = _normalize_ws(excerpt)
        if excerpt_norm not in body_norm:
            return _err(
                f"excerpt is not a substring of {source!r}. You must quote "
                f"actual text from the source body (whitespace is normalized "
                f"for the comparison, but no paraphrasing). Excerpt was: "
                f"{excerpt.strip()[:200]!r}"
            )

        # Stage 3: mention-in-evidence. Uses surface mentions, not canonical.
        excerpt_norm_l = _normalize_ws(excerpt).lower()
        missing = []
        if _normalize_ws(subject_mention).lower() not in excerpt_norm_l:
            missing.append(f"subject_mention {subject_mention!r}")
        if _normalize_ws(str(object_mention)).lower() not in excerpt_norm_l:
            missing.append(f"object_mention {object_mention!r}")
        if missing:
            return _err(
                f"excerpt does not contain {' and '.join(missing)} as a "
                f"literal substring. Either pick a different passage that "
                f"contains the named mentions, or use propose_triples with "
                f"aliases so the plugin can resolve mentions for you."
            )

        # Stage 4: predicate support — judge sees the CANONICAL subject/object
        # for semantic meaning, while the excerpt grounds via the mentions.
        # Source-local aliases (e.g. birth names extracted from the bio page)
        # are threaded through so witness-candidate generation can find a
        # subject mention even when the source uses a form like
        # "Amy Vivian Coney" instead of canonical "Amy Coney Barrett".
        local_aliases = self._subject_aliases_for(subject, source)
        try:
            supported, reason = self._claim_judge(
                subject, predicate, str(object), excerpt,
                subject_aliases=local_aliases,
            )
        except TypeError:
            # Older claim judges (e.g. disabled_claim_judge or user-supplied
            # 4-arg callables) don't accept the kwarg. Fall back gracefully.
            supported, reason = self._claim_judge(
                subject, predicate, str(object), excerpt,
            )
        if not supported:
            return _err(
                f"the excerpt does not support predicate {predicate!r}. "
                f"Judge: {reason}. "
                f"Pick a predicate that matches what the excerpt actually "
                f"says, or find a different excerpt that supports your predicate."
            )

        tid = f"t-{len(self.triples):04d}"
        triple = _Triple(
            id=tid,
            subject=subject,
            predicate=predicate,
            object=str(object),
            source=source,
            excerpt=excerpt[:1000],
            judge_reason=reason,
            subject_mention=subject_mention,
            object_mention=str(object_mention),
        )
        self.triples.append(triple)
        return {
            "added": triple.to_dict(),
            "graph_size": len(self.triples),
            "judge_reason": reason,
        }

    def _add_triple(self, subject: str, predicate: str, object: str,
                    source: str, excerpt: str) -> dict:
        """Strict low-level write. The caller must supply subject/object
        strings that literally appear in the excerpt. For canonical-vs-
        mention asymmetry (e.g. canonical='Brett Kavanaugh', excerpt says
        'Kavanaugh'), use propose_triples instead."""
        return self._add_triple_with_mentions(
            subject=subject, predicate=predicate, object=object,
            source=source, excerpt=excerpt,
            subject_mention=subject, object_mention=str(object),
        )

    _add_triple._tool_spec = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": "add_triple",
            "description": (
                "Record a (subject, predicate, object) fact in the graph. "
                "The `source` MUST be a source_id previously returned by "
                "wiki_read. The `excerpt` MUST be a literal passage from "
                "that source containing BOTH the subject and the object. "
                "An LLM judge then verifies the excerpt actually supports "
                "the predicate (so e.g. asserting `founded` while the "
                "excerpt says `provides` is rejected). Returns the new "
                "triple's id (t-NNNN) for citation in finish()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "source": {
                        "type": "string",
                        "description": (
                            "A source_id from a previous wiki_read response "
                            "(e.g. 'wiki:Rhodes_College')."
                        ),
                    },
                    "excerpt": {
                        "type": "string",
                        "description": (
                            "A passage QUOTED verbatim from the source body. "
                            "Must contain subject and object as literal "
                            "substrings."
                        ),
                    },
                },
                "required": ["subject", "predicate", "object", "source", "excerpt"],
            },
        },
    }

    # ---- tool: query_graph -------------------------------------------------

    def _query_graph(self, subject: str = "", predicate: str = "") -> dict:
        """Return triples already in the graph matching subject and/or predicate."""
        s_l = subject.lower() if subject else ""
        p_l = predicate.lower() if predicate else ""
        hits = []
        for t in self.triples:
            if s_l and s_l not in t.subject.lower():
                continue
            if p_l and p_l not in t.predicate.lower():
                continue
            hits.append(t.to_dict())
        return {"matches": hits, "total_in_graph": len(self.triples)}

    _query_graph._tool_spec = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": "query_graph",
            "description": (
                "Look up triples already in the graph by subject and/or "
                "predicate (case-insensitive substring). Cheap; use before "
                "wiki_read to skip duplicate work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                },
            },
        },
    }

    # ---- tool: wiki_search -------------------------------------------------

    def _wiki_search(self, term: str) -> dict:
        from .. import wiki_tool  # noqa: PLC0415
        if not term or not term.strip():
            return {"error": "term is required"}
        results = wiki_tool.search(term)
        return {
            "results": [
                {"title": r["title"], "snippet": r.get("description", "") or ""}
                for r in results
            ],
        }

    _wiki_search._tool_spec = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Full-text Wikipedia search. Use to pick a wiki_read target.",
            "parameters": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
            },
        },
    }

    # ---- tool: wiki_read ---------------------------------------------------

    def _wiki_read(self, title: str) -> dict:
        """Fetch a Wikipedia article body and REGISTER it as an evidence
        source. The returned source_id can then be cited by add_triple.

        If the source is already registered (e.g. from a prior wiki_read or
        from a test stub), use the cached body — both an efficiency win in
        production and what lets the offline executor tests stub
        plugin.sources directly without monkey-patching wiki_tool."""
        source_id = f"wiki:{title}"
        if source_id in self.sources:
            text = self.sources[source_id]
        else:
            from .. import wiki_tool  # noqa: PLC0415
            text = wiki_tool.fetch_page(title)
            if not text:
                return {"title": title, "found": False, "body": ""}
            self.sources[source_id] = text

        # Extract source-local aliases for the page's main subject (the
        # title). Birth-name, parenthetical, and "née" patterns picked up
        # here let witness generation match e.g. canonical='Amy Coney
        # Barrett' against an excerpt mentioning 'Amy Vivian Coney'
        # without compromising the canonical-vs-mention invariant.
        if source_id not in self.source_aliases:
            local_aliases = extract_source_local_aliases(text, title)
            if local_aliases:
                self.source_aliases.setdefault(source_id, {})[title] = local_aliases
        return {
            "title": title,
            "source_id": source_id,
            "found": True,
            "body": text,
            "total_chars": len(text),
            "note": (
                "The runtime truncates tool responses to ~8KB for the model's "
                "context. The FULL body is still registered server-side, so "
                "excerpts from any part of the article can be cited via "
                "add_triple. If the answer might be later in the article, use "
                "wiki_find."
            ),
        }

    _wiki_read._tool_spec = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": "wiki_read",
            "description": (
                "Fetch a Wikipedia article and register it as an evidence "
                "source. The response body is truncated to the runtime's "
                "tool-output cap (~8KB), but the FULL body is kept server-side "
                "for excerpt verification — so quoting a passage from anywhere "
                "in the article works, as long as you actually saw it. For "
                "later passages, use wiki_find."
            ),
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    }

    # ---- tool: wiki_find ---------------------------------------------------

    def _wiki_find(self, title: str, query: str, context: int = 1) -> dict:
        """Grep a Wikipedia article for lines containing `query`. Also
        registers the source so excerpts can be cited via add_triple."""
        if not title or not query:
            return {"error": "title and query are required"}
        # Reuse cached source if registered (avoids double-fetch + supports
        # test stubs).
        source_id = f"wiki:{title}"
        if source_id in self.sources:
            body = self.sources[source_id]
        else:
            from .. import wiki_tool  # noqa: PLC0415
            body = wiki_tool.fetch_page(title)
            if not body:
                return {"title": title, "found": False, "matches": []}
            self.sources[source_id] = body
        try:
            hits = wiki_tool.grep(title, query, line_context=context)
        except Exception as e:  # noqa: BLE001
            return {"error": f"grep failed: {e}"}
        return {
            "title": title,
            "source_id": source_id,
            "matches": hits,
            "count": len(hits) if isinstance(hits, list) else 0,
        }

    _wiki_find._tool_spec = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": "wiki_find",
            "description": (
                "Search within a Wikipedia article for passages containing "
                "a query term. Returns up to a few matching lines with "
                "surrounding context. Useful when the answer is in a long "
                "article and wiki_read's truncation might miss it. Also "
                "registers the article as a source for add_triple."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "query": {"type": "string"},
                    "context": {
                        "type": "integer",
                        "description": "Lines of context around each match (default 1)",
                    },
                },
                "required": ["title", "query"],
            },
        },
    }

    # ---- tool: wiki_windows_around -----------------------------------------

    def _wiki_windows_around(self, title: str, terms: list[str],
                             window_chars: int = 400) -> dict:
        """Find passages in a Wikipedia article around each occurrence of any
        given term. Returns excerpts that are GUARANTEED to be substrings of
        the source body — paste them directly into add_triple's `excerpt`
        argument without paraphrasing. Also registers the source.

        This converts "quote exact text from memory" (LLM-call expensive,
        often fails the excerpt-in-source check) into a deterministic
        retrieval task.
        """
        if not title or not terms:
            return {"error": "title and at least one term required"}
        if isinstance(terms, str):
            terms = [terms]
        source_id = f"wiki:{title}"
        if source_id in self.sources:
            body = self.sources[source_id]
        else:
            from .. import wiki_tool  # noqa: PLC0415
            body = wiki_tool.fetch_page(title)
            if not body:
                return {"title": title, "found": False, "windows": []}
            self.sources[source_id] = body

        body_l = body.lower()
        seen_ranges: list[tuple[int, int]] = []
        windows: list[dict] = []
        for term in terms[:10]:
            if not term:
                continue
            term_l = term.lower()
            cursor = 0
            while True:
                idx = body_l.find(term_l, cursor)
                if idx == -1:
                    break
                half = max(50, window_chars // 2)
                ws = max(0, idx - half)
                we = min(len(body), idx + len(term) + half)
                # Snap window start to the nearest preceding sentence boundary
                # so the excerpt starts at a clean sentence. Do NOT snap the
                # forward edge — windows must be allowed to cover multiple
                # sentences (a typical bio puts the appointed-by claim and
                # the father-of claim in consecutive sentences).
                back = body.rfind(". ", ws, idx)
                if back >= 0 and back > ws + half // 2:
                    ws = back + 2
                # De-dupe overlapping windows.
                if any(ws < pe and we > ps for ps, pe in seen_ranges):
                    cursor = idx + len(term)
                    continue
                seen_ranges.append((ws, we))
                excerpt = body[ws:we].strip()
                windows.append({
                    "matched_term": term,
                    "excerpt": excerpt,
                })
                cursor = idx + len(term)
                if len(windows) >= 8:
                    break
            if len(windows) >= 8:
                break

        return {
            "title": title,
            "source_id": source_id,
            "found": True,
            "windows": windows,
            "note": (
                "Each `excerpt` is a verbatim substring of the source body — "
                "you can paste it directly into add_triple as `excerpt` without "
                "paraphrasing. Pair it with subject/object that ALSO appear in "
                "the excerpt."
            ),
        }

    _wiki_windows_around._tool_spec = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": "wiki_windows_around",
            "description": (
                "Return passages of a Wikipedia article surrounding occurrences "
                "of any of the given terms. Each returned excerpt is a verbatim "
                "substring of the source body — guaranteed to pass the "
                "excerpt-in-source check when pasted into add_triple."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Look for windows around any of these terms.",
                    },
                    "window_chars": {
                        "type": "integer",
                        "description": "Approximate width of each window (default 400).",
                    },
                },
                "required": ["title", "terms"],
            },
        },
    }

    # ---- tool: propose_triples (batch add_triple) --------------------------

    def _propose_triples(self, triples: list[dict]) -> dict:
        """Batch-validate triples with EVIDENCE-BOUND MENTION RESOLUTION.

        Each entry's `subject`/`object` may be either:
          - a plain canonical string ("Brett Kavanaugh"), or
          - {"canonical": "Brett Kavanaugh", "aliases": ["Justice Kavanaugh"]}

        Before validation, the plugin finds the longest surface form
        (alias OR derived candidate like the last word) that actually
        appears in the excerpt. The graph stores the CANONICAL string;
        the recorded `subject_mention`/`object_mention` are the surface
        forms that grounded the triple.

        This lets the model write its INTENT (canonical entity) while
        Wikipedia prose using shortened names still passes evidence
        validation. If no mention candidate is found in the excerpt, the
        triple is rejected BEFORE the predicate judge runs.
        """
        if not isinstance(triples, list):
            return {"error": "`triples` must be a list of objects"}
        accepted: list[dict] = []
        rejected: list[dict] = []
        for i, t in enumerate(triples[:30]):
            if not isinstance(t, dict):
                rejected.append({"index": i, "error": "each entry must be an object"})
                continue
            try:
                subj_canon, subj_aliases = _parse_entity_spec(t.get("subject"))
                obj_canon, obj_aliases = _parse_entity_spec(t.get("object"))
            except ValueError as e:
                rejected.append({"index": i, "candidate": t, "error": str(e)})
                continue

            predicate = str(t.get("predicate", "") or "")
            source = str(t.get("source", "") or "")
            excerpt = str(t.get("excerpt", "") or "")

            # Combine caller-supplied aliases with source-local aliases (from
            # the source's bio opening, e.g. birth names). Source-local
            # aliases are SCOPED to this `source` — they never leak.
            local_subj_aliases = self._subject_aliases_for(subj_canon, source)
            combined_subj_aliases = list(subj_aliases) + local_subj_aliases

            # Resolve surface mentions BEFORE running the strict validator.
            # If a canonical can't be grounded by any candidate form, that's
            # a clean structural rejection — don't burn an LLM judge call.
            # Subjects use the looser surname-allowed policy; objects use
            # the stricter policy where bare last-words are excluded.
            subj_mention = _find_mention(
                subj_canon, excerpt, combined_subj_aliases, role="subject",
            )
            if subj_mention is None:
                tried = list(set(combined_subj_aliases) | set(_mention_candidates(subj_canon)))
                err = (
                    f"no mention of canonical subject {subj_canon!r} found in "
                    f"the excerpt. Tried: {tried}. Either pick an excerpt that "
                    f"contains one of these forms, or supply `aliases` with "
                    f"an alternative surface form the excerpt does use."
                )
                self._record_rejection(subj_canon, predicate, obj_canon, source, excerpt, err)
                rejected.append({"index": i, "candidate": t, "error": err})
                continue

            # Compound-object aliases from the relation schema: e.g.
            # 'mock trial program' is allowed to be mentioned as 'mock trial'
            # or 'the program' under the 'founded' schema. This lets
            # compound-entity object resolution succeed without weakening
            # the role-sensitive policy for person-object cases.
            compound_obj_aliases: list[str] = []
            ob_schema = _schema_for_predicate(predicate)
            if ob_schema and ob_schema.get("compound_object_aliases"):
                compound_obj_aliases = list(
                    ob_schema["compound_object_aliases"].get(obj_canon, [])
                )
            obj_mention = _find_mention(
                obj_canon, excerpt,
                obj_aliases + compound_obj_aliases,
                role="object",
            )
            if obj_mention is None:
                tried = list(set(obj_aliases) | set(_mention_candidates(obj_canon)))
                err = (
                    f"no mention of canonical object {obj_canon!r} found in "
                    f"the excerpt. Tried: {tried}. Either pick an excerpt that "
                    f"contains one of these forms, or supply `aliases` with "
                    f"an alternative surface form the excerpt does use."
                )
                self._record_rejection(subj_canon, predicate, obj_canon, source, excerpt, err)
                rejected.append({"index": i, "candidate": t, "error": err})
                continue

            result = self._add_triple_with_mentions(
                subject=subj_canon, predicate=predicate, object=obj_canon,
                source=source, excerpt=excerpt,
                subject_mention=subj_mention, object_mention=obj_mention,
            )
            if "error" in result:
                rejected.append({"index": i, "candidate": t, "error": result["error"]})
            else:
                accepted.append({"index": i, **result["added"]})
        return {
            "accepted": accepted,
            "rejected": rejected,
            "stats": {
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "graph_size_after": len(self.triples),
            },
        }

    _propose_triples._tool_spec = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": "propose_triples",
            "description": (
                "Batch version of add_triple WITH MENTION RESOLUTION. "
                "Submit the canonical entity names you want the graph to "
                "store; the plugin finds the surface form in the excerpt "
                "(e.g. canonical='Brett Kavanaugh' grounded by 'Kavanaugh', "
                "canonical='Donald Trump' grounded by 'President Trump'). "
                "If no surface mention is found, the triple is rejected "
                "BEFORE the predicate judge runs (cheaper). For first-name "
                "or single-token mentions that the default heuristics won't "
                "consider, pass an explicit `aliases` list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "triples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["subject", "predicate", "object",
                                         "source", "excerpt"],
                            "properties": {
                                "subject": {
                                    "description": (
                                        "Either a canonical string (e.g. 'Brett "
                                        "Kavanaugh') or {canonical, aliases: [...]} "
                                        "if the excerpt uses a non-standard form."
                                    ),
                                },
                                "predicate": {"type": "string"},
                                "object": {
                                    "description": (
                                        "Same shape as `subject` — string or "
                                        "{canonical, aliases}."
                                    ),
                                },
                                "source": {"type": "string"},
                                "excerpt": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["triples"],
            },
        },
    }

    # ---- Plugin interface --------------------------------------------------

    def tools(self):
        return {
            "add_triple":          self._add_triple,
            "propose_triples":     self._propose_triples,
            "query_graph":         self._query_graph,
            "wiki_search":         self._wiki_search,
            "wiki_read":           self._wiki_read,
            "wiki_find":           self._wiki_find,
            "wiki_windows_around": self._wiki_windows_around,
        }

    def verifiers(self):
        return {
            "claim_evidence_alignment": ClaimEvidenceAlignmentVerifier(self),
            "chain_completeness":       ChainCompletenessVerifier(self),
            "slot_certificate":         SlotCertificateVerifier(self),
        }

    def render_contract_context(self, contract: TaskContract) -> str:
        lines = [
            "PLUGIN: knowledge_qa",
            f"Graph state: {len(self.triples)} triple(s) recorded; "
            f"{len(self.sources)} source(s) registered.",
            "",
            "Workflow:",
            "  1. wiki_search to discover relevant article titles.",
            "  2. wiki_read or wiki_find to fetch source text and register",
            "     a source_id (e.g. 'wiki:Rhodes_College').",
            "  3. For each fact, pick a single sentence from a registered",
            "     source and call:",
            "       add_triple(subject, predicate, object, source, excerpt)",
            "     where:",
            "       - subject and object are short phrases that LITERALLY",
            "         appear in the excerpt",
            "       - source is the source_id from step 2",
            "       - excerpt is the quoted sentence (substring of the source)",
            "     The plugin checks: (a) source was registered, (b) excerpt",
            "     is a substring of the source body, (c) subject AND object",
            "     appear in the excerpt, (d) an LLM judge confirms the",
            "     predicate is actually supported by the excerpt.",
            "  4. finish(output={answer: '...', supporting_triple_ids: [...]})",
            "",
            "Important: do NOT paraphrase the excerpt. Quote it. If the",
            "excerpt says 'provides', do not assert predicate='founded' —",
            "the judge will reject it. Try predicate='provides' instead, or",
            "find a sentence that explicitly mentions founding.",
            "",
            "If the first source you read doesn't name a candidate answer,",
            "wiki_search for the likely answer entity directly (e.g. a",
            "person's name) and wiki_read THAT article. The answer is often",
            "on the person's own page, not on the organization's page.",
        ]
        if contract.output_schema:
            req = contract.output_schema.get("required", [])
            if req:
                lines.append("")
                lines.append("Shape: finish.output MUST include fields: " + ", ".join(req))

        # Consumed slots: if the runtime threaded slot values into the child's
        # inputs, surface them prominently. The child agent should USE these
        # values, not look them up again.
        consumed = (contract.inputs or {}).get("consumed_slots") or {}
        if consumed:
            lines.append("")
            lines.append("CONSUMED SLOTS (values produced by sibling sub-tasks; USE THESE):")
            for k, v in consumed.items():
                lines.append(f"  - {k!r} = {v!r}")
        expected = (contract.inputs or {}).get("expected_produces") or []
        if expected:
            lines.append("")
            lines.append(
                f"EXPECTED OUTPUT SLOTS: this task must populate slot_values "
                f"for {expected!r} in finish().output."
            )
        relation_lock = (contract.inputs or {}).get("relation_lock")
        if relation_lock:
            lines.append("")
            lines.append(
                f"RELATION LOCK: the relation you preserve MUST be "
                f"{relation_lock!r}. Do NOT substitute a different relation "
                f"(e.g. 'law school attended' is NOT 'college attended')."
            )

        # Slot-research shape: when a chain executor spawns a per-slot
        # research task, the inputs declare kind='slot_research_task' along
        # with the slot_spec (proof obligations, disallowed predicates,
        # preferred method). Render those as concrete obligations the
        # child must satisfy.
        if (contract.inputs or {}).get("kind") == "slot_research_task":
            slot_name = (contract.inputs or {}).get("slot_name", "<unnamed>")
            spec = (contract.inputs or {}).get("slot_spec", {}) or {}
            obligations = spec.get("proof_obligations", []) or []
            disallowed = spec.get("disallowed_predicates", []) or []
            method = spec.get("preferred_method", "")
            lines.append("")
            lines.append(f"SLOT-RESEARCH TASK — resolve the {slot_name!r} slot.")
            lines.append(f"  Description: {spec.get('description', '')}")
            if method:
                lines.append(f"  Preferred method: {method!r}")
            lines.append("")
            lines.append("PROOF OBLIGATIONS — your `value` must be the subject of triples")
            lines.append("matching ALL of these patterns (each will be checked at verification):")
            for i, ob in enumerate(obligations):
                pred = ob.get("predicate_contains", "")
                obj = ob.get("object_contains", "")
                desc = ob.get("description", "")
                line = f"  [{i}] predicate ~ {pred!r}"
                if obj:
                    line += f"  AND object ~ {obj!r}"
                if desc:
                    line += f"   ({desc})"
                lines.append(line)
            if disallowed:
                lines.append("")
                lines.append("DISALLOWED PREDICATES — these CANNOT satisfy any obligation:")
                for p in disallowed:
                    lines.append(f"  - {p!r}")
            lines.append("")
            lines.append("Your finish() output MUST be:")
            lines.append("  {")
            lines.append("    'value': '<the selected entity>',")
            lines.append("    'certificate': {")
            method_hint = method or "direct_lookup"
            lines.append(f"      'method': '<e.g. {method_hint}>',")
            lines.append("      'cited_obligation_triples': {")
            lines.append("        '<obligation_index>': '<t-NNNN triple id>',")
            lines.append("        ...")
            lines.append("      },")
            if method == "enumerate_filter":
                lines.append("      'candidate_table': [")
                lines.append("        {'candidate': '...', 'satisfies': true|false, 'triple_ids': [...]},")
                lines.append("        ...")
                lines.append("      ]")
            lines.append("    },")
            lines.append("    'supporting_triple_ids': ['t-NNNN', ...]   # all triples used")
            lines.append("  }")
            lines.append("")
            lines.append("Every triple in cited_obligation_triples MUST be a real triple")
            lines.append("you wrote via add_triple (or propose_triples) in this task. The")
            lines.append("slot_certificate verifier checks each obligation triple actually")
            lines.append("matches its pattern with `value` as the subject.")
            lines.append("")
            lines.append("EFFICIENT EVIDENCE ACQUISITION (use these to avoid rejection loops):")
            lines.append("  - wiki_windows_around(title, terms): get verbatim source excerpts")
            lines.append("    around any of the given terms. The returned excerpts are")
            lines.append("    GUARANTEED to pass the excerpt-in-source check.")
            lines.append("  - propose_triples(triples): batch-validate several triples at")
            lines.append("    once and get per-triple verdicts back, without spending one")
            lines.append("    LLM call per add_triple.")
            lines.append("  - query_graph(subject): check whether a fact is already in the")
            lines.append("    graph from earlier in this run; cite the existing t-id rather")
            lines.append("    than re-adding.")

            # Surface the rejection ledger so repair attempts don't re-make the
            # same mistakes. This is the user's "research ledger rendered into
            # repair prompts" — without it, each repair attempt is a fresh start.
            recent_rejections = self.rejected_attempts[-10:]
            accepted_summary = [
                {"id": t.id, "subject": t.subject, "predicate": t.predicate,
                 "object": t.object}
                for t in self.triples[-10:]
            ]
            if accepted_summary or recent_rejections:
                lines.append("")
                lines.append("RESEARCH LEDGER (carried across repair attempts):")
                if accepted_summary:
                    lines.append("  Already-accepted triples (cite their ids; do NOT re-add):")
                    for t in accepted_summary:
                        lines.append(
                            f"    {t['id']}: ({t['subject']!r}, "
                            f"{t['predicate']!r}, {t['object']!r})"
                        )
                if recent_rejections:
                    lines.append("  Recently-REJECTED attempts (avoid these patterns):")
                    for r in recent_rejections:
                        lines.append(
                            f"    ({r['subject']!r}, {r['predicate']!r}, "
                            f"{r['object']!r}) -> {r['error'][:120]}"
                        )

        # Multi-hop chain shape: when contract.inputs declares
        # kind='multi_hop_chain', render the slot/edge structure so the
        # ROOT agent knows what dataflow it must orchestrate.
        if (contract.inputs or {}).get("kind") == "multi_hop_chain":
            slots = (contract.inputs or {}).get("slots", {}) or {}
            edges = (contract.inputs or {}).get("edges", []) or []
            final_slot = (contract.inputs or {}).get("final_answer_slot", "")
            lines.append("")
            lines.append("MULTI-HOP CHAIN STRATEGY (typed dataflow):")
            lines.append("Slots to fill (in order):")
            for k, v in slots.items():
                desc = v.get("description", "") if isinstance(v, dict) else str(v)
                rl = v.get("must_preserve_relation", "") if isinstance(v, dict) else ""
                nr = v.get("not_relation", "") if isinstance(v, dict) else ""
                line = f"  - {k}: {desc}"
                if rl:
                    line += f"  [relation_lock={rl!r}]"
                if nr:
                    line += f"  [NOT {nr!r}]"
                lines.append(line)
            lines.append("")
            lines.append("Edges (each MUST be supported by a triple at verification):")
            for e in edges:
                lines.append(
                    f"  - {e.get('from')} --[{e.get('relation','?')}]--> {e.get('to')}"
                )
            lines.append("")
            lines.append("How to execute:")
            lines.append("  1. Spawn one child per slot, in dependency order.")
            lines.append("     Each child should declare:")
            lines.append("       consumes=[slot names from earlier in the chain]")
            lines.append("       produces=[the slot it fills]")
            lines.append("       relation_lock='<the must_preserve_relation>'")
            lines.append("       verifier='claim_evidence_alignment'")
            lines.append("     The runtime will inject consumed slot values into")
            lines.append("     each child's inputs.consumed_slots.")
            lines.append("  2. Each child's finish() output MUST include:")
            lines.append("       slot_values={its_slot: '<value>'}")
            lines.append("     so the runtime can thread it to dependent children.")
            lines.append("  3. After all children complete, ROOT calls finish with:")
            lines.append("       output={")
            lines.append("         answer: <value of final_answer_slot>,")
            lines.append("         supporting_triple_ids: [all cited triples],")
            lines.append("         slot_values: {<every slot>: <value>}")
            lines.append("       }")
            lines.append(f"  4. The final answer = slot_values[{final_slot!r}].")
            lines.append("")
            lines.append("The chain_completeness verifier will check:")
            lines.append("  - every slot is populated")
            lines.append("  - every edge has a triple in the graph linking the slot values")
            lines.append("  - the answer matches slot_values[final_answer_slot]")

        # Constraint-resolution shape: when the contract input declares
        # kind='entity_constraint_resolution', inject the enumerate→fill→
        # filter strategy explicitly. This is a *reusable research primitive*
        # — not specific to any one question. The pattern is:
        #   "find an X whose Y is Z"  →
        #     enumerate candidates of class X,
        #     fill attribute Y for each,
        #     filter to those where Y == Z.
        if (contract.inputs or {}).get("kind") == "entity_constraint_resolution":
            ec = contract.inputs.get("entity_class", "<unspecified>")
            cs = contract.inputs.get("constraints", []) or []
            lines.append("")
            lines.append("CONSTRAINT-RESOLUTION STRATEGY (enumerate -> fill -> filter):")
            lines.append(f"  Entity class: {ec}")
            lines.append("  Constraints to satisfy:")
            for c in cs:
                lines.append(f"    - {c.get('relation', '?')} = {c.get('value', '?')!r}")
            lines.append("")
            lines.append("  Direct search for the fully-constrained entity often fails")
            lines.append("  (Wikipedia rarely has an article titled with all the constraints).")
            lines.append("  Use this strategy instead:")
            lines.append("    1. ENUMERATE: list the candidate members of the entity class.")
            lines.append("       Usually a wiki_read of a 'List of ...' article, or naming")
            lines.append("       them from prior knowledge then wiki_reading each.")
            lines.append("    2. FILL: for EACH candidate, wiki_read their page and look up")
            lines.append("       the constrained attribute. Record a triple per candidate so")
            lines.append("       the proof tree shows the enumeration was complete.")
            lines.append("    3. FILTER: select the candidate(s) satisfying all constraints.")
            lines.append("  If your output_schema accepts a 'candidates' field, populate it")
            lines.append("  with the table of (name, attribute_value, satisfies, triple_ids)")
            lines.append("  so the answer is independently auditable.")
        return "\n".join(lines)

    def coherence_check(self, contract: TaskContract):
        if contract.verifier and contract.verifier not in self.verifiers():
            return IncoherentContract(
                reason=(
                    f"unknown verifier '{contract.verifier}' for knowledge_qa "
                    f"(available: {sorted(self.verifiers())})"
                )
            )
        if contract.verifier == "claim_evidence_alignment":
            goal_l = (contract.goal or "").lower()
            for marker in _SUBJECTIVE_MARKERS:
                if marker in goal_l:
                    return IncoherentContract(
                        reason=(
                            f"goal contains subjective marker {marker!r} but "
                            f"verifier='claim_evidence_alignment' requires "
                            f"evidence-settleable claims."
                        )
                    )
        return None


# -----------------------------------------------------------------------------
# Verifier
# -----------------------------------------------------------------------------

@dataclass
class ClaimEvidenceAlignmentVerifier:
    plugin: KnowledgeQAPlugin

    def check(self, contract: TaskContract, result: TaskResult, workspace: dict) -> Verdict:
        output = result.output or {}
        if not isinstance(output, dict):
            return RejectWithRepairHint(
                reason="output must be an object with answer + supporting_triple_ids",
                hint=(
                    "Call finish(output={'answer': '<your answer>', "
                    "'supporting_triple_ids': ['t-NNNN', ...]})."
                ),
                missing_requirements=["object output with answer+supporting_triple_ids"],
            )
        triple_ids = output.get("supporting_triple_ids", [])
        if not isinstance(triple_ids, list) or not triple_ids:
            return RejectWithRepairHint(
                reason="supporting_triple_ids is empty or missing",
                hint=(
                    "Your answer must cite at least one triple from the graph. "
                    "Use add_triple() to record a fact (it returns a t-NNNN id), "
                    "then include that id in supporting_triple_ids."
                ),
                missing_requirements=["non-empty supporting_triple_ids"],
            )
        known = {t.id: t for t in self.plugin.triples}
        unknown = [tid for tid in triple_ids if tid not in known]
        if unknown:
            return RejectWithRepairHint(
                reason=f"unknown triple ids: {unknown}",
                hint=(
                    f"These triple ids don't exist in the graph: {unknown}. "
                    f"Only use ids returned by add_triple() in this run. "
                    f"Currently {len(self.plugin.triples)} triple(s) exist."
                ),
                missing_requirements=[f"valid triple id {tid}" for tid in unknown],
            )
        cited = [known[tid] for tid in triple_ids]

        # Answer/cited-triple overlap: the answer entity must actually appear
        # in at least one cited triple. Without this, the model can "succeed"
        # by elimination — citing triples about eliminated candidates and
        # naming the un-grounded remaining one as the answer.
        answer = str(output.get("answer", "") or "")
        if answer:
            overlap_ok = any(
                _has_overlap(answer, t.subject) or _has_overlap(answer, t.object)
                for t in cited
            )
            if not overlap_ok:
                cited_summary = "; ".join(
                    f"({t.subject!r}, {t.predicate!r}, {t.object!r})"
                    for t in cited[:5]
                )
                return RejectWithRepairHint(
                    reason=(
                        f"answer {answer[:80]!r} does not overlap with any "
                        f"cited triple's subject or object"
                    ),
                    hint=(
                        "Your cited triples must include at least one whose "
                        "subject or object mentions the answer entity. "
                        "Eliminating wrong candidates is not the same as "
                        "grounding the right one — add (and cite) a triple "
                        "ABOUT the answer entity itself.\n"
                        f"Currently cited: {cited_summary}"
                    ),
                    missing_requirements=[
                        "at least one cited triple mentions the answer entity"
                    ],
                )

        record = {
            "verifier": "claim_evidence_alignment",
            "answer": answer,
            "cited_triple_ids": triple_ids,
            "cited_triples": [t.to_dict() for t in cited],
            "graph_size_at_verification": len(self.plugin.triples),
            "sources_registered": sorted(self.plugin.sources.keys()),
        }
        return Accept(
            reason=(
                f"{len(triple_ids)} cited triple(s); all exist and survived "
                f"source-provenance, excerpt-containment, value-in-evidence, "
                f"predicate-support, and answer-overlap checks"
            ),
            record=record,
        )


# -----------------------------------------------------------------------------
# EntityConstraintResolutionExecutor — compiled slot acquisition
# -----------------------------------------------------------------------------

def resolve_entity_constraint_slot(
    plugin: "KnowledgeQAPlugin",
    slot_name: str,
    slot_spec: dict,
    candidates: list[str],
) -> dict:
    """Mechanical enumerate→fill→filter for an entity-constraint slot.

    The executor owns the workflow. The LLM is NOT invoked at the executor
    level. For each candidate × each obligation, the executor:

      1. wiki_reads the candidate's bio page (registering the source)
      2. wiki_windows_around the bio to find passages containing the
         candidate mention + the obligation's object term
      3. propose_triples for each window (canonical-vs-mention resolution
         handles surname-only excerpts) — the predicate judge inside
         propose_triples is the only LLM cost
      4. Records which obligations are satisfied and which triples are
         the per-obligation evidence

    Then selects the candidate where all obligations are satisfied AND
    no disallowed predicate appears in the evidence chain. Returns the
    candidate_table and a fully-formed SlotCertificate output.

    This addresses both bottlenecks from the prior live run:
      - "model uses add_triple incorrectly" — executor doesn't expose
        add_triple at all; only propose_triples is used.
      - "list rows do not contain Trump" — executor reads BIOGRAPHY
        pages, not list pages, so excerpts contain the real terms.
    """
    obligations = slot_spec.get("proof_obligations", []) or []
    disallowed = [p.lower() for p in (slot_spec.get("disallowed_predicates", []) or [])]

    # Step 1: register every candidate's bio page as a source.
    for cand in candidates:
        plugin._wiki_read(cand)

    candidate_rows: list[dict] = []

    for candidate in candidates:
        source_id = f"wiki:{candidate}"
        if source_id not in plugin.sources:
            candidate_rows.append({
                "candidate": candidate,
                "satisfies": False,
                "reason": "no biography page registered",
                "obligation_status": {},
                "accepted_triples": [],
                "rejected_triples": [],
            })
            continue

        ob_status: dict = {}
        accepted_for_cand: list[str] = []
        rejected_for_cand: list[dict] = []

        for i, ob in enumerate(obligations):
            pred_pat = (ob.get("predicate_contains", "") or "").strip()
            obj_target = (ob.get("object_contains", "") or "").strip()
            # object_first_word: instead of accepting bare "Michael" as the
            # object, require the object to be a multi-word person name
            # whose first word equals the given string. Closes the
            # "father_first_name=Michael" attachment problem where bare
            # 'Michael' can attach to a great-grandfather mention or to
            # the subject's own middle name.
            obj_first_word = (ob.get("object_first_word", "") or "").strip()

            # Look up a relation schema for the obligation's predicate. This
            # lets the executor search for SCHEMA CUES (e.g. 'nominated',
            # 'Trump-appointed', 'nominee') in addition to the obligation's
            # literal predicate string, AND search for OBJECT ALIASES (e.g.
            # 'President Trump' for canonical 'Donald Trump'). Wikipedia
            # rarely uses one literal cue per relation.
            ob_schema = _schema_for_predicate(pred_pat) if pred_pat else None

            # Build search terms for wiki_windows_around.
            search_terms = [candidate]
            if obj_target:
                search_terms.append(obj_target)
            if obj_first_word:
                # Critical for full-name extraction obligations: include the
                # target first name as a search term so windows around e.g.
                # "Michael Coney" in the bio surface and the extractor can
                # then find the full person name.
                search_terms.append(obj_first_word)
            if pred_pat:
                search_terms.append(pred_pat)
            if ob_schema:
                # Add the schema's allowed cues and object aliases as
                # additional search terms — windows around any of these
                # surfaces give the judge more material to certify with.
                search_terms.extend(
                    (ob_schema.get("allowed_cues") or [])[:6]
                )
                search_terms.extend(
                    (ob_schema.get("object_aliases") or [])[:3]
                )

            windows_result = plugin._wiki_windows_around(candidate, search_terms)
            windows = windows_result.get("windows", []) or []

            chosen_triple_id: str | None = None
            for w in windows:
                excerpt = w.get("excerpt", "")
                if not excerpt:
                    continue
                if not (obj_target or obj_first_word):
                    continue
                # Relation-aware pre-skip.
                excerpt_l = excerpt.lower()
                if ob_schema:
                    cues = [c.lower() for c in (ob_schema.get("allowed_cues") or [])]
                    if cues and not any(c in excerpt_l for c in cues):
                        continue
                else:
                    if pred_pat and pred_pat.lower() not in excerpt_l:
                        continue

                # Build the proposal. If object_first_word is set, extract
                # a multi-word person name starting with that word from the
                # window and use it as the object — that way the graph
                # stores `(Barrett, father, Michael Coney)` rather than
                # `(Barrett, father, Michael)`. The "Michael" alone class
                # of false positive becomes structurally impossible because
                # we never propose a bare-first-name object.
                if obj_first_word:
                    self_spans = [candidate]
                    self_spans.extend(
                        plugin._subject_aliases_for(candidate, source_id)
                    )
                    # Pass the full source body so the truncation guard can
                    # detect when wiki_windows_around cut a name in half
                    # (the 'Professor Ma' class of false positive).
                    proposed_object = _extract_person_name_avoiding_self(
                        excerpt, obj_first_word,
                        exclude_alias_spans=self_spans,
                        source_text=plugin.sources.get(source_id),
                    )
                    if not proposed_object:
                        continue
                else:
                    proposed_object = obj_target
                proposal = {
                    "subject": candidate,           # canonical full name
                    "predicate": pred_pat or "relates_to",
                    "object": proposed_object,
                    "source": source_id,
                    "excerpt": excerpt,
                }
                result = plugin._propose_triples([proposal])
                accepted = result.get("accepted", [])
                rejected = result.get("rejected", [])
                if accepted:
                    triple = accepted[0]
                    # Reject if predicate is in disallowed list (defensive —
                    # shouldn't fire here since we built the predicate from
                    # obligation hint, but keeps the executor honest).
                    pl = str(triple.get("predicate", "")).lower()
                    if any(d in pl or pl in d for d in disallowed):
                        rejected_for_cand.append({
                            "obligation": i,
                            "reason": f"predicate {triple['predicate']!r} disallowed",
                        })
                        continue
                    chosen_triple_id = triple["id"]
                    accepted_for_cand.append(triple["id"])
                    break
                if rejected:
                    rejected_for_cand.append({
                        "obligation": i,
                        "candidate_proposal": proposal,
                        "error": rejected[0].get("error", "")[:200],
                    })

            ob_status[i] = {
                "obligation": ob,
                "triple_id": chosen_triple_id,
                "satisfied": chosen_triple_id is not None,
            }

        satisfies = bool(ob_status) and all(s["satisfied"] for s in ob_status.values())
        reason = (
            "all obligations satisfied" if satisfies else
            "unsatisfied obligations: " + ", ".join(
                str(i) for i, s in ob_status.items() if not s["satisfied"]
            )
        )
        candidate_rows.append({
            "candidate": candidate,
            "satisfies": satisfies,
            "reason": reason,
            "obligation_status": {
                str(i): {
                    "obligation": v["obligation"],
                    "triple_id": v["triple_id"],
                    "satisfied": v["satisfied"],
                }
                for i, v in ob_status.items()
            },
            "accepted_triples": accepted_for_cand,
            "rejected_triples": rejected_for_cand[:5],
        })

    # UNIQUENESS INVARIANT for enumerate_filter:
    #   - 0 satisfying candidates → escalate (no answer)
    #   - >1 satisfying candidates → escalate (ambiguity — usually means a
    #     verifier-strength gap let a false positive through; selecting the
    #     first arbitrarily would propagate the wrong upstream value
    #     through the chain)
    #   - exactly 1 → accept
    satisfying_rows = [r for r in candidate_rows if r["satisfies"]]

    selected_row = None
    selected_value = ""
    selection_reason = "no candidate satisfies all obligations"
    ambiguous = False

    if len(satisfying_rows) == 1:
        selected_row = satisfying_rows[0]
        selected_value = selected_row["candidate"]
        selection_reason = "exactly one candidate satisfies all obligations"
    elif len(satisfying_rows) > 1:
        ambiguous = True
        names = [r["candidate"] for r in satisfying_rows]
        selection_reason = (
            f"AMBIGUOUS: {len(satisfying_rows)} candidates satisfy all "
            f"obligations ({names}). Slot will not be certified — a "
            f"verifier-strength gap likely let a false positive through. "
            f"Resolve the conflicting evidence (e.g. tighten predicate "
            f"judge or disallowed-cue filter) and re-run."
        )

    cited_obligation_triples: dict = {}
    if selected_row:
        for i, s in selected_row["obligation_status"].items():
            if s["triple_id"]:
                cited_obligation_triples[i] = s["triple_id"]

    all_triple_ids: list[str] = []
    for r in candidate_rows:
        all_triple_ids.extend(r["accepted_triples"])

    return {
        "slot_name": slot_name,
        "value": selected_value,
        "method": "enumerate_filter",
        "candidate_table": candidate_rows,
        "selected_row": selected_row,
        "cited_obligation_triples": cited_obligation_triples,
        "supporting_triple_ids": all_triple_ids,
        "sources_registered": [f"wiki:{c}" for c in candidates],
        "selection_reason": selection_reason,
        "ambiguous": ambiguous,
        "satisfying_count": len(satisfying_rows),
    }


# -----------------------------------------------------------------------------
# Graph-backed relation_follow slot executor
# -----------------------------------------------------------------------------
#
# Companion to resolve_entity_constraint_slot. Where that one is for
# enumerate→filter slots (with known candidates), this one is for
# relation_follow slots: given a consumed slot value, find a graph triple
# linking it to the next slot's value through an allowed relation.
#
# Crucially, this executor treats the graph as an ACTIVE proof substrate.
# If an accepted triple already exists in the graph that satisfies the
# slot's relation pattern, the slot is certified directly — no need for
# the LLM child to call finish() with exactly the right shape. The
# previous chain run found (Amy Coney Barrett, attended, Rhodes College)
# in the graph but the LLM child failed to package it; this executor
# would have salvaged that.

def resolve_relation_follow_slot(
    plugin: "KnowledgeQAPlugin",
    slot_name: str,
    slot_spec: dict,
    consumed_slots: dict,
) -> dict:
    """Find slot value from accepted graph triples using the slot's
    relation pattern. Returns a dict shaped like resolve_entity_constraint_slot
    so the chain executor can treat both uniformly.
    """
    # Which consumed slot supplies the subject?
    subject_slot_name = (
        slot_spec.get("subject_slot")
        or (slot_spec.get("consumes") or ["?"])[0]
    )
    subject_value = str(consumed_slots.get(subject_slot_name, "") or "")
    if not subject_value:
        return {
            "slot_name": slot_name,
            "value": "",
            "method": "relation_follow",
            "supporting_triple_ids": [],
            "cited_obligation_triples": {},
            "candidate_table": [],
            "selection_reason": (
                f"consumed slot {subject_slot_name!r} has no value; cannot "
                f"follow the relation"
            ),
            "ambiguous": False,
            "satisfying_count": 0,
        }

    allowed_preds = [
        p.lower() for p in (slot_spec.get("allowed_predicates") or [])
    ]
    disallowed_preds = [
        p.lower() for p in (slot_spec.get("disallowed_predicates") or [])
    ]
    disallowed_objs = [
        o.lower() for o in (slot_spec.get("disallowed_objects") or [])
    ]
    # Optional subject aliases let the executor match triples whose subject
    # isn't the consumed value verbatim but a related entity (e.g. the
    # founder slot consumes "Rhodes College" but the relevant triple's
    # subject is "mock trial program at Rhodes College").
    extra_subj_aliases = list(slot_spec.get("subject_aliases") or [])
    # value_position: which side of the triple yields the slot value.
    #   "object" (default): triple.subject overlaps consumed → value = triple.object
    #   "subject":           triple.object  overlaps consumed → value = triple.subject
    #   "either":            try both; useful when the relation can be expressed
    #                        as (program, founded by, founder) OR (founder, founded, program)
    value_position = slot_spec.get("value_position", "object")

    def _matches(text: str) -> bool:
        """True iff `text` overlaps the consumed value or any subject alias."""
        if _has_overlap(subject_value, text):
            return True
        for alias in extra_subj_aliases:
            if _has_overlap(alias, text):
                return True
        return False

    # Find candidate triples already accepted in the graph.
    # `matching` carries (triple, slot_value_str) — slot_value depends on
    # which side matched the consumed slot under value_position.
    matching: list = []
    for t in plugin.triples:
        consumed_side_matches = None  # "subject" / "object" / None
        if value_position in ("object", "either") and _matches(t.subject):
            consumed_side_matches = "subject"
        elif value_position in ("subject", "either") and _matches(str(t.object)):
            consumed_side_matches = "object"
        if consumed_side_matches is None:
            continue
        p_l = t.predicate.lower()
        # Predicate must match an allowed pattern if any are specified.
        if allowed_preds and not any(
            ap in p_l or p_l in ap for ap in allowed_preds
        ):
            continue
        # Reject disallowed predicates / objects (check on the SLOT-VALUE
        # side — for value_position=subject the slot value is t.subject,
        # so disallowed_objects should be checked against t.subject too).
        if any(dp in p_l for dp in disallowed_preds):
            continue
        slot_side_str = (
            str(t.object) if consumed_side_matches == "subject"
            else t.subject
        )
        slot_side_l = slot_side_str.lower()
        if any(do in slot_side_l for do in disallowed_objs):
            continue
        matching.append((t, slot_side_str))

    # Uniqueness invariant (mirrors entity_constraint_resolution).
    uniqueness = slot_spec.get("uniqueness", "exactly_one")

    if len(matching) == 0:
        return {
            "slot_name": slot_name,
            "value": "",
            "method": "relation_follow",
            "supporting_triple_ids": [],
            "cited_obligation_triples": {},
            "candidate_table": [],
            "selection_reason": (
                f"no accepted graph triple links subject={subject_value!r} "
                f"via allowed predicates {allowed_preds or '(any)'}"
            ),
            "ambiguous": False,
            "satisfying_count": 0,
        }

    # Dedupe by NORMALIZED slot value. The normalization is slot-spec-driven:
    # if value_canonicalization == 'person', honorifics are stripped before
    # the comparison. This collapses "Marcus Pohlmann" and "Professor Marcus
    # Pohlmann" into one canonical bucket while preserving original surface
    # forms in the proof record.
    canonicalization = slot_spec.get("value_canonicalization", "")
    if canonicalization == "person":
        normalize_fn = normalize_person_value
    else:
        normalize_fn = lambda s: s.strip()  # noqa: E731

    distinct_values: dict[str, list] = {}
    for t, slot_val in matching:
        key = normalize_fn(slot_val).lower().strip()
        distinct_values.setdefault(key, []).append((t, slot_val))

    if len(distinct_values) > 1 and uniqueness == "exactly_one":
        return {
            "slot_name": slot_name,
            "value": "",
            "method": "relation_follow",
            "supporting_triple_ids": [t.id for t, _ in matching],
            "cited_obligation_triples": {},
            "candidate_table": [t.to_dict() for t, _ in matching],
            "selection_reason": (
                f"AMBIGUOUS: {len(distinct_values)} distinct (normalized) "
                f"slot values satisfy the relation pattern: "
                f"{sorted(distinct_values.keys())}"
            ),
            "ambiguous": True,
            "satisfying_count": len(distinct_values),
        }

    # Pick the first triple for the single satisfying slot value.
    chosen, chosen_slot_value = matching[0]
    # Preserve original surface; the slot value is what was on the
    # non-consumed side of the triple under value_position.
    return {
        "slot_name": slot_name,
        "value": chosen_slot_value,
        "method": "relation_follow",
        "cited_obligation_triples": {"0": chosen.id},
        "supporting_triple_ids": [chosen.id],
        "candidate_table": [chosen.to_dict()],
        "selection_reason": (
            f"single graph triple satisfies: "
            f"({chosen.subject!r}, {chosen.predicate!r}, {chosen.object!r}) "
            f"-> slot value = {chosen_slot_value!r}"
        ),
        "ambiguous": False,
        "satisfying_count": 1,
        "selected_triple": chosen.to_dict(),
        "surface_values": [v for _, v in matching],
        "canonicalization": canonicalization,
    }


# -----------------------------------------------------------------------------
# ChainCompletenessVerifier — proof-tree-level chain check
# -----------------------------------------------------------------------------

@dataclass
class ChainCompletenessVerifier:
    """Verifies a typed multi-hop chain end-to-end at the proof-tree level.

    Requires contract.inputs.kind == 'multi_hop_chain' with `slots`, `edges`,
    and `final_answer_slot`. The verifier checks:

      1. Every declared slot is populated in result.output.slot_values.
      2. Every edge has a supporting triple in the plugin graph: a triple
         whose subject/object overlaps the `from`/`to` slot values and whose
         predicate overlaps the edge's `relation`.
      3. result.output.answer matches slot_values[final_answer_slot].
      4. All cited supporting_triple_ids exist in the graph.

    This catches the verifier-depth gap exposed in the prior runs: an answer
    that doesn't actually follow from a connected path through the graph,
    even if individual cited triples are well-grounded.
    """
    plugin: KnowledgeQAPlugin

    def check(self, contract: TaskContract, result: TaskResult, workspace: dict) -> Verdict:
        chain = contract.inputs or {}
        if chain.get("kind") != "multi_hop_chain":
            return Escalate(
                reason="chain_completeness verifier requires contract.inputs.kind='multi_hop_chain'"
            )

        slots_def = chain.get("slots", {}) or {}
        edges = chain.get("edges", []) or []
        final_slot = chain.get("final_answer_slot", "")

        output = result.output if isinstance(result.output, dict) else {}
        if not output:
            return RejectWithRepairHint(
                reason="output must be an object with answer + supporting_triple_ids + slot_values",
                hint=(
                    "Call finish(output={answer: ..., supporting_triple_ids: [...], "
                    "slot_values: {slot_name: value, ...}})."
                ),
                missing_requirements=["object output with chain fields"],
            )

        slot_values = output.get("slot_values") or {}
        answer = str(output.get("answer", "") or "")
        triple_ids = output.get("supporting_triple_ids", []) or []

        # 1. Every slot populated.
        missing_slots = [s for s in slots_def if not slot_values.get(s)]
        if missing_slots:
            return RejectWithRepairHint(
                reason=f"missing slot values: {missing_slots}",
                hint=(
                    f"Populate slot_values for: {missing_slots}. "
                    f"The chain is incomplete without every slot filled."
                ),
                missing_requirements=[f"slot {s}" for s in missing_slots],
            )

        # 2. Every edge supported by a triple.
        edge_evidence = []
        all_triples = self.plugin.triples
        for edge in edges:
            from_slot = edge.get("from", "")
            to_slot = edge.get("to", "")
            from_v = str(slot_values.get(from_slot, "") or "")
            to_v = str(slot_values.get(to_slot, "") or "")
            rel = str(edge.get("relation", "") or "").lower()

            match = None
            for t in all_triples:
                s, o, p = t.subject, str(t.object), t.predicate
                # Lenient relation check: either rel is empty (any predicate),
                # or predicate substring-overlaps the relation.
                rel_ok = (not rel) or (rel in p.lower()) or (p.lower() in rel) or (
                    _has_overlap(rel, p)
                )
                if not rel_ok:
                    continue
                # Triple should link from_v and to_v in some direction.
                if ((_has_overlap(from_v, s) and _has_overlap(to_v, o)) or
                        (_has_overlap(from_v, o) and _has_overlap(to_v, s))):
                    match = t
                    break

            if match is None:
                return RejectWithRepairHint(
                    reason=(
                        f"chain edge {from_slot}->{to_slot} via {rel!r} "
                        f"lacks supporting triple"
                    ),
                    hint=(
                        f"Add a triple linking {from_v!r} to {to_v!r} via a "
                        f"predicate matching {rel!r}. The graph must contain "
                        f"evidence for EACH edge — chain_completeness checks "
                        f"path connectivity, not just individual claims."
                    ),
                    missing_requirements=[f"edge {from_slot}->{to_slot}"],
                )
            edge_evidence.append({
                "edge": edge,
                "triple_id": match.id,
                "from_value": from_v,
                "to_value": to_v,
            })

        # 3. answer == slot_values[final_answer_slot].
        if final_slot:
            final_v = str(slot_values.get(final_slot, "") or "")
            if not final_v:
                return RejectWithRepairHint(
                    reason=f"final_answer_slot {final_slot!r} not populated",
                    hint=f"Set slot_values[{final_slot!r}] before answering.",
                    missing_requirements=[f"slot {final_slot}"],
                )
            if not (_has_overlap(answer, final_v) or _has_overlap(final_v, answer)):
                return RejectWithRepairHint(
                    reason=(
                        f"answer {answer!r} does not match "
                        f"slot_values[{final_slot!r}]={final_v!r}"
                    ),
                    hint=(
                        f"The final answer must equal the value of the "
                        f"final_answer_slot ({final_slot!r}={final_v!r})."
                    ),
                    missing_requirements=["answer matches final_answer_slot"],
                )

        # 4. All cited triples exist.
        known_ids = {t.id for t in all_triples}
        unknown_cited = [tid for tid in triple_ids if tid not in known_ids]
        if unknown_cited:
            return RejectWithRepairHint(
                reason=f"unknown cited triple ids: {unknown_cited}",
                hint=f"These triple IDs don't exist: {unknown_cited}.",
                missing_requirements=[f"valid triple id {tid}" for tid in unknown_cited],
            )

        return Accept(
            reason=(
                f"chain verified: {len(edges)} edges supported by triples; "
                f"final answer = slot_values[{final_slot!r}]"
            ),
            record={
                "verifier": "chain_completeness",
                "answer": answer,
                "slot_values": slot_values,
                "edge_evidence": edge_evidence,
                "cited_triple_ids": triple_ids,
                "graph_size": len(all_triples),
                "sources_registered": sorted(self.plugin.sources.keys()),
            },
        )


# -----------------------------------------------------------------------------
# SlotCertificateVerifier — slot validation, kept separate from fact validation
# -----------------------------------------------------------------------------

@dataclass
class SlotCertificateVerifier:
    """Verifies that a proposed slot value is certified by triples matching
    the slot's proof obligations.

    Crucial design point: keep fact validation and slot validation separate.
    A triple like (Kavanaugh, middle_name, Michael) is a valid fact and
    belongs in the graph — but it does NOT certify Kavanaugh as the answer
    to a slot whose obligation is "father's first name". This verifier
    enforces the slot side of that distinction.

    Reads from contract.inputs:
      slot_name:  the name of the slot being resolved
      slot_spec:  {description, proof_obligations, disallowed_predicates,
                   preferred_method}
      consumed_slots: values of upstream slots this one depends on

    Reads from result.output:
      value:                the proposed slot value
      certificate:          {method, cited_obligation_triples, candidate_table?}
      supporting_triple_ids: all triples cited overall

    For each obligation i, the child cites a triple t-NNNN in
    certificate.cited_obligation_triples[str(i)]. The verifier checks that:
      - the triple exists in the graph
      - its predicate is NOT in disallowed_predicates
      - its predicate matches obligation.predicate_contains
      - its subject overlaps the slot value
      - if obligation.object_contains is set, the triple's object overlaps it
    """
    plugin: KnowledgeQAPlugin

    def check(self, contract: TaskContract, result: TaskResult, workspace: dict) -> Verdict:
        if (contract.inputs or {}).get("kind") != "slot_research_task":
            return Escalate(
                reason="slot_certificate verifier requires inputs.kind='slot_research_task'"
            )

        slot_name = (contract.inputs or {}).get("slot_name", "")
        spec = (contract.inputs or {}).get("slot_spec", {}) or {}
        obligations = spec.get("proof_obligations", []) or []
        disallowed = [p.lower() for p in (spec.get("disallowed_predicates", []) or [])]
        preferred_method = (spec.get("preferred_method", "") or "").strip()

        output = result.output if isinstance(result.output, dict) else {}
        if not output:
            return RejectWithRepairHint(
                reason="output must be an object with value + certificate",
                hint=(
                    "Call finish(output={'value': '<entity>', 'certificate': "
                    "{...}, 'supporting_triple_ids': [...]})."
                ),
                missing_requirements=["object output with slot certificate"],
            )

        value = str(output.get("value", "") or "")
        if not value:
            return RejectWithRepairHint(
                reason="no `value` in output",
                hint="Set output.value to the selected entity (e.g. the justice's name).",
                missing_requirements=["non-empty value"],
            )

        certificate = output.get("certificate") or {}
        if not isinstance(certificate, dict):
            return RejectWithRepairHint(
                reason="`certificate` must be an object",
                hint="Set output.certificate = {method, cited_obligation_triples, ...}",
                missing_requirements=["object certificate"],
            )

        method = str(certificate.get("method", "") or "")

        # If the slot's preferred_method is enumerate_filter, require a
        # candidate_table OR a search_ledger documenting why enumeration
        # was infeasible. This is the user's "research ledger" requirement
        # in place of a give-up-threshold proxy.
        if preferred_method == "enumerate_filter":
            ct = certificate.get("candidate_table")
            sl = certificate.get("search_ledger")
            ok = (isinstance(ct, list) and ct) or (isinstance(sl, list) and sl)
            if not ok:
                return RejectWithRepairHint(
                    reason=(
                        f"slot {slot_name!r} prefers method='enumerate_filter' "
                        f"but neither candidate_table nor search_ledger was provided"
                    ),
                    hint=(
                        "Either: list candidates in certificate.candidate_table "
                        "with {candidate, satisfies, triple_ids} per row; "
                        "OR explain why the candidate set is not enumerable in "
                        "certificate.search_ledger."
                    ),
                    missing_requirements=["candidate_table or search_ledger"],
                )

        # Map obligation index -> required triple id.
        cited_map = certificate.get("cited_obligation_triples") or {}
        if not isinstance(cited_map, dict):
            return RejectWithRepairHint(
                reason="cited_obligation_triples must be {obligation_index: triple_id}",
                hint="Cite one triple ID per proof obligation index (as string keys).",
                missing_requirements=["cited_obligation_triples"],
            )

        known = {t.id: t for t in self.plugin.triples}

        # Check each obligation.
        evidence: list[dict] = []
        for i, ob in enumerate(obligations):
            tid = cited_map.get(str(i)) or cited_map.get(i)
            if not tid:
                return RejectWithRepairHint(
                    reason=(
                        f"obligation [{i}] ({ob.get('description') or ob}) "
                        f"has no cited triple"
                    ),
                    hint=(
                        f"Add cited_obligation_triples['{i}'] = '<t-NNNN>' for "
                        f"a triple whose subject overlaps {value!r} and matches "
                        f"obligation {i} ({ob})."
                    ),
                    missing_requirements=[f"cited triple for obligation {i}"],
                )
            if tid not in known:
                return RejectWithRepairHint(
                    reason=f"obligation [{i}] cites unknown triple {tid!r}",
                    hint=f"Triple {tid!r} doesn't exist. Use only ids returned by add_triple.",
                    missing_requirements=[f"valid triple id {tid}"],
                )
            t = known[tid]

            # Predicate must not be in disallowed list. Semantics: the
            # disallowed PHRASE must appear in the predicate, not vice
            # versa. "law school attended" disallowed should NOT reject a
            # predicate of just "attended" — the bare predicate doesn't
            # imply law school. Also check the OBJECT for disallowed
            # markers (so a predicate of "attended" with object "Notre
            # Dame Law School" is still caught for the college slot).
            p_l = t.predicate.lower()
            o_l = str(t.object).lower()
            for d in disallowed:
                if d in p_l or d in o_l:
                    return RejectWithRepairHint(
                        reason=(
                            f"obligation [{i}] cites triple {tid!r} which is "
                            f"DISALLOWED for slot {slot_name!r}: disallowed marker "
                            f"{d!r} appears in "
                            f"{'predicate' if d in p_l else 'object'} of "
                            f"({t.subject!r}, {t.predicate!r}, {t.object!r}). "
                            f"Disallowed list: {disallowed}"
                        ),
                        hint=(
                            f"The triple ({t.subject!r}, {t.predicate!r}, "
                            f"{t.object!r}) may be a valid FACT but does NOT "
                            f"certify slot {slot_name!r}. Find a different "
                            f"triple — e.g. for the college slot, use the "
                            f"justice's undergraduate institution, not their "
                            f"law school."
                        ),
                        missing_requirements=[
                            f"non-disallowed predicate/object for obligation {i}"
                        ],
                    )

            # Subject must overlap the slot value.
            if not _has_overlap(value, t.subject):
                return RejectWithRepairHint(
                    reason=(
                        f"obligation [{i}] cites triple {tid!r} whose subject "
                        f"{t.subject!r} does not overlap value {value!r}"
                    ),
                    hint=(
                        f"The certifying triple's subject must reference the "
                        f"slot value. Find a triple where the subject is "
                        f"(or overlaps) {value!r}."
                    ),
                    missing_requirements=[f"triple about {value!r} for obligation {i}"],
                )

            # Predicate must match the obligation pattern.
            pred_pat = (ob.get("predicate_contains", "") or "").lower()
            if pred_pat and pred_pat not in p_l and p_l not in pred_pat and not _has_overlap(pred_pat, p_l):
                return RejectWithRepairHint(
                    reason=(
                        f"obligation [{i}] requires predicate ~ {pred_pat!r} "
                        f"but cited triple has predicate {t.predicate!r}"
                    ),
                    hint=(
                        f"Cite a triple whose predicate matches {pred_pat!r}. "
                        f"The obligation is about a specific relation."
                    ),
                    missing_requirements=[f"predicate matching {pred_pat!r}"],
                )

            # Object must overlap the obligation's object_contains if specified.
            obj_pat = ob.get("object_contains", "") or ""
            if obj_pat and not _has_overlap(obj_pat, str(t.object)):
                return RejectWithRepairHint(
                    reason=(
                        f"obligation [{i}] requires object ~ {obj_pat!r} but "
                        f"cited triple has object {t.object!r}"
                    ),
                    hint=(
                        f"Cite a triple whose object overlaps {obj_pat!r}, "
                        f"not {t.object!r}."
                    ),
                    missing_requirements=[f"object matching {obj_pat!r}"],
                )

            evidence.append({
                "obligation_index": i,
                "obligation": ob,
                "triple_id": tid,
                "triple": t.to_dict(),
            })

        record = {
            "verifier": "slot_certificate",
            "slot_name": slot_name,
            "value": value,
            "method": method,
            "obligations_satisfied": evidence,
            "candidate_table": certificate.get("candidate_table"),
            "search_ledger": certificate.get("search_ledger"),
        }
        return Accept(
            reason=(
                f"slot {slot_name!r} = {value!r} certified by "
                f"{len(evidence)} obligation(s)"
            ),
            record=record,
        )
