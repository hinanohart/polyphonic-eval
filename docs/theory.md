# Theory and naming

This page exists so the rest of the documentation can stay focused on practical
usage. Nothing here is required reading for using the library.

## Where the name comes from

Mikhail Bakhtin's literary-critical concept of *polyphony* describes a text in
which multiple voices coexist without being collapsed into a single
authoritative perspective. The canonical example is Dostoevsky's novels: each
character holds a worldview that the narration treats as fully formed and
non-reducible, rather than as a step on the way to one authorial truth.

For multi-judge LLM evaluation the analogy is loose but useful: if five judges
return three substantively different reasons for their scores, "averaging the
scores" reduces five voices to one. That reduction *might* be the right move
for the downstream task. It might also be a mistake — especially when one of
the minority voices is flagging a safety or factuality concern. ``polyphonic-eval``
makes that mistake hard to commit accidentally: the public type system refuses
the reduction by default.

## What is NOT borrowed from Bakhtin

Most of Bakhtin's apparatus does not transfer. Public function and type names
deliberately avoid:

- **chronotope** (time-space narrative integration)
- **heteroglossia** (the social-linguistic stratification of language)
- **carnival** (ritual inversion of authority)
- **unfinalizability** (the impossibility of closing a character)
- **dialogism**, **addressivity**, **outsideness**, **answerability**

These concepts are interesting and informed the early design discussions, but
each of them either (a) duplicates existing engineering vocabulary or (b)
would require theoretical commitments the library shouldn't impose. They
remain in this document for the curious reader; they are not in the public API.

## What IS borrowed

Exactly one operational principle:

**Two equally-justified, mutually incompatible verdicts must not be summed and
divided by two.**

This is the load-bearing constraint. Everything else — the choice of HDBSCAN,
the bootstrap stability test, the score-spread-tolerance consensus check (with
Krippendorff α planned for v0.2.0) — is implementation detail that could be
swapped without breaking the contract.

## Related literature

If you want to dig further into the surrounding research:

- *Free-MAD* (arXiv 2509.11035) — anti-conformity mechanisms in multi-agent debate.
- *DMAD* (ICLR 2025) — Diverse Multi-Agent Debate, mental-set breaking.
- *X-MAS* (arXiv 2505.16997) — heterogeneous LLM ensembles.
- *Sociolinguistic Foundations of Language Models* (arXiv 2407.09241) —
  background on engineering with linguistic variation.
- *Dialogic Evolution of AI Products* (Tandfonline 2024) — peer-reviewed
  Bakhtin × AI direct treatment.
- *Krippendorff (2018)* — inter-rater agreement; v0.1.0 ships with a simple
  spread-tolerance consensus check (see ``consensus.py``); a Krippendorff α
  implementation is on the v0.2.0 roadmap.

None of these papers are pre-requisites for the library; they're entry points
if the idea-space looks interesting.
