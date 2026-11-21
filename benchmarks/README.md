# Validity cases

Questions with known answers, run through rtool to measure how often it gets them right. The format,
the rules for writing a case and the scoring are in [docs/stage0_design.md](../docs/stage0_design.md).

```
benchmarks/<pack>/<case>/
  case.yaml       the brief, the fixed hypothesis, the data it provides, the expected outcome
  generate.py     optional: writes the case's data (standard library only, deterministic)
  reference.py    optional: a hand-written version that confirms the expected outcome at the case's budget
```

Run them on Barkla, from the code folder (nobody is asked at gates, so the GPU budget is the only limit):

```
rtool validity submit generic          # creates and starts every missing run of this code version
rtool validity score generic           # scores finished runs, writes the card under <runs root>/.validity-cards
```

A case counts in a card's headline only once `confirmed_by` says what confirmed its expected outcome:
a formula, or a `reference.py` run (with date and machine). The others are reported apart.

| pack | case | kind | confirmed by |
|---|---|---|---|
| generic | mm1-time-in-system | replication | formula, and `reference.py` |
| generic | two-servers-shorter-wait | planted | formula (Erlang C), and `reference.py` |
| generic | rng-choice-null | null | construction: both generators draw the same distributions |
| ml_dl | xor-needs-hidden-layer | planted | `reference.py` (standard library) |
| ml_dl | label-noise-hurts | planted | not yet: `reference.py` needs torch and MNIST (Barkla) |
| ml_dl | random-halves-null | null | construction: the halves are exchangeable under the seed |
| ml_dl | softmax-mnist-accuracy | replication | not yet: `reference.py` needs torch and MNIST (Barkla) |
| ml_dl | open-topic | open | nothing to confirm (scored L0–L1 only) |
