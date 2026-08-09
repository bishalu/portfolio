# /naam — the entry, and the lighthouses that steer it

Direction, not measurements. The rubric changes every pass; this changes rarely
and deliberately. Read this before touching the load sequence.

Companion to `DESIGN.md` (the site) and `MOTION.md` (the motion set). `/naam`
is a sanctioned exception to Alpenglow and has its own tokens; this file is
about its **entry** specifically.

---

## The objective, in the words it was given in

> "address your concerns · ensure the user gets an immersive feel out of this ·
> and give their mind space to breathe"

It is a **constrained** objective and the constraint is the interesting half:

**Maximise immersion, subject to the mind having room.**

Written as a priority rule that becomes "immersion *instead of* room", it
inverts — because the obvious way to raise immersion is to add motion, and the
obvious way to add motion is to fill the rests. The rests are the constraint.
Anything that fills one has failed the objective while appearing to serve it.

Components, each of which must be reported every pass even when unmeasured:

| # | Component | Where it is measured |
|---|---|---|
| O1 | The two clocks stop disagreeing — words and sky are one sequence | E1 |
| O2 | The chips stop answering before the question finishes | E3 |
| O3 | The phone gets an entry of its own, not a plainer one | E6 |
| O4 | Immersion — arriving somewhere, not loading something | E7, E8 |
| O5 | Room to breathe — rests are real and nothing competes inside one | E2, E5 |
| O6 | Acting always beats waiting — the page is usable before it is finished | E4 |

---

## Lighthouses

**L1 · The rests are the design, not the gaps between it.** (weight 5 —
defines the objective.) Immersion here comes from silence with something in
it, not from density. When a beat and a rest compete for the same 400ms, the
rest wins. Any change that fills a rest to add interest has inverted the
objective.

**L2 · One sequence, never two.** (weight 5 — defines the objective.) The
words and the sky are one performance on one stage. Nothing on the right may
start while something on the left is still arriving. Two well-made
choreographies running concurrently read as neither.

**L3 · The sky answers the invitation; it does not accompany it.** (weight 3 —
architecture.) The lanterns are other people's names. They belong *after* the
line that asks for yours, as its reply. This is what makes them meaningful
rather than ambient.

**L4 · Acting always beats waiting.** (weight 5 — objective.) The composer is
live as soon as it can be, and typing cuts the remaining choreography short
(`opening.rush()`). Because the sequence is skippable by acting, it is allowed
to be unhurried. Never make someone wait for a performance to finish before the
page will answer them.

**L5 · The phone gets its own entry, not a subset of the desktop one.**
(weight 3 — architecture.) Most of this family is on a phone. A stacked layout
is a different instrument, not a smaller stage — the third beat there is the
family's names arriving in the thread, and it must be composed, not inherited.

**L6 · A name never appears without the paper it is written on.** (weight 3 —
architecture.) Earned twice: the labels are DOM and the lanterns are canvas, and
they have come apart twice, in opposite directions. Any change to the reveal
path re-checks this.

**L7 · Grade what the visitor sees, in a recording.** (weight 1 — method.)
Three faults here were invisible to every probe and obvious in a filmstrip: a
name flickering in-out-back, a doodle graded F for being deliberately hidden, an
opening judged unstaged because the probe read the wrong element. Numbers rank;
the recording decides.

**L8 · The document is the source; the page never invents a name.** (weight 5 —
root.) Carried from DESIGN.md §4. Nothing in an entry pass may make a name
appear that the 6,715-row document does not contain.

---

## Contradicted / revised

**L4 was contradicted by the code, not by the world — pass 1.** "Acting always
beats waiting" was written as though it were already true. It was not:
`opening.rush()` was wired to submit, under a comment in the source reading
"typing is consent to get on with it", which is what it should have done and
was not what it did. Measured: typing at 700ms left the invitation still
arriving at 3773ms, identical to not touching anything.

The lighthouse stands and the code was changed to meet it. Recorded here
because the failure mode is worth naming: **a lighthouse can be contradicted by
a comment that already agrees with it.** The intent was written down twice, in
two places, and neither was the behaviour.

**L2 caught its own fix twice in one pass.** Moving the sky to answer the ask
put it in the same window as the doodle, which was also waiting on
`opening.done`. Moving the chips to land with the ask put them in the same frame
the sky started rising. Both were the fault L2 exists to prevent, reintroduced
by the fix for it — because everything that waits on the end of the invitation
waits on the same signal. Anything else gated on `opening.done` in future needs
its own beat, not the same one.
