# Report rules

These are the rules for writing the report in `report/`. They combine this
project's own rules (from `CLAUDE.md` and earlier corrections) with the
`plain-docs` writing rules. Follow all of them when you draft or edit any
`.tex` file under `report/`.

## Project rules

### Before writing

- Ask the user what the text should refer to: which sources, experiments
  or sections it draws on. Do not choose them yourself.
- Ask all the open questions at once: sources, non-standard terms, and
  anything you are unsure belongs in the report.
- Check `TRACKER.md` goal 7 for the current state of the report.
- Write the abstract last. `report.tex` leaves it empty until then.

### Sources and citations

- Cite only sources you have verified. Add each one to
  `report/references.bib` and cite it with `\citep{}`.
- Verify every claim against the source, the documentation or a run before
  you state it.
- Everything under `experiments/` (`VMMVector/`, `VMMRemapShared/`,
  `VMMSlab/`, `pytorch-vmm-study/` and the rest) is scratch work. Some of
  it will be cleaned up and some discarded. Never cite it as established
  fact, and never treat a verdict or results file in this repo as an
  external source.
- Attribute this project's results as preliminary observations, for
  example "in our experiments we observed ..." and not "X causes Y" stated
  flat. Facts marked **[Observed]** in `CLAUDE.md` are working assumptions
  and get the same treatment.
- Report results as measured, including negative results.
- Never mention or link `notes/artifacts/` in the report. State the fact
  itself and cite its real source.

### Style

- Write formal, simple academic prose. The punchy register of the repo's
  own notes and READMEs does not belong in the report.
- Apply every `plain-docs` rule below.

### Build and files

- Build with `cd report && make`. The built `report/report.pdf` is
  committed, and it is the only tracked PDF. Do not commit cited papers or
  other PDFs.
- Commit only when the user asks.

## Plain-docs core rules

1. State things directly. Say what something is, without first saying what
   it is not.
2. Hedge once, on purpose, in one place. Gather real caveats into a short
   section near the end. Do not remove real uncertainty, because false
   confidence is worse than a hedge.
3. Start with the real information and end on the last real point. Skip
   filler openers, closers and recap paragraphs.
4. Prefer concrete specifics such as a number, a command, a filename or a
   version. If a claim cannot be made concrete, cut it.
5. Use plain ASCII with no emoji. In LaTeX source this means no Unicode
   dashes, arrows or curly quotes; use LaTeX markup (`--`, `$\to$`,
   ``` ``...'' ```) instead.
6. Vary sentence rhythm, not vocabulary. Repeating a term is clearer than
   cycling synonyms.
7. Write full, natural sentences with a subject and a verb. Use "we" for
   design choices and the component's name (or "it") for behavior. Lead
   into lists with a sentence, and say why when a choice is not obvious.
   Tables and code can stay terse.
8. Match the format asked for. Here that is the TMLR-style LaTeX report.
9. Write it ready to ship, with no further editing pass assumed.

## Report patterns to follow

- **Reason from first principles.** Give the observation first, then the
  explanation, then the evidence for the explanation. For example: "We saw
  a 12% drop in latency. We think the smaller batch size causes this. To
  check, we reran the test with the old batch size, and the drop went
  away."
- **Write short, simple sentences.** Put one idea in each sentence and join
  at most two clauses. Prefer plain verbs such as "use", "show", "find",
  "cut" and "rise", with few adverbs and adjectives.
- **Use an explanatory, polite tone.** Write in the active voice with "we".
  Explain what you did and found, as you would to a colleague. Avoid a
  commanding or impersonal tone and the passive voice.
- **Keep the original names.** Call a metric, quantity or identifier by the
  same name everywhere, matching the figures, tables and code. Prefer the
  simplest name the user accepts.
- **Build ideas in a few steps.** Use one section for a simple idea and up
  to three for a complex one, each building on the last. Use bullet points
  for steps and findings, one or two simple sentences each with at most one
  number.
- **Describe graphs with a few numbers.** Describe the trend in words, then
  give one to three anchoring numbers such as the start, the end or a
  turning point.
- **Match claims to evidence.** Say what the data shows for the setup you
  tested, and name the limits of that setup instead of generalizing past
  it.
- **Handle special terms with the user.** For each project-specific name,
  coined metric or internal acronym, try a plain term first. If the term
  has to stay, ask the user whether to define it and where (the setup
  section or a short glossary). Do not add definitions on your own.

## Banned patterns

### Never use

1. Non-ASCII punctuation and symbols: em and en dashes, arrows, curly
   quotes, the single-character ellipsis. Most em dashes read better as a
   period or comma.
2. Emoji, anywhere.
3. Negative parallelism: "not X, it's Y", "not just X but Y", "Y rather
   than X", "no X, no Y, just Z". State the true thing directly.
4. Inflated vocabulary: delve, leverage, robust, seamless, utilize,
   showcase, boast, intricate, meticulous, pivotal, crucial, landscape
   (abstract), tapestry, testament, foster, garner, multifaceted, vibrant,
   holistic, elevate, empower, cutting-edge, underscore (as a verb).
5. Inflated significance and promotional tone: "stands as a testament to",
   "plays a pivotal role", "marks a shift", "the evolving landscape",
   "groundbreaking", "renowned", "in the heart of".
6. Filler openers and closers: "It's worth noting that", "Essentially",
   "Simply put", "In conclusion", "Overall", recap paragraphs and upbeat
   endings.
7. Chatbot and tool artifacts: "I hope this helps", "Here's a ...",
   unfilled placeholders, `oaicite`, `contentReference`, tracking
   parameters in URLs.

### Use rarely

8. Rule of three. Keep a three-item list only when every item carries real
   information.
9. Scattered hedging ("might", "could potentially", "generally", "in some
   cases"). Move real caveats to one section near the end.
10. Trailing "-ing" padding such as "..., ensuring reliability". Make it
    concrete or cut it.
11. "Serves as", "functions as", "stands as", "boasts" or "refers to" in
    place of a plain "is" or "has".
12. Vague claims and attribution ("widely considered", "experts
    recommend", "associated with"). Name the source or the actual
    relationship, or cut the claim.
13. Synonym cycling. Use one term for one thing.
14. Bold in running prose, and lists where every item starts with a bold
    label and a colon.
15. Title Case headings. Use sentence case.
16. Telegraphic prose: fragments, "Label: fact" lines, colon chains, and
    dropped subjects or articles.

### Use moderately

17. Sentence-initial "Additionally", "Furthermore" and "Moreover". Most can
    be deleted.
18. Same-rhythm sentence chains joined by "and", "which" or semicolons. Mix
    short and long sentences.
19. Structure the content does not need, such as headings on very short
    sections or deep nesting.

### Report-specific

- R1. Overclaiming: "proves", "always", "solves", "significantly" without a
  significance test, or results stretched beyond the tested setup.
- R2. Packed sentences that carry several numbers, causes and comparisons
  at once. Split them, one idea per sentence.
- R3. Assertive, impersonal or passive tone, such as "A sweep was done."
  Rewrite with "we" and a plain verb: "We ran a sweep."
- R4. Undefined jargon. Replace it with plain terms or ask the user about a
  definition.
- R5. Number dumps in graph descriptions or single lines.
- R6. Background leakage: exploration or side experiments that are not
  part of the report's results. Leave them out unless the user says they
  belong.
- R7. Renamed quantities: the same metric under different names in the
  figures and the text.
- R8. Complex verbs and modifiers: "necessitates", "exhibits",
  "demonstrates", "facilitates", "substantially", "markedly", "notably".

## Pre-ship checklist

Reread the draft once per item. One pass is not enough, because padding and
hedging are the default habit.

- [ ] Did you ask the user about sources, non-standard terms and unclear
      content before drafting?
- [ ] Is every source verified, in `references.bib` and cited with
      `\citep{}`?
- [ ] Is every result from `experiments/` attributed as this project's
      preliminary observation?
- [ ] Is there no mention of `notes/artifacts/`?
- [ ] Does `grep -nP '[^\x00-\x7F]' FILE` find nothing?
- [ ] Is there no negative parallelism, inflated vocabulary, promotional
      phrasing, filler opener or closer, or chatbot debris?
- [ ] Is hedging gathered in one place?
- [ ] Is each thing called by one name, matching the figures and tables?
- [ ] Does each sentence hold one idea, with at most two clauses?
- [ ] Is the tone "we did, we found", with no passive or impersonal voice?
- [ ] Do graph descriptions use only one to three numbers?
- [ ] Does every claim match the evidence and the tested setup?
- [ ] Is anything included from outside the report's sources?
- [ ] Are headings in sentence case, with bold used for at most one or two
      key points?
- [ ] Do the first and last sentences of each section carry real
      information?
- [ ] Does `cd report && make` build without errors?
