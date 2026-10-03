# References — how the citation system works

**Rule: no borrowed knowledge without a visible reference.** Every technique the
agents recommend must trace back to a cited card in the knowledge library.

## The chain of custody

1. **Knowledge cards** (`src/knowledge/*.md`) — each card carries labeled sections:
   `ID`, `TITLE`, `WHEN_TO_USE`, `STEPS`, `SOURCE`, `LICENSE`, `ACCESSED`.
2. **RAG Researcher agent** — must cite the `CARD ID` of every technique it uses.
   Output without card IDs is rejected by the task contract.
3. **Treatment Planner** — every technique in the plan carries its card ID and is
   listed in the plan's `REFERENCES` section with its source line.
4. **UI Report tab** — shows the full reference list (ID, title, source, license,
   URL, accessed date).
5. **DOCX report** — same references section, generated from the same structured
   data (not re-typed by an LLM), so the file matches the screen exactly.

## Source-license rules (what we may and may not store)

- **Original summaries of standard techniques: YES.** Ideas and methods are not
  copyrightable — only an author's exact wording is. All 12 cards are written in
  our own words; no source text is reproduced. Each card still cites the
  origin (book title + author + chapter) as a courtesy and for authority.
- **Openly licensed sources: YES, with attribution.**
  - OpenStax Psychology 2e — CC BY 4.0 (attribute + link).
  - NIMH topic pages — U.S. public domain for text (their images may NOT be
    reused; cite NIMH as the source).
  - WHO mental-health publications — only those issued under CC BY-NC-SA 3.0 IGO
    (WHO publications from 1 January 2017 onward), and only for non-commercial
    use. Older or non-CC items need WHO permission — check each publication's
    copyright notice before storing it.
- **Copyrighted books: cite, don't copy.** Book title + author + chapter as a
  reference line. Never paste passages.
- **Jurisdiction note:** the ideas-vs-expression principle above follows U.S.
  copyright law (17 U.S.C. §102(b)); the equivalent local statute has not been
  verified — both follow the Berne Convention framework.

## Adding a new card

1. Write the summary in your own words (plain language, 5–7 steps).
2. Fill `SOURCE` (title, author, chapter/URL) and `LICENSE` honestly.
3. Set `ACCESSED` to today's date for web sources.
4. Drop the `.md` file in `src/knowledge/` and rebuild the index
   (`build_index()` in `src/rag.py`, or delete `data/chroma` to force rebuild).
