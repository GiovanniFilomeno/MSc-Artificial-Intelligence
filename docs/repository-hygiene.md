# Repository Hygiene

This repository is an academic portfolio mirror, not a single deployable software package. Its semester structure preserves academic context, while the root README and `docs/` provide the curated entry points.

## Current state

The public repository was initialized from a sanitized snapshot in June 2026. Its fresh public history began with one portfolio commit and does not contain the original private academic commit sequence.

The public working tree has been reduced to student-authored work, concise documentation, selected notebook output, small illustrative resources, clearly identified third-party source, and a limited number of archival model/data artifacts. Some student submissions still embed course scaffolding; [`NOTICE.md`](../NOTICE.md) describes that ownership boundary. Runtime logs, R session state, duplicated archives, compiled SAT binaries, large generated InstanceFlow exports, obvious instructor-only copies, and explicitly restricted notebooks were removed during the portfolio pass.

The root `.gitignore` excludes common local configuration, generated output, build products, archives, and model artifacts from normal future additions. It does not remove files that are already tracked and can still be overridden explicitly. The root `.gitattributes` normalizes text files, marks binary formats, and identifies generated or vendored paths for clearer diffs and more representative GitHub language statistics.

## Portfolio artifact policy

Prefer keeping:

- authored source code and concise project documentation;
- a small number of polished, executed showcase notebooks;
- lightweight test fixtures and illustrative images;
- reproducible environment specifications and build instructions;
- upstream license and attribution files.

Prefer publishing elsewhere, with a documented source and checksum:

- raw or easily downloadable datasets;
- trained models and serialized Python or R objects;
- long training logs, TensorBoard events, and repeated videos;
- archives that duplicate extracted files;
- compiled executables and object files;
- scratch notebooks and large embedded image output.

Notebook output should be deliberate. Keep output that materially demonstrates a result, but clear failed executions, repeated plots, debug traces, and output from exploratory copies. A concise result image in project documentation is often easier for a reviewer to evaluate than a multi-megabyte embedded notebook payload.

## Automated checks

The `Portfolio quality` GitHub Actions workflow uses only the Python standard library and shell tools available on the runner. It:

- runs the repository privacy scan;
- checks high-confidence secret and private-key formats without printing matched values;
- validates inline and reference-style local links in tracked Markdown files;
- parses tracked Python files with the Python AST;
- validates the JSON structure of every tracked notebook.

The workflow does not import project dependencies, execute notebooks, train models, or claim that old coursework remains reproducible on a modern environment.

Every missing local Markdown target fails the link check; there is no repository-wide exception list.

## Rights and provenance

The repository combines individual submissions, group work, course scaffolding, data, and third-party code. A notice is not a substitute for redistribution permission. Keep upstream licenses with vendored software, document group contributions, and remove course-provided prompts, templates, or grading utilities unless public redistribution is permitted.

See [`NOTICE.md`](../NOTICE.md) for the current ownership and reuse boundary.

## Before promoting the portfolio

1. Review external links and group media for consent and continued availability.
2. Move any remaining large or opaque artifact to a release or its authoritative data source when it no longer adds review value.
3. Replace archival build notes with portable build instructions before promoting the SAT project.
4. Re-run all local quality checks and review the GitHub-rendered repository.
