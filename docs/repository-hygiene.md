# Repository Hygiene

This repository is intentionally kept as an academic portfolio rather than a raw submission dump. The public copy keeps source code, notebooks, project READMEs, and small support files; generated reports, submission archives, large datasets, caches, and trained model checkpoints are omitted.

## Current Scale

- The working tree has been pruned to source-first material.
- The Git history may still contain older large artifacts until history is rewritten or the project is republished as a fresh public mirror.
- Large local files should be regenerated, downloaded, or stored externally instead of committed.

## GitHub-facing Decisions

- The semester folders were left in place to preserve academic traceability.
- The root README now acts as the public portfolio entry point.
- `docs/showcase.md` provides a curated route through the strongest projects.
- `docs/course-catalog.md` provides the complete semester and course index.
- `.gitignore` prevents future caches, local environments, exported reports, model dumps, compressed archives, raw datasets, and compiled binaries from being added accidentally.
- `.gitattributes` marks common binary, generated, vendored, and data-heavy paths so diffs and GitHub language statistics are cleaner.

## Large Artifacts

Several large files may still exist in Git history. The new ignore rules prevent similar files from being added again, but they do not rewrite past commits.

For a lean public mirror, consider this future cleanup path:

1. Move raw datasets, generated training archives, and model checkpoints to Git LFS, an external release, or documented download links.
2. Remove build artifacts, caches, compiled binaries, and local IDE settings from history.
3. Keep notebooks, source code, READMEs, and small illustrative support files in Git.
4. Re-run a privacy scan before making the repository widely public.

## Publishing Recommendation

Do not publish the old academic history as-is. Earlier commits may still contain removed reports, archives, datasets, model checkpoints, local paths, student identifiers, or collaborator metadata.

The safest public release path is to create a fresh Git history from the cleaned working tree:

1. Keep the original repository as a private local/archive copy.
2. Create a new public mirror from the sanitized current tree.
3. Initialize a fresh Git repository for that mirror.
4. Make a first public commit such as `Initial public portfolio release`.
5. Push that clean history to GitHub.

If an older version has already been pushed publicly, make that repository private or delete it and publish a new sanitized repository. Rewriting and force-pushing history is useful, but it cannot guarantee removal from forks, local clones, or external caches.

## Privacy And Third-party Material

The cleanup removes known student IDs, collaborator names, course staff contacts, local absolute paths, and exported assignment metadata from the public working tree.

Before sharing the repository broadly, re-run the privacy scan and review high-visibility notebooks for accidental reintroduced metadata.

## Reproducibility

The assignments were completed across multiple courses and environments. There is no single repository-wide environment file because dependencies differ by course and semester. Treat each assignment folder as its own context and check local notebooks, scripts, or course instructions for setup details.
