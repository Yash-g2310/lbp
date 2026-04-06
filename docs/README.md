# Documentation Hub

This folder is the canonical, modular documentation entry point for the LBP project.

## Navigation

- Start here for first run and daily workflows:
  - `guides/quickstart.md`
  - `guides/quickcheck_guide.md`
  - `guides/slurm_operations.md`
- Use these as stable technical references:
  - `reference/project_layout.md`
  - `reference/data_configuration.md`
  - `reference/reporting_and_artifacts.md`
- Generated reports are isolated from curated docs:
  - `generated/README.md`

## Context For Future Chats

Project context handoff files live outside docs in:

- `../context/README.md`

The `context/` tree is intentionally modular and curated. Raw chat transcripts are not copied into tracked project docs.

## Authoring Rules

1. Keep each file single-purpose.
2. Prefer linking to canonical references over duplicating command blocks.
3. Put machine-generated output in `docs/generated/`.
4. Keep docs in repo-relative paths to avoid machine-specific leakage.
