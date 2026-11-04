---
id: TASK-43
title: Consolidate mldag tools under one discoverable CLI
status: To Do
assignee: []
created_date: '2026-11-04 20:16'
updated_date: '2026-11-04 20:17'
labels:
  - cli
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
mldag's user-facing tools are hard to find and remember. There are five separate mldag-* commands (gen, csv, report, dashboard, monitor) built with two different CLI frameworks (argparse and typer), so help output, argument styles and defaults differ from tool to tool. The DAG monitor, one of the most-used tools, has no command at all and is only reachable as 'python -m mldag.monitor.dagman <dag> --tail'. Meanwhile the similarly named mldag-monitor is a different tool that queries the schedd, which makes the naming actively misleading. Nothing lists what tools exist, so the only way to find them is reading pyproject.toml or the source.

The CLI surface has also accumulated commands, subcommands and flags that aren't used or wanted any more (legacy positional arguments, experiment-specific special cases, debug flags, one-off analysis scripts). Consolidating is the moment to cut those, so the new CLI only carries what is actually used.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Running 'mldag --help' lists every user-facing tool with a one-line description of each
- [ ] #2 The DAG-log monitor is runnable without python -m, with whichever of its options the audit keeps
- [ ] #3 Tool names make clear which one follows a DAG's logs and which one queries the HTCondor schedd
- [ ] #4 Help output, positional-vs-option inputs, and output-directory flag naming follow one consistent convention across all tools
- [ ] #5 The old mldag-* entry points are removed rather than aliased, and the justfile and README use the new invocations
- [ ] #6 An automated test confirms every tool's --help runs successfully
- [ ] #7 Every existing command, subcommand, flag and standalone script entry point is audited with the user and recorded as keep or remove; everything marked remove is deleted along with code that only it used
<!-- AC:END -->
