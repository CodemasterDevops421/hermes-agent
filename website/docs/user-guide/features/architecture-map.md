---
sidebar_position: 1
title: "Architecture Map"
description: "A concise map of Hermes Agent's main pillars and where they live in the codebase"
---

# Architecture Map

Hermes is easier to understand when the main pillars are mapped directly to code.

## Core pillars

| Pillar | What it does | Typical code area |
|---|---|---|
| **Loop** | Runs the model/tool interaction turn by turn | agent / loop / CLI |
| **Memory** | Persists user preferences and project facts across sessions | memory / session store |
| **Tools** | Exposes actions like terminal, web, browser, file, and automation | tools / toolsets |
| **Gateway** | Connects Hermes to Telegram, Discord, Slack, and other surfaces | gateway / platform adapters |
| **Profiles** | Keeps separate agent environments isolated | profile / config / state |
| **Delegation** | Spawns focused subagents for parallel work | delegation / worker orchestration |
| **Cron** | Runs scheduled jobs unattended | cron / scheduler |
| **Skills** | Stores reusable procedures and workflows | skills / skill management |

## Why this matters

This map is meant to make Hermes easier to learn:

- new users can find the right feature faster
- contributors can locate the relevant code path quickly
- the agent stays powerful without feeling opaque

## Suggested reading order

1. [Start Here](start-here.md)
2. [Features Overview](overview.md)
3. [Tools & Toolsets](tools.md)
4. [Persistent Memory](memory.md)
5. [Skills System](skills.md)
6. [Subagent Delegation](delegation.md)
7. [Scheduled Tasks (Cron)](cron.md)

