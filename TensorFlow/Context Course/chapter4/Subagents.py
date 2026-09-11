"""
================================================================================
 Context Course · Chapter 4 · 子智能体模式(Subagent Patterns)（学习笔记描述）
================================================================================
 一句话：多智能体工作流就那么几种"形状"——本章讲 4 种常见编排模式。
 本章讲：
   ① 常见多 agent 形态(主管-子工、流水线、并行分治、反馈循环)。
   ② 每种模式适合什么任务、怎么传递上下文。
   ③ 与 实战练习/Agent进阶案例/案例2(深度研搜多智能体)呼应。
 要点：子 agent 隔离上下文/职责，避免单个 agent 上下文爆炸。
 说明：超长讲解(概念为主)，参考为主。
================================================================================
"""

# # # Subagent Patterns
# #
# # Multi - agent
# # workflows
# # tend
# # to
# # fall
# # into
# # a
# # handful
# # of
# # shapes.We
# # will
# # go
# # over
# # 4
# # of
# # them
# # that
# # cover
# # almost
# # anything
# # you
# # would
# # build.
# #
# # < iframe
# # src = "https://context-course-subagent-patterns.static.hf.space"
# # frameborder = "0"
# # width = "850"
# # height = "450"
# # >
# #
# # ## Pattern 1: Fan-Out / Fan-In
# #
# # ** What
# # it is: ** Spawn
# # multiple
# # subagents in parallel
# # for independent subtasks, then aggregate results.
# #
# # ** When
# # to
# # use: ** Independent
# # tasks
# # with no dependencies between them.
# #
# # ** Example: ** Evaluate
# # 5
# # models
# # on
# # the
# # same
# # benchmark.
# #
# # ```
# # Parent: "Compare models"
# # ├─ Subagent
# # 1: Evaluate
# # Model
# # A → Score: 92 %
# # ├─ Subagent
# # 2: Evaluate
# # Model
# # B → Score: 89 %
# # ├─ Subagent
# # 3: Evaluate
# # Model
# # C → Score: 94 %
# # ├─ Subagent
# # 4: Evaluate
# # Model
# # D → Score: 88 %
# # └─ Subagent
# # 5: Evaluate
# # Model
# # E → Score: 91 %
# #
# # Parent: Aggregate → Ranking
# # report
# # ```
# #
# # ** Pseudocode: **
# # ```text
# # parent_agent:
# # subagents = []
# # for model in [A, B, C, D, E]:
# #     subagent = spawn(
# #         task=f"Evaluate {model}",
# #         context={model}
# #     )
# #     subagents.append(subagent)
# #
# # results = wait_all(subagents)
# # report = aggregate(results)
# # return report
# # ```
# #
# # ** Pros: ** Fully
# # parallel, max
# # speed, easy
# # to
# # scale
# #
# # ** Cons: ** Total
# # time = slowest
# # subagent
# #
# # ## Pattern 2: Pipeline
# #
# # ** What
# # it is: ** Chain
# # subagents
# # where
# # output
# # of
# # one
# # becomes
# # input
# # to
# # the
# # next.
# #
# # ** When
# # to
# # use: ** Sequential
# # processing
# # with data flowing through stages.
# #
# # ** Example: ** Data
# # processing: extract → clean → analyze → report
# #
# # ```
# # Raw
# # Data
# # ↓
# # [Subagent 1: Extract]
# # ↓
# # Extracted
# # Data
# # ↓
# # [Subagent 2: Clean]
# # ↓
# # Clean
# # Data
# # ↓
# # [Subagent 3: Analyze]
# # ↓
# # Results
# # ↓
# # [Subagent 4: Report]
# # ↓
# # Final
# # Report
# # ```
# #
# # ** Pseudocode: **
# # ```text
# # parent_agent:
# # data = load_raw_data()
# #
# # extracted = spawn(task="Extract", input=data).wait()
# # cleaned = spawn(task="Clean", input=extracted).wait()
# # analyzed = spawn(task="Analyze", input=cleaned).wait()
# # report = spawn(task="Report", input=analyzed).wait()
# #
# # return report
# # ```
# #
# # ** Pros: ** Clear
# # stages, easy
# # to
# # track
# # progress, failures
# # are
# # local
# #
# # ** Cons: ** No
# # parallelism—each
# # stage
# # waits
# # for previous
# #
# # ## Pattern 3: Supervisor (Hierarchical)
# #
# # ** What
# # it is: ** Parent
# # agent
# # directs
# # multiple
# # specialized
# # subagents, each
# # with different tools and expertise.
# #
# # ** When
# # to
# # use: ** Complex
# # tasks
# # requiring
# # multiple
# # specialist
# # agents.
# #
# # ** Example: ** Product
# # launch
# # report
# # needs
# # sales, engineering, and marketing
# # data.
# #
# # ```
# # Parent
# # Agent(Orchestrator)
# # ├─ Sales
# # Subagent(has: CRM
# # tools, revenue
# # metrics)
# # ├─ Engineering
# # Subagent(has: GitHub
# # API, deployment
# # tools)
# # └─ Marketing
# # Subagent(has: analytics, social
# # tools)
# #
# # Parent
# # aggregates → Launch
# # Report
# # ```
# #
# # ** Pseudocode: **
# # ```text
# # parent_agent:
# # sales = spawn(
# #     task="Get sales metrics",
# #     tools=[crm, revenue_db]
# # )
# # engineering = spawn(
# #     task="Get feature status",
# #     tools=[github, ci_cd]
# # )
# # marketing = spawn(
# #     task="Get market sentiment",
# #     tools=[analytics, social]
# # )
# #
# # results = wait_all([sales, engineering, marketing])
# # report = combine(results)
# # return report
# # ```
# #
# # ** Pros: ** Specialized
# # agents, clean
# # separation
# # of
# # tools, parallel
# # execution
# #
# # ** Cons: ** More
# # complex
# # coordination
# #
# # ## Pattern 4: Swarm (Collaborative)
# #
# # ** What
# # it is: ** Multiple
# # subagents
# # work
# # on
# # the
# # same
# # problem, comparing
# # approaches and converging
# # on
# # best
# # solution.
# #
# # ** When
# # to
# # use: ** Complex
# # problems
# # where
# # multiple
# # perspectives
# # improve
# # quality.
# #
# # ** Example: ** Design
# # review
# # with architect, security, and performance subagents.
# #
# # ```
# # Parent: "Design new API"
# # ├─ Subagent
# # 1(Architect): Review
# # structure
# # ├─ Subagent
# # 2(Security): Review
# # for vulnerabilities
# #     ├─ Subagent
# #     3(Performance): Review
# #     for bottlenecks
# #
# # Parent: Incorporate
# # feedback → Final
# # design
# # ```
# #
# # ** Pseudocode: **
# # ```text
# # parent_agent:
# # design = "Initial API design..."
# #
# # # Round 1: Independent reviews
# # architect_review = spawn(task=f"Review design as architect", input=design).wait()
# # security_review = spawn(task=f"Review design for security", input=design).wait()
# # perf_review = spawn(task=f"Review design for performance", input=design).wait()
# #
# # # Incorporate feedback
# # improved_design = combine(design, architect_review, security_review, perf_review)
# #
# # return improved_design
# # ```
# #
# # ** Pros: ** Multiple
# # perspectives
# # improve
# # quality, catches
# # blind
# # spots
# #
# # ** Cons: ** Can
# # be
# # slow(requires
# # multiple
# # rounds)
# #
# # ## Pattern Comparison
# #
# # | Pattern | Use
# # Case | Parallelism | Coordination |
# # | --------- | ---------- | ------------- | -------------- |
# # | ** Fan - Out / Fan - In ** | Independent
# # tasks | High | Low |
# # | ** Pipeline ** | Sequential
# # stages | None | Medium |
# # | ** Supervisor ** | Multiple
# # specialists | Medium - High | Medium - High |
# # | ** Swarm ** | Collaborative
# # design | Low - Medium | High |
# #
# # ## When NOT to Use Subagents
# #
# # We
# # covered
# # the
# # high - level
# # guidance in the
# # unit
# # introduction.Here
# # 's how it plays out in practice with code examples.
# #
# # ### 1. Sequential Dependent Work
# # ```text
# # # DON'T do this
# # subagent_a = spawn(task="Step 1: Set up environment").wait()
# # subagent_b = spawn(task="Step 2: Deploy code", input=subagent_a).wait()
# #
# # # BETTER: Single agent for sequential work
# # single_agent(task="Set up environment, then deploy")
# # ```
# #
# # ### 2. Same-File Parallel Edits
# # ```text
# # # DON'T do this
# # subagent_a = spawn(task="Add feature A to app.py")
# # subagent_b = spawn(task="Add feature B to app.py")  # Conflict!
# #
# # # BETTER: Fan-out on different files
# # subagent_a = spawn(task="Write feature A in feature_a.py")
# # subagent_b = spawn(task="Write feature B in feature_b.py")
# # ```
# #
# # ### 3. Small Quick Tasks
# # ```text
# # # DON'T do this for simple tasks
# # subagent = spawn(task="Format this JSON")  # Overhead: 1-2s, task: 0.1s
# #
# # # BETTER: Do it in parent
# # formatted = json.dumps(data)
# # ```
# #
# # ### 4. Too Many Specialist Agents
# # ```text
# # # DON'T do this
# # for i in range(20):  # 20 specialist subagents
# #     subagent = spawn(task=f"Specialist {i} task")
# #
# # # BETTER: Group specialists
# # senior_architect = spawn(task="Lead architecture design")
# # implementation_team = spawn(task="Implement based on design")
# # ```
# #
# # ## Key Takeaways
# #
# # Fan - out / fan - in parallelises
# # independent
# # work, pipeline
# # chains
# # stages, supervisor
# # coordinates
# # specialists, and swarm
# # cross - reviews
# # a
# # single
# # artefact.Reach
# # for subagents when you have many files or several independent tasks; stick with a single agent for small jobs and tightly coupled workflows.
# #
# # Next, how
# # to
# # actually
# # invoke
# # subagents in Claude
# # Code and Codex.
# #
# # Subagent Patterns
#
# Multi - agent
# workflows
# tend
# to
# fall
# into
# a
# handful
# of
# shapes.We
# will
# go
# over
# 4
# of
# them
# that
# cover
# almost
# anything
# you
# would
# build.
#
# < iframe
# src = "https://context-course-subagent-patterns.static.hf.space"
# frameborder = "0"
# width = "850"
# height = "450"
# >
#
# ## Pattern 1: Fan-Out / Fan-In
#
# ** What
# it is: ** Spawn
# multiple
# subagents in parallel
# for independent subtasks, then aggregate results.
#
# ** When
# to
# use: ** Independent
# tasks
# with no dependencies between them.
#
# ** Example: ** Evaluate
# 5
# models
# on
# the
# same
# benchmark.
#
# ```
# Parent: "Compare models"
# ├─ Subagent
# 1: Evaluate
# Model
# A → Score: 92 %
# ├─ Subagent
# 2: Evaluate
# Model
# B → Score: 89 %
# ├─ Subagent
# 3: Evaluate
# Model
# C → Score: 94 %
# ├─ Subagent
# 4: Evaluate
# Model
# D → Score: 88 %
# └─ Subagent
# 5: Evaluate
# Model
# E → Score: 91 %
#
# Parent: Aggregate → Ranking
# report
# ```
#
# ** Pseudocode: **
# ```text
# parent_agent:
# subagents = []
# for model in [A, B, C, D, E]:
#     subagent = spawn(
#         task=f"Evaluate {model}",
#         context={model}
#     )
#     subagents.append(subagent)
#
# results = wait_all(subagents)
# report = aggregate(results)
# return report
# ```
#
# ** Pros: ** Fully
# parallel, max
# speed, easy
# to
# scale
#
# ** Cons: ** Total
# time = slowest
# subagent
#
# ## Pattern 2: Pipeline
#
# ** What
# it is: ** Chain
# subagents
# where
# output
# of
# one
# becomes
# input
# to
# the
# next.
#
# ** When
# to
# use: ** Sequential
# processing
# with data flowing through stages.
#
# ** Example: ** Data
# processing: extract → clean → analyze → report
#
# ```
# Raw
# Data
# ↓
# [Subagent 1: Extract]
# ↓
# Extracted
# Data
# ↓
# [Subagent 2: Clean]
# ↓
# Clean
# Data
# ↓
# [Subagent 3: Analyze]
# ↓
# Results
# ↓
# [Subagent 4: Report]
# ↓
# Final
# Report
# ```
#
# ** Pseudocode: **
# ```text
# parent_agent:
# data = load_raw_data()
#
# extracted = spawn(task="Extract", input=data).wait()
# cleaned = spawn(task="Clean", input=extracted).wait()
# analyzed = spawn(task="Analyze", input=cleaned).wait()
# report = spawn(task="Report", input=analyzed).wait()
#
# return report
# ```
#
# ** Pros: ** Clear
# stages, easy
# to
# track
# progress, failures
# are
# local
#
# ** Cons: ** No
# parallelism—each
# stage
# waits
# for previous
#
# ## Pattern 3: Supervisor (Hierarchical)
#
# ** What
# it is: ** Parent
# agent
# directs
# multiple
# specialized
# subagents, each
# with different tools and expertise.
#
# ** When
# to
# use: ** Complex
# tasks
# requiring
# multiple
# specialist
# agents.
#
# ** Example: ** Product
# launch
# report
# needs
# sales, engineering, and marketing
# data.
#
# ```
# Parent
# Agent(Orchestrator)
# ├─ Sales
# Subagent(has: CRM
# tools, revenue
# metrics)
# ├─ Engineering
# Subagent(has: GitHub
# API, deployment
# tools)
# └─ Marketing
# Subagent(has: analytics, social
# tools)
#
# Parent
# aggregates → Launch
# Report
# ```
#
# ** Pseudocode: **
# ```text
# parent_agent:
# sales = spawn(
#     task="Get sales metrics",
#     tools=[crm, revenue_db]
# )
# engineering = spawn(
#     task="Get feature status",
#     tools=[github, ci_cd]
# )
# marketing = spawn(
#     task="Get market sentiment",
#     tools=[analytics, social]
# )
#
# results = wait_all([sales, engineering, marketing])
# report = combine(results)
# return report
# ```
#
# ** Pros: ** Specialized
# agents, clean
# separation
# of
# tools, parallel
# execution
#
# ** Cons: ** More
# complex
# coordination
#
# ## Pattern 4: Swarm (Collaborative)
#
# ** What
# it is: ** Multiple
# subagents
# work
# on
# the
# same
# problem, comparing
# approaches and converging
# on
# best
# solution.
#
# ** When
# to
# use: ** Complex
# problems
# where
# multiple
# perspectives
# improve
# quality.
#
# ** Example: ** Design
# review
# with architect, security, and performance subagents.
#
# ```
# Parent: "Design new API"
# ├─ Subagent
# 1(Architect): Review
# structure
# ├─ Subagent
# 2(Security): Review
# for vulnerabilities
#     ├─ Subagent
#     3(Performance): Review
#     for bottlenecks
#
# Parent: Incorporate
# feedback → Final
# design
# ```
#
# ** Pseudocode: **
# ```text
# parent_agent:
# design = "Initial API design..."
#
# # Round 1: Independent reviews
# architect_review = spawn(task=f"Review design as architect", input=design).wait()
# security_review = spawn(task=f"Review design for security", input=design).wait()
# perf_review = spawn(task=f"Review design for performance", input=design).wait()
#
# # Incorporate feedback
# improved_design = combine(design, architect_review, security_review, perf_review)
#
# return improved_design
# ```
#
# ** Pros: ** Multiple
# perspectives
# improve
# quality, catches
# blind
# spots
#
# ** Cons: ** Can
# be
# slow(requires
# multiple
# rounds)
#
# ## Pattern Comparison
#
# | Pattern | Use
# Case | Parallelism | Coordination |
# | --------- | ---------- | ------------- | -------------- |
# | ** Fan - Out / Fan - In ** | Independent
# tasks | High | Low |
# | ** Pipeline ** | Sequential
# stages | None | Medium |
# | ** Supervisor ** | Multiple
# specialists | Medium - High | Medium - High |
# | ** Swarm ** | Collaborative
# design | Low - Medium | High |
#
# ## When NOT to Use Subagents
#
# We
# covered
# the
# high - level
# guidance in the
# unit
# introduction.Here
# 's how it plays out in practice with code examples.
#
# ### 1. Sequential Dependent Work
# ```text
# # DON'T do this
# subagent_a = spawn(task="Step 1: Set up environment").wait()
# subagent_b = spawn(task="Step 2: Deploy code", input=subagent_a).wait()
#
# # BETTER: Single agent for sequential work
# single_agent(task="Set up environment, then deploy")
# ```
#
# ### 2. Same-File Parallel Edits
# ```text
# # DON'T do this
# subagent_a = spawn(task="Add feature A to app.py")
# subagent_b = spawn(task="Add feature B to app.py")  # Conflict!
#
# # BETTER: Fan-out on different files
# subagent_a = spawn(task="Write feature A in feature_a.py")
# subagent_b = spawn(task="Write feature B in feature_b.py")
# ```
#
# ### 3. Small Quick Tasks
# ```text
# # DON'T do this for simple tasks
# subagent = spawn(task="Format this JSON")  # Overhead: 1-2s, task: 0.1s
#
# # BETTER: Do it in parent
# formatted = json.dumps(data)
# ```
#
# ### 4. Too Many Specialist Agents
# ```text
# # DON'T do this
# for i in range(20):  # 20 specialist subagents
#     subagent = spawn(task=f"Specialist {i} task")
#
# # BETTER: Group specialists
# senior_architect = spawn(task="Lead architecture design")
# implementation_team = spawn(task="Implement based on design")
# ```
#
# ## Key Takeaways
#
# Fan - out / fan - in parallelises
# independent
# work, pipeline
# chains
# stages, supervisor
# coordinates
# specialists, and swarm
# cross - reviews
# a
# single
# artefact.Reach
# for subagents when you have many files or several independent tasks; stick with a single agent for small jobs and tightly coupled workflows.
#
# Next, how
# to
# actually
# invoke
# subagents in Claude
# Code and Codex.
#
# # Hands-On: Multi-Agent Workflow
#
# This
# project
# builds
# a ** research → implement → verify ** pipeline.A
# researcher
# subagent
# maps
# the
# existing
# code, an
# implementer
# writes
# the
# change, and two
# read - only
# reviewers(security and performance)
# sign
# off in parallel
# before
# the
# change is merged.
#
# ## Project Setup
#
# Create
# a
# new
# project:
#
# ```bash
# mkdir
# code - quality - pipeline
# cd
# code - quality - pipeline
#
# # Create files
# mkdir - p.claude / agents
# touch
# main.py
# touch.claude / CLAUDE.md
# ```
#
# ```bash
# mkdir
# code - quality - pipeline
# cd
# code - quality - pipeline
#
# # Create files
# mkdir - p.codex / agents
# touch
# main.py
# touch
# AGENTS.md
# ```
#
# ```bash
# mkdir
# code - quality - pipeline
# cd
# code - quality - pipeline
#
# # Create files
# mkdir - p.pi / agents.pi / extensions / subagent.pi / prompts
# touch
# main.py
# touch
# AGENTS.md
# ```
#
# Use
# the
# official
# Pi
# subagent
# example( or your
# own
# equivalent
# package) to
# populate
# `.pi / extensions / subagent / ` before
# you
# start
# Pi.
#
# ## Step 1: Define Custom Agents
#
# The
# workflow is the
# same
# on
# both
# platforms: define
# a
# narrow
# researcher, a
# writer, and two
# read - only
# reviewers.The
# file
# format
# differs.
#
# Create
# `.claude / agents / researcher.md
# `:
#
# ```markdown
# ---
# name: researcher
# description: Explores
# codebase and documents
# architecture
# tools: Read, Grep, Glob
# model: sonnet
# ---
# You
# are
# an
# architecture
# researcher.Your
# job is to:
# 1.
# Explore
# the
# codebase
# structure
# 2.
# Identify
# key
# modules, dependencies, and patterns
# 3.
# Document
# the
# architecture
# clearly
# 4.
# Note
# areas
# for improvement
#
# Be
# thorough.Read
# 10 + files
# to
# build
# a
# complete
# picture.
# ```
#
# Create
# `.claude / agents / implementer.md
# `:
#
# ```markdown
# ---
# name: implementer
# description: Writes
# code
# based
# on
# specifications
# tools: Read, Write, Glob, Bash
# model: sonnet
# ---
# You
# are
# a
# skilled
# developer.Your
# job is to:
# 1.
# Read
# the
# specification and architecture
# docs
# 2.
# Write
# clean, well - tested
# code
# 3.
# Follow
# existing
# patterns and conventions
# 4.
# Document
# your
# changes
#
# Focus
# on
# correctness and maintainability.
# ```
#
# Create
# `.claude / agents / security - reviewer.md
# `:
#
# ```markdown
# ---
# name: security - reviewer
# description: Reviews
# code
# for vulnerabilities
#     tools: Read, Grep
# model: sonnet
# ---
# You
# are
# a
# security
# expert.Your
# job is to:
# 1.
# Review
# code
# for security vulnerabilities
#     2.
#     Check
#     for data leakage, injection attacks, etc.
# 3.
# Recommend
# fixes
# 4.
# Assess
# overall
# security
# posture
#
# Be
# thorough.This
# code
# might
# be
# production - critical.
# ```
#
# Create
# `.claude / agents / performance - reviewer.md
# `:
#
# ```markdown
# ---
# name: performance - reviewer
# description: Reviews
# code
# for performance issues
#     tools: Read, Grep
# model: sonnet
# ---
# You
# are
# a
# performance
# specialist.Your
# job is to:
# 1.
# Identify
# performance
# bottlenecks
# 2.
# Suggest
# optimizations
# 3.
# Calculate
# impact
# 4.
# Recommend
# best
# practices
#
# Be
# specific
# with numbers and concrete suggestions.
# ```
#
# Create
# `.codex / agents / researcher.toml
# `:
#
# ```toml
# name = "researcher"
# description = "Read-only codebase explorer for mapping architecture before implementation."
# sandbox_mode = "read-only"
# developer_instructions = """
# Stay in exploration mode.
# Read broadly, trace the real execution path, and summarize patterns before proposing changes.
# """
# ```
#
# Create
# `.codex / agents / implementer.toml
# `:
#
# ```toml
# name = "implementer"
# description = "Implementation-focused agent for writing the smallest defensible code change."
# developer_instructions = """
# Read the specification and research notes first.
# Own the code change, keep scope tight, and validate what you modify.
# """
# ```
#
# Create
# `.codex / agents / security - reviewer.toml
# `:
#
# ```toml
# name = "security_reviewer"
# description = "Read-only reviewer focused on vulnerabilities, secrets, and privilege boundaries."
# sandbox_mode = "read-only"
# developer_instructions = """
# Lead with concrete security risks.
# Prioritize data leakage, injection, auth mistakes, and unsafe defaults.
# """
# ```
#
# Create
# `.codex / agents / performance - reviewer.toml
# `:
#
# ```toml
# name = "performance_reviewer"
# description = "Read-only reviewer focused on latency, query count, and caching opportunities."
# sandbox_mode = "read-only"
# developer_instructions = """
# Look for expensive loops, repeated queries, and missing caches.
# Prefer concrete bottlenecks over generic performance advice.
# """
# ```
#
# Create
# `.pi / agents / researcher.md
# `:
#
# ```markdown
# ---
# name: researcher
# description: Explores
# codebase and documents
# architecture
# tools: read, grep, find, ls
# ---
# You
# are
# an
# architecture
# researcher.Your
# job is to:
# 1.
# Explore
# the
# codebase
# structure
# 2.
# Identify
# key
# modules, dependencies, and patterns
# 3.
# Document
# the
# architecture
# clearly
# 4.
# Note
# areas
# for improvement
#
# Be
# thorough.Read
# 10 + files
# to
# build
# a
# complete
# picture.
# ```
#
# Create
# `.pi / agents / implementer.md
# `:
#
# ```markdown
# ---
# name: implementer
# description: Writes
# code
# based
# on
# specifications
# tools: read, grep, find, ls, bash, edit, write
# ---
# You
# are
# a
# skilled
# developer.Your
# job is to:
# 1.
# Read
# the
# specification and architecture
# docs
# 2.
# Write
# clean, well - tested
# code
# 3.
# Follow
# existing
# patterns and conventions
# 4.
# Document
# your
# changes
#
# Focus
# on
# correctness and maintainability.
# ```
#
# Create
# `.pi / agents / security - reviewer.md
# `:
#
# ```markdown
# ---
# name: security - reviewer
# description: Reviews
# code
# for vulnerabilities
#     tools: read, grep, find, ls
# ---
# You
# are
# a
# security
# expert.Your
# job is to:
# 1.
# Review
# code
# for security vulnerabilities
#     2.
#     Check
#     for data leakage, injection attacks, and unsafe secrets handling
# 3.
# Recommend
# fixes
# 4.
# Assess
# overall
# security
# posture
#
# Be
# thorough.This
# code
# might
# be
# production - critical.
# ```
#
# Create
# `.pi / agents / performance - reviewer.md
# `:
#
# ```markdown
# ---
# name: performance - reviewer
# description: Reviews
# code
# for performance issues
#     tools: read, grep, find, ls
# ---
# You
# are
# a
# performance
# specialist.Your
# job is to:
# 1.
# Identify
# performance
# bottlenecks
# 2.
# Suggest
# optimizations
# 3.
# Calculate
# impact
# 4.
# Recommend
# best
# practices
#
# Be
# specific
# with numbers and concrete suggestions.
# ```
#
# ## Step 2: Define Project Policies
#
# Keep
# the
# preferred
# workflow in a
# repo - level
# instruction
# file
# so
# the
# team
# uses
# the
# same
# sequence
# every
# time.
#
# Create
# `.claude / CLAUDE.md
# `:
#
# ```markdown
# # Code Quality Pipeline
#
# ## Architecture
#
# This
# project
# uses
# a
# multi - agent
# workflow:
# 1. ** Research ** — Understand
# existing
# code
# 2. ** Implement ** — Write
# new
# features
# 3. ** Verify ** — Security and performance
# review
#
# ## Workflow Triggers
#
# ### Before Implementation
# When
# the
# user
# asks
# for a new feature:
#     1.
#     Use
#     the
#     researcher
#     subagent
#     first
# 2.
# Have
# it
# explore
# the
# codebase(15 + files)
# 3.
# Document
# architecture and relevant
# patterns
# 4.
# Pass
# findings
# to
# implementation
# stage
#
# ### During Implementation
# 1.
# Spawn
# the
# implementer
# subagent
# 2.
# Provide
# it
# with:
#     - Feature
#     specification(
#     from user)
#     - Architecture
#     findings(
#     from researcher)
#     - Existing
#     code
#     patterns
# 3.
# Have
# it
# write
# the
# code
#
# ### Before Merge
# 1.
# Spawn
# security - reviewer and performance - reviewer in parallel
# 2.
# Both
# are
# read - only(no
# Write
# access)
# 3.
# Collect
# findings
# 4.
# Report
# to
# user
# before
# committing
#
# ## Example: Add Payment System
#
# User: "Add Stripe integration"
#
# → Researcher
# explores
# payment - related
# code
# → Implementer
# writes
# integration
# based
# on
# findings
# → Security - reviewer
# checks
# for token leakage
#     → Performance - reviewer
#     checks
#     database
#     queries
# → User
# reviews
# all
# findings and decides
# to
# merge
# ```
#
# Create
# `AGENTS.md`:
#
# ```markdown
# # Code Quality Pipeline
#
# For
# feature
# work, prefer
# a
# research -> implement -> verify
# flow.
#
# - Start
# by
# spawning
# `researcher`
# to
# map
# the
# affected
# code
# paths and document
# current
# patterns.
# - Then
# spawn
# `implementer`
# to
# own
# the
# change.
# - Before
# merge, spawn
# `security_reviewer` and `performance_reviewer` in parallel and wait
# for both.
#     - Keep
#     review
#     agents
#     read - only and report
#     findings
#     before
#     committing.
# ```
#
# Create
# `AGENTS.md`:
#
# ```markdown
# # Code Quality Pipeline
#
# For
# feature
# work, prefer
# a
# research -> implement -> verify
# flow.
#
# - Start
# by
# using
# `researcher`
# to
# map
# the
# affected
# code
# paths and document
# current
# patterns.
# - Then
# hand
# the
# findings
# to
# `implementer`
# so
# one
# agent
# owns
# the
# change.
# - Before
# merge, run
# `security - reviewer` and `performance - reviewer` in parallel and wait
# for both.
#     - Keep
#     review
#     agents
#     read - only and report
#     findings
#     before
#     committing.
# ```
#
# If
# you
# use
# the
# official
# subagent
# example, you
# can
# also
# add
# project - local
# prompt
# templates
# under
# `.pi / prompts / `
# for common flows such as `research -> implement -> review`.
#
# ## Step 3: Using the Pipeline
#
# ### Scenario: Add Feature
#
# Prompt
# your
# agent:
#
# ```
# I
# want
# to
# add
# OAuth2
# authentication.
#
# 1.
# Use
# the
# researcher
# subagent
# to
# explore
# our
# current
# auth
# system.
# Read
# the
# auth
# module
# thoroughly(15 + files).
# Document
# the
# current
# architecture and patterns.
#
# 2.
# Use
# the
# implementer
# subagent
# to
# write
# the
# OAuth2
# integration.
# Base
# it
# on
# the
# architecture
# findings.
# Follow
# existing
# patterns.
#
# 3.
# Once
# complete, have
# security - reviewer and performance - reviewer
# review
# the
# code in parallel.
# I
# want
# to
# see
# vulnerabilities and bottlenecks
# before
# merging.
# ```
#
# ```
# I
# want
# to
# add
# OAuth2
# authentication.
#
# Spawn
# `researcher`
# first and have
# it
# map
# our
# current
# auth
# system
# across
# 15 + files.
# Then
# spawn
# `implementer`
# to
# add
# the
# OAuth2
# integration
# based
# on
# those
# findings.
# Finally, spawn
# `security_reviewer` and `performance_reviewer` in parallel, wait
# for both, and summarize the risks before we merge.
# ```
#
# ```
# I
# want
# to
# add
# OAuth2
# authentication.
#
# Use
# `researcher`
# to
# map
# our
# current
# auth
# system
# across
# 15 + files.
# Then
# use
# `implementer`
# to
# add
# the
# OAuth2
# integration
# based
# on
# those
# findings.
# Finally, run
# `security - reviewer` and `performance - reviewer` in parallel, wait
# for both, and summarize the risks before we merge.
# ```
#
# Typical
# result:
#
# ```
# Step
# 1: Spawning
# researcher
# subagent...
# ✓ Exploring
# auth / directory
# ✓ Reading
# 18
# relevant
# files
# ✓ Documenting
# patterns and dependencies
# → Findings: Current
# system
# uses
# JWT
# with Redis cache.
# Token
# rotation
# every
# 1
# hour.
# Validates
# on
# every
# API
# call.
#
# Step
# 2: Spawning
# implementer
# subagent...
# ✓ Writing
# OAuth2
# provider
# integration
# ✓ Following
# existing
# JWT
# pattern
# ✓ Added
# tests
# → Implementation: 450
# lines
# of
# new
# code + tests
#
# Step
# 3: Spawning
# security - reviewer and performance - reviewer in parallel...
# ✓ Security
# review
# complete
# Findings: HTTPS
# enforced? ✓ Tokens
# hashed? ✓ No
# secrets in logs? ✓
# Recommendation: Add
# rate
# limiting
# to
# OAuth
# endpoint
#
# ✓ Performance
# review
# complete
# Findings: 3
# database
# queries
# per
# auth(expected)
# Recommendation: Cache
# OAuth
# provider
# metadata
#
# Ready
# for merge? Review findings above.
# ```
#
# ## Step 4: Fan-Out Example
#
# When
# researching, the
# parent
# agent
# can
# fan
# out
# into
# parallel
# exploration
# threads:
#
# ```
# Research
# codebase
# structure
#
# [Research subagent 1: Explore
# backend /]
# ├─ authentication /
# ├─ database /
# └─ api /
#
# [Research subagent 2: Explore
# frontend /]
# ├─ components /
# ├─ state /
# └─ pages /
#
# [Research subagent 3: Explore
# infrastructure /]
# ├─ docker /
# ├─ kubernetes /
# └─ monitoring /
#
# All
# 3
# run in parallel → Parent
# aggregates
# findings
# ```
#
# Prompt
# your
# agent:
#
# ```
# Use
# 3
# subagents
# to
# research
# our
# codebase:
# - Researcher
# 1: Backend
# architecture
# - Researcher
# 2: Frontend
# architecture
# - Researcher
# 3: Infrastructure
# setup
#
# Each
# should
# explore
# 15 + files and document
# patterns.
# Then
# combine
# findings
# into
# an
# architecture
# overview.
# ```
#
# ## Step 5: Pipeline Review Pattern
#
# Classic
# pipeline: design → code → test → deploy
#
# ```
# Stage
# 1: Architecture
# Design
# └─ Designer
# subagent: Create
# schema, define
# interfaces
#
# Stage
# 2: Implementation
# └─ Developer
# subagent: Write
# code
# based
# on
# design
#
# Stage
# 3: Security
# Review
# └─ Security
# subagent: Review
# for vulnerabilities
#
# Stage
# 4: Testing
# └─ QA
# subagent: Write and run
# tests
#
# Stage
# 5: Deployment
# Planning
# └─ DevOps
# subagent: Plan
# deployment
# strategy
#
# Parent: Aggregate
# findings and present
# to
# user
# ```
#
# Command:
#
# ```
# Use
# a
# pipeline
# workflow
# to
# implement
# a
# new
# API
# endpoint:
#
# Stage
# 1: Design
# the
# endpoint
# schema and request / response
# formats
# Stage
# 2: Implement
# the
# endpoint
# using
# the
# design
# Stage
# 3: Security
# review
# the
# implementation
# Stage
# 4: Write
# comprehensive
# tests
# Stage
# 5: Plan
# the
# deployment
#
# Each
# stage is a
# subagent.Run
# sequentially(output
# of
# stage
# N → input
# to
# stage
# N + 1).
# ```
#
# ## Advanced: Keep Automation Separate from Delegation
#
# Use
# repo
# instructions
# to
# define
# when
# review
# agents
# should
# be
# used.If
# you
# also
# want
# automation, use
# the
# platform
# 's real automation surface instead of inventing a subagent DSL.
#
# Use
# `.claude / CLAUDE.md
# `
# for delegation guidance.If you also want automated checks, use `.claude / settings.json` hooks for commands such as test or lint scripts:
#
# ```json
# {
#     "hooks": {
#         "PostToolUse": [
#             {
#                 "matcher": "Write|Edit",
#                 "hooks": [
#                     {
#                         "type": "command",
#                         "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/run-checks.sh"
#                     }
#                 ]
#             }
#         ]
#     }
# }
# ```
#
# Use
# `AGENTS.md`
# for delegation guidance.Keep runtime limits such as thread caps in `.codex / config.toml`, and ask Codex explicitly when you want it to spawn named agents.
#
# Use
# `AGENTS.md` and prompt
# templates
# for delegation.Keep automation in extensions, for example `.pi / extensions / pipeline-guard.ts`:
#
# ```ts
# import type
#
# {ExtensionAPI}
# from
#
# "@mariozechner/pi-coding-agent";
#
# export
# default
# function(pi: ExtensionAPI) {
#     pi.on("tool_call", async(event) = > {
# if (event.toolName !== "bash")
# return;
#
# const
# command = event.input.command as string;
# if (/ rm -rf | git reset --hard /.test(command)) {
# return {block: true, reason: "Dangerous command blocked by pipeline policy"};
# }
# });
# }
# ```
#
# ## Example Results
#
# When
# the
# pipeline
# completes:
#
# ```json
# {
#     "researcher": {
#         "architecture": {
#             "modules": ["auth", "api", "database"],
#             "patterns": ["MVC", "Repository pattern", "Middleware"],
#             "dependencies": ["Express.js", "MongoDB", "Redis"]
#         }
#     },
#     "implementer": {
#         "files_created": ["oauth2.js", "oauth2.test.js"],
#         "lines_of_code": 450,
#         "test_coverage": 94
#     },
#     "security_reviewer": {
#         "vulnerabilities": 0,
#         "warnings": 1,
#         "recommendations": ["Add rate limiting"]
#     },
#     "performance_reviewer": {
#         "database_queries_per_request": 3,
#         "bottlenecks": 0,
#         "recommendations": ["Cache provider metadata"]
#     }
# }
# ```
#
# ## Best Practices Demonstrated
#
# This
# project
# demonstrates
# clear
# stage
# separation(research, implement, verify), specialized
# agents
# with specific tools and prompts, parallel execution for security and performance reviews, pipeline flow where each stage's output feeds the next, and repo-level instructions that keep the workflow consistent for the whole team.
#
# ## Key Takeaways
#
# A researcher agent explores and documents, an implementer writes code, and security and performance reviewers sign off in parallel.Define the specialists in `.claude / agents / *.md`, `.codex / agents / *.toml`, or `.pi / agents / *.md`, and keep the workflow in `CLAUDE.md` or `AGENTS.md`.Hooks, extensions, and runtime settings are for automation around the workflow, not for driving subagent orchestration.
#
# Next, the unit 4 quizzes.
#
