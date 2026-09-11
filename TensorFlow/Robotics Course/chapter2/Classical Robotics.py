"""
================================================================================
 Robotics Course · Chapter 2 · 经典机器人学（学习笔记描述）
================================================================================
 一句话：先懂经典机器人学(运动生成/正逆运动学),才明白为什么学习式方法更强。
 本章讲：
   ① 机器人如何产生运动、常见运动类型。
   ② 一个具体的运动学例子(正/逆运动学)。
   ③ 经典方法的局限 → 引出学习式(模仿/强化)方法。
 要点：经典法精确但难泛化/难处理复杂接触 → 学习法补位。
 说明：超长讲解(概念为主),参考为主。
================================================================================
"""

# # Classical Robotics
#
# In
# this
# section, we
# 'll build a foundation in classical robotics that will help you understand why learning-based methods are so powerful.
#
# We
# 'll start by exploring how robots generate motion, look at common types of robot movement, and work through a concrete example before discussing the limitations that motivate modern approaches.
#
# > [!TIP]
# >  ## Key Takeaway
# >
# > Learning - based
# approaches
# to
# robotics
# address
# fundamental
# challenges
# that
# traditional
# methods
# struggle
# with.
# >
# > Modern
# robotics
# needs
# methods
# that
# can
# work
# across
# different
# tasks and robot
# types, allowing
# one
# approach
# to
# work
# effectively in many
# situations
# rather
# than
# requiring
# custom
# solutions
# for each problem.We also need to reduce our dependency on human experts who manually design rules and models for every situation.Finally, the field needs approaches that can take advantage of the rapidly growing collection of robotics datasets, learning from the collective knowledge captured in these large-scale data collections.
#
# ## Different Approaches to Robot Motion
#
# Let
# 's start with the big picture: how do different approaches make robots move?
#
# Different
# methods
# for generating robot motion can be grouped based on whether they use explicit mathematical models or learn patterns implicitly from data.
#
# This is merely
# an
# overview
# of
# different
# methods
# to
# generate
# motion, and is clearly
# non - exhaustive.Still, it
# provides
# a
# good
# primer
# on
# what
# the
# most
# common
# approaches
# are in this
# circumstance.The
# most
# important
# grouping
# by
# far
# depend
# on
# whether
# the
# different
# methods
# explicitly(_dynamics - based_) or implicitly(_learning - based_)
# model
# robot - environment
# interactions.
#
# Further, knowledge
# of
# mechanical, electrical, and software
# engineering, as well as rigid‑body
# mechanics and control
# theory
# have
# proven
# quintessential in robotics
# since
# the
# field
# first
# developed in the
# 1950
# s.More
# recently, Machine
# Learning(ML)
# has
# also
# proved
# effective in robotics, complementing
# these
# more
# traditional
# disciplines.
#
# As
# a
# direct
# consequence
# of
# its
# multi‑disciplinary
# nature(to
# the
# very
# least, combining
# hardware and software), robotics
# has
# developed as a
# wide
# array
# of
# methods, all
# concerned
# with the main purpose of ** producing artificial motion in the physical world **.
#
# In
# this
# section, our
# goal is to
# introduce
# where
# classical
# methods
# excel, where
# they
# struggle, and why ** learning‑based
# approaches ** are
# helpful.
#
# > [!TIP]
# > ** Explicit
# vs
# Implicit
# Models: **
# >
# > ** Implicit(learning - based)
# approaches ** take
# a
# fundamentally
# different
# strategy
# by
# learning
# patterns
# directly
# from data rather
#
# than
# requiring
# explicit
# mathematical
# models.These
# methods
# require
# less
# domain - specific
# engineering and can
# adapt
# to
# complex, uncertain
# environments
# that
# would
# be
# difficult
# to
# model
# analytically.Neural
# networks and reinforcement
# learning
# algorithms
# are
# prime
# examples
# of
# this
# approach.
# >
# > ** Explicit(dynamics - based)
# approaches ** rely
# on
# hand - crafted
# mathematical
# models
# of
# physics and require
# deep
# domain
# expertise
# to
# be
# implemented
# effectively.These
# methods
# work
# exceptionally
# well
# for well - understood, controlled scenarios where the physics can be precisely modeled.Classic examples include PID controllers and Model Predictive Control systems that have been the backbone of industrial robotics for decades.
# >
# > ** Hybrid
# approaches ** represent
# an
# exciting
# middle
# ground, combining
# the
# reliability
# of
# physics
# knowledge
# with the adaptability of learning systems.These methods use physics knowledge to guide and constrain the learning process, often achieving better performance than either approach alone.
#
# ## Different Types of Motion
#
# Now
# that
# we
# have
# the
# big
# picture, we
# can
# situate
# the
# problem: what
# kinds
# of
# motion
# do
# robots
# typically
# perform?
#
# Different
# kinds
# of
# motions
# are
# achieved
# with potentially very different robotic platforms.From left to right, top to bottom: ViperX, SO - 100, Boston
# Dynamics
# ' Spot, Open-Duck, 1X'
# s
# NEO, Boston
# Dynamics
# ' Atlas. This is an example list of robotic platforms and is (very) far from being exhaustive.
#
# At
# a
# high
# level, most
# systems
# you’ll
# encounter
# fall
# into
# one
# of
# these
# three
# categories.Knowing
# which
# bucket
# you’re in helps
# you
# choose
# models, datasets, and controllers
# appropriately.
#
# In
# the
# vast
# majority
# of
# instances, robotics
# deals
# with producing motion via actuating joints connecting nearly entirely-rigid links.A key distinction between focus areas in robotics is based on whether the generated motion modifies the absolute state of the environment through dexterous interactions, changes the relative state of the robot with respect to its environment through mobility, or combines both capabilities.
#
# ** Manipulation ** involves
# generating
# motion
# to
# perform
# actions
# that
# induce
# desirable
# modifications in the
# environment.These
# effects
# are
# typically
# achieved * through * the
# robot -
# for example, a robotic arm grasping objects, assembling components, or using tools.The robot changes the world around it while remaining in a fixed location.
#
# ** Locomotion ** encompasses
# motions
# that
# result in changes
# to
# the
# robot
# 's physical location within its environment. This general category includes both *wheeled locomotion* (like mobile bases and autonomous vehicles) and *legged locomotion* (like walking robots and quadrupeds), depending on the mechanism the robot uses to move through its environment.
#
# > [!TIP]
# > Quick
# classifier: ask
# "what changes?"
# If
# mainly
# the
# world
# changes(object
# pose / state), you
# 're in manipulation. If mainly the robot state changes, you'
# re in locomotion.If
# both
# change
# meaningfully
# within
# the
# task, you
# 're in mobile manipulation. This simple test helps when designing observations, actions, and evaluation.
#
# We’ll
# reuse
# this
# taxonomy
# when
# discussing
# datasets(what
# sensors
# you
# need) and policies(what
# action
# spaces
# you
# predict) in the
# next
# sections.
#
# ## Example: Planar Manipulation
#
# Let’s
# ground
# the
# ideas
# with a concrete, minimal example you can reason about step by step.
#
# Robot
# manipulators
# typically
# consist
# of
# a
# series
# of
# links and joints, articulated in a
# chain finally connected
# to
# an * end - effector *.Actuated
# joints
# are
# considered
# responsible
# for generating motion of the links, while the end effector is instead used to perform specific actions at the target location (e.g., grasping / releasing objects via closing / opening a gripper end-effector, using a specialized tool like a screwdriver, etc.).
#
# Recently, the
# development
# of
# low - cost
# manipulators
# like
# the
# ALOHA, ALOHA - 2 and SO - 100 / SO - 101
# platforms
# significantly
# lowered
# the
# barrier
# to
# entry
# to
# robotics, considering
# the
# increased
# accessibility
# of
# these
# robots
# compared
# to
# more
# traditional
# platforms
# like
# the
# Franka
# Emika
# Panda
# arm.
#
# Cheaper, more
# accessible
# robots
# are
# starting
# to
# rival
# traditional
# platforms
# like
# the
# Panda
# arm
# platforms in adoption in resource - constrained
# scenarios.The
# SO - 100, in particular, has
# a
# cost in the
# 100
# s
# of
# Euros, and can
# be
# entirely
# 3
# D - printed in hours,
# while the industrially-manufactured Panda arm costs tens of thousands of Euros and is not openly available.
#
# ### Forward and Inverse Kinematics
#
# The
# SO - 100
# arm
# simplified
# to
# a
# 2
# D
# planar
# manipulator
# by
# preventing
# some
# joints
# from moving.
#
# Consider
# a
# simplified
# version
# of
# the
# SO - 100
# where
# we
# prevent
# some
# joints
# from moving.This reduces
#
# the
# complexity
# from
#
# 6
# degrees
# of
# freedom
# to
# just
# 2(plus
# the
# gripper).We
# can
# control
# two
# angles
# θ₁ and θ₂, which
# together
# define
# the
# robot
# 's configuration: q = [θ₁, θ₂].
#
# *Free
# to
# move *
#
# *Constrained
# by
# the
# surface *
#
# *Constrained
# by
# surface and (fixed)
# obstacle *
#
# Considering
# this
# example, we
# can
# analytically
# write
# the
# end - effector
# 's position $p \in \mathbb{R}^2$ as a function of the robot'
# s
# configuration, $p = p(q)$:
#
# $$p(q) = \begin
# {pmatrix}
# l \cos(\theta_1) + l \cos(\theta_1 + \theta_2) \\ l \sin(\theta_1) + l \sin(\theta_1 + \theta_2) \end
# {pmatrix}$$
#
# ** Forward
# Kinematics(FK) ** maps
# a
# robot
# configuration
# into
# the
# corresponding
# end - effector
# pose, whereas ** Inverse
# Kinematics(IK) ** is used
# to
# reconstruct
# the
# configuration(s)
# given
# an
# end - effector
# pose.
#
# In
# the
# simplified
# case
# here
# considered, one
# can
# solve
# the
# problem
# of
# controlling
# the
# end - effector
# 's location to reach a goal position $p^*$ by solving analytically for $q: p(q) = p^*$. However, in the general case, one might not be able to solve this problem analytically, and can typically resort to iterative optimization methods:
#
# $$\min_
# {q \ in  \mathcal
# {Q}} \ | p(q) - p ^ * \ | _2 ^ 2$$
#
# Exact
# analytical
# solutions
# to
# IK
# are
# even
# less
# appealing
# when
# one
# considers
# the
# presence
# of
# obstacles in the
# robot
# 's workspace, resulting in constraints on the possible values of $q$.
#
# > [!TIP]
# > If
# the
# math
# feels
# dense, focus
# on
# the
# mapping: FK
# answers
# "where is the hand given the joints?", IK
# asks
# "what joints reach that hand position?".The
# rest
# of
# the
# unit
# shows
# why
# the
# IK
# direction
# becomes
# hard in realistic
# settings.
#
# ### Differential Inverse Kinematics
#
# When
# IK is hard
# to
# solve
# directly, we
# can
# often
# make
# progress
# by
# working
# with small motions (velocities) instead of absolute positions.
#
# Let $J(q)$ denote
# the
# Jacobian
# matrix
# of(partial)
# derivatives
# of
# the
# FK - function.Then, one
# can
# apply
# the
# chain
# rule
# to
# any $p(q)$, deriving $\dot
# {p} = J(q) \dot
# {q}$, and thus finally relating
# variations in the
# robot
# configurations
# to
# variations in pose.
#
# Given
# a
# desired
# end - effector
# trajectory, differential
# IK
# finds $\dot
# {q}(t)$ solving
# for joints' *velocities* instead of *configurations*:
#
# $$\dot
# {q}(t) = \arg\min_\nu \ | J(q(t)) \nu - \dot
# {p} ^
# *(t)\ | _2 ^ 2$$
#
# This
# often
# admits
# the
# closed - form
# solution $\dot
# {q} = J(q) ^ + \dot
# {p} ^ * $, where $J ^ +(q)$ denotes
# the
# Moore - Penrose
# pseudo - inverse
# of $J(q)$.
#
# ### Adding Feedback Loops
#
# While
# very
# effective
# when
# a
# goal
# trajectory
# has
# been
# well
# specified, the
# performance
# of
# differential
# IK
# can
# degrade
# significantly in the
# presence
# of
# modeling / tracking
# errors, or in the
# presence
# of
# non - modeled
# dynamics in the
# environment.
#
# To
# mitigate
# the
# effect
# of
# modeling
# errors, sensing
# noise and other
# disturbances, classical
# pipelines
# indeed
# do
# augment
# differential
# IK
# with feedback control looping back quantities of interest.In practice, following a trajectory with a closed feedback loop might consist in backwarding the error between the target and measured pose, $\Delta p = p ^ * - p(q)$, hereby modifying the control applied to $\dot{q} = J(q) ^ + (\dot{p} ^ * + k_p \Delta p)$, with $k_p$ defined as the (proportional) gain.
#
# More
# advanced
# techniques
# for control consisting in feedback linearization, PID control, Linear Quadratic Regulator (LQR) or Model-Predictive Control (MPC) can be employed to stabilize tracking and reject moderate perturbations.
#
# ## Limitations of Dynamics-based Robotics
#
# This
# brings
# us
# to
# the “so
# what?”: where
# do
# these
# classical
# tools
# struggle in practice, and why
# does
# that
# motivate
# learning?
#
# Despite
# the
# last
# 60 + years
# of
# robotics
# research, autonomous
# robots
# are
# still
# largely
# incapable
# of
# performing
# tasks
# at
# human - level
# performance in the
# physical
# world
# generalizing
# across(1)
# robot
# embodiments(different
# manipulators, different
# locomotion
# platforms, etc.) and (2)
# tasks(tying
# shoe - laces, manipulating
# a
# diverse
# set
# of
# objects).
#
# Dynamics - based
# approaches
# to
# robotics
# suffer
# from several limitations: (1)
# orchestrating
# multiple
# components
# poses
# integration
# challenges;
# (2)
# the
# need
# to
# develop
# custom
# processing
# pipelines
# for the sensing modalities and tasks considered hinders scalability; (3) simplified analytical models of physical phenomena limit real-world performance.Lastly, (4) dynamics-based methods overlook trends in the availability and growth of robotics data.
#
# ### Key Limitations
#
# ** 1.
# Integration
# Challenges **
# Dynamics - based
# robotics
# pipelines
# have
# historically
# been ** developed
# sequentially, engineering
# the
# different
# blocks ** now
# within
# most
# architectures
# for specific purposes.That is, sensing, state estimation, mapping, planning, (diff-)IK, and low-level control have been traditionally developed as distinct modules with fixed interfaces.Pipelining these specific modules proved error-prone, and brittleness emerges—alongside compounding errors—whenever changes incur.
#
# ** 2.
# Limited
# Scalability **
# Classical
# planners
# operate
# on
# compact, assumed - sufficient
# state
# representations;
# extending
# them
# to
# reason
# directly
# over
# raw, heterogeneous and noisy
# data
# streams is non - trivial.This
# results in a ** limited
# scalability
# to
# multimodal
# data and multitask
# settings **, as incorporating
# high - dimensional
# perceptual
# inputs(RGB, depth, tactile, audio)
# traditionally
# required
# extensive
# engineering
# efforts
# to
# extract
# meaningful
# features
# for control.
#
# ** 3.
# Modeling
# Limitations **
# Setting
# aside
# integration and scalability
# challenges: developing
# accurate
# modeling
# of
# contact, friction, and compliance
# for complicated systems remains difficult.Rigid-body approximations are often insufficient in the presence of deformable objects, and ** relying on approximated models hinders real-world applicability ** of the methods developed.
#
# ** 4.
# Overlooking
# Data
# Trends **
# Lastly, dynamics - based
# methods(naturally)
# overlook
# the
# rather
# recent ** increase in availability
# of
# openly - available
# robotics
# datasets **.The
# curation
# of
# academic
# datasets
# by
# large
# centralized
# groups
# of
# human
# experts in robotics is now
# increasingly
# complemented
# by
# a ** growing
# number
# of
# robotics
# datasets
# contributed in a
# decentralized
# fashion ** by
# individuals
# with varied expertise.
#
# Taken
# together, these
# limitations
# motivate
# the
# exploration
# of
# learning - based
# approaches
# that
# can:
# 1. ** Integrate
# perception and control
# more
# tightly **
# 2. ** Adapt
# across
# tasks and embodiments **
# with reduced expert modeling interventions
# 3. ** Scale
# gracefully in performance ** as more
# robotics
# data
# becomes
# available
#
# ## References
#
# For
# a
# full
# list
# of
# references, check
# out
# the[tutorial](https: // huggingface.co / spaces / lerobot / robot - learning - tutorial).
#
# - ** Modern
# Robotics: Mechanics, Planning, and Control ** (2017)
# Kevin
# M.Lynch and Frank
# C.Park
# A
# comprehensive
# textbook
# covering
# the
# foundations
# of
# classical
# robotics, including
# kinematics, dynamics, and control.Essential
# reading
# for understanding the traditional approaches discussed in this unit.
# [Book Website](http: // hades.mech.northwestern.edu / index.php / Modern_Robotics)
#
# - ** Springer
# Handbook
# of
# Robotics ** (2016)
# Edited
# by
# Bruno
# Siciliano and Oussama
# Khatib
# An
# authoritative
# reference
# covering
# all
# aspects
# of
# robotics,
# from classical control
#
# theory
# to
# emerging
# learning - based
# approaches.
# [DOI: 10.1007 / 978 - 3 - 319 - 32552 - 1](https: // doi.org / 10.1007 / 978-3-319-32552-1)
#
# # Classical Robotics
#
# In
# this
# section, we
# 'll build a foundation in classical robotics that will help you understand why learning-based methods are so powerful.
#
# We
# 'll start by exploring how robots generate motion, look at common types of robot movement, and work through a concrete example before discussing the limitations that motivate modern approaches.
#
# > [!TIP]
# >  ## Key Takeaway
# >
# > Learning - based
# approaches
# to
# robotics
# address
# fundamental
# challenges
# that
# traditional
# methods
# struggle
# with.
# >
# > Modern
# robotics
# needs
# methods
# that
# can
# work
# across
# different
# tasks and robot
# types, allowing
# one
# approach
# to
# work
# effectively in many
# situations
# rather
# than
# requiring
# custom
# solutions
# for each problem.We also need to reduce our dependency on human experts who manually design rules and models for every situation.Finally, the field needs approaches that can take advantage of the rapidly growing collection of robotics datasets, learning from the collective knowledge captured in these large-scale data collections.
#
# ## Different Approaches to Robot Motion
#
# Let
# 's start with the big picture: how do different approaches make robots move?
#
# Different
# methods
# for generating robot motion can be grouped based on whether they use explicit mathematical models or learn patterns implicitly from data.
#
# This is merely
# an
# overview
# of
# different
# methods
# to
# generate
# motion, and is clearly
# non - exhaustive.Still, it
# provides
# a
# good
# primer
# on
# what
# the
# most
# common
# approaches
# are in this
# circumstance.The
# most
# important
# grouping
# by
# far
# depend
# on
# whether
# the
# different
# methods
# explicitly(_dynamics - based_) or implicitly(_learning - based_)
# model
# robot - environment
# interactions.
#
# Further, knowledge
# of
# mechanical, electrical, and software
# engineering, as well as rigid‑body
# mechanics and control
# theory
# have
# proven
# quintessential in robotics
# since
# the
# field
# first
# developed in the
# 1950
# s.More
# recently, Machine
# Learning(ML)
# has
# also
# proved
# effective in robotics, complementing
# these
# more
# traditional
# disciplines.
#
# As
# a
# direct
# consequence
# of
# its
# multi‑disciplinary
# nature(to
# the
# very
# least, combining
# hardware and software), robotics
# has
# developed as a
# wide
# array
# of
# methods, all
# concerned
# with the main purpose of ** producing artificial motion in the physical world **.
#
# In
# this
# section, our
# goal is to
# introduce
# where
# classical
# methods
# excel, where
# they
# struggle, and why ** learning‑based
# approaches ** are
# helpful.
#
# > [!TIP]
# > ** Explicit
# vs
# Implicit
# Models: **
# >
# > ** Implicit(learning - based)
# approaches ** take
# a
# fundamentally
# different
# strategy
# by
# learning
# patterns
# directly
# from data rather
#
# than
# requiring
# explicit
# mathematical
# models.These
# methods
# require
# less
# domain - specific
# engineering and can
# adapt
# to
# complex, uncertain
# environments
# that
# would
# be
# difficult
# to
# model
# analytically.Neural
# networks and reinforcement
# learning
# algorithms
# are
# prime
# examples
# of
# this
# approach.
# >
# > ** Explicit(dynamics - based)
# approaches ** rely
# on
# hand - crafted
# mathematical
# models
# of
# physics and require
# deep
# domain
# expertise
# to
# be
# implemented
# effectively.These
# methods
# work
# exceptionally
# well
# for well - understood, controlled scenarios where the physics can be precisely modeled.Classic examples include PID controllers and Model Predictive Control systems that have been the backbone of industrial robotics for decades.
# >
# > ** Hybrid
# approaches ** represent
# an
# exciting
# middle
# ground, combining
# the
# reliability
# of
# physics
# knowledge
# with the adaptability of learning systems.These methods use physics knowledge to guide and constrain the learning process, often achieving better performance than either approach alone.
#
# ## Different Types of Motion
#
# Now
# that
# we
# have
# the
# big
# picture, we
# can
# situate
# the
# problem: what
# kinds
# of
# motion
# do
# robots
# typically
# perform?
#
# Different
# kinds
# of
# motions
# are
# achieved
# with potentially very different robotic platforms.From left to right, top to bottom: ViperX, SO - 100, Boston
# Dynamics
# ' Spot, Open-Duck, 1X'
# s
# NEO, Boston
# Dynamics
# ' Atlas. This is an example list of robotic platforms and is (very) far from being exhaustive.
#
# At
# a
# high
# level, most
# systems
# you’ll
# encounter
# fall
# into
# one
# of
# these
# three
# categories.Knowing
# which
# bucket
# you’re in helps
# you
# choose
# models, datasets, and controllers
# appropriately.
#
# In
# the
# vast
# majority
# of
# instances, robotics
# deals
# with producing motion via actuating joints connecting nearly entirely-rigid links.A key distinction between focus areas in robotics is based on whether the generated motion modifies the absolute state of the environment through dexterous interactions, changes the relative state of the robot with respect to its environment through mobility, or combines both capabilities.
#
# ** Manipulation ** involves
# generating
# motion
# to
# perform
# actions
# that
# induce
# desirable
# modifications in the
# environment.These
# effects
# are
# typically
# achieved * through * the
# robot -
# for example, a robotic arm grasping objects, assembling components, or using tools.The robot changes the world around it while remaining in a fixed location.
#
# ** Locomotion ** encompasses
# motions
# that
# result in changes
# to
# the
# robot
# 's physical location within its environment. This general category includes both *wheeled locomotion* (like mobile bases and autonomous vehicles) and *legged locomotion* (like walking robots and quadrupeds), depending on the mechanism the robot uses to move through its environment.
#
# > [!TIP]
# > Quick
# classifier: ask
# "what changes?"
# If
# mainly
# the
# world
# changes(object
# pose / state), you
# 're in manipulation. If mainly the robot state changes, you'
# re in locomotion.If
# both
# change
# meaningfully
# within
# the
# task, you
# 're in mobile manipulation. This simple test helps when designing observations, actions, and evaluation.
#
# We’ll
# reuse
# this
# taxonomy
# when
# discussing
# datasets(what
# sensors
# you
# need) and policies(what
# action
# spaces
# you
# predict) in the
# next
# sections.
#
# ## Example: Planar Manipulation
#
# Let’s
# ground
# the
# ideas
# with a concrete, minimal example you can reason about step by step.
#
# Robot
# manipulators
# typically
# consist
# of
# a
# series
# of
# links and joints, articulated in a
# chain finally connected
# to
# an * end - effector *.Actuated
# joints
# are
# considered
# responsible
# for generating motion of the links, while the end effector is instead used to perform specific actions at the target location (e.g., grasping / releasing objects via closing / opening a gripper end-effector, using a specialized tool like a screwdriver, etc.).
#
# Recently, the
# development
# of
# low - cost
# manipulators
# like
# the
# ALOHA, ALOHA - 2 and SO - 100 / SO - 101
# platforms
# significantly
# lowered
# the
# barrier
# to
# entry
# to
# robotics, considering
# the
# increased
# accessibility
# of
# these
# robots
# compared
# to
# more
# traditional
# platforms
# like
# the
# Franka
# Emika
# Panda
# arm.
#
# Cheaper, more
# accessible
# robots
# are
# starting
# to
# rival
# traditional
# platforms
# like
# the
# Panda
# arm
# platforms in adoption in resource - constrained
# scenarios.The
# SO - 100, in particular, has
# a
# cost in the
# 100
# s
# of
# Euros, and can
# be
# entirely
# 3
# D - printed in hours,
# while the industrially-manufactured Panda arm costs tens of thousands of Euros and is not openly available.
#
# ### Forward and Inverse Kinematics
#
# The
# SO - 100
# arm
# simplified
# to
# a
# 2
# D
# planar
# manipulator
# by
# preventing
# some
# joints
# from moving.
#
# Consider
# a
# simplified
# version
# of
# the
# SO - 100
# where
# we
# prevent
# some
# joints
# from moving.This reduces
#
# the
# complexity
# from
#
# 6
# degrees
# of
# freedom
# to
# just
# 2(plus
# the
# gripper).We
# can
# control
# two
# angles
# θ₁ and θ₂, which
# together
# define
# the
# robot
# 's configuration: q = [θ₁, θ₂].
#
# *Free
# to
# move *
#
# *Constrained
# by
# the
# surface *
#
# *Constrained
# by
# surface and (fixed)
# obstacle *
#
# Considering
# this
# example, we
# can
# analytically
# write
# the
# end - effector
# 's position $p \in \mathbb{R}^2$ as a function of the robot'
# s
# configuration, $p = p(q)$:
#
# $$p(q) = \begin
# {pmatrix}
# l \cos(\theta_1) + l \cos(\theta_1 + \theta_2) \\ l \sin(\theta_1) + l \sin(\theta_1 + \theta_2) \end
# {pmatrix}$$
#
# ** Forward
# Kinematics(FK) ** maps
# a
# robot
# configuration
# into
# the
# corresponding
# end - effector
# pose, whereas ** Inverse
# Kinematics(IK) ** is used
# to
# reconstruct
# the
# configuration(s)
# given
# an
# end - effector
# pose.
#
# In
# the
# simplified
# case
# here
# considered, one
# can
# solve
# the
# problem
# of
# controlling
# the
# end - effector
# 's location to reach a goal position $p^*$ by solving analytically for $q: p(q) = p^*$. However, in the general case, one might not be able to solve this problem analytically, and can typically resort to iterative optimization methods:
#
# $$\min_
# {q \ in  \mathcal
# {Q}} \ | p(q) - p ^ * \ | _2 ^ 2$$
#
# Exact
# analytical
# solutions
# to
# IK
# are
# even
# less
# appealing
# when
# one
# considers
# the
# presence
# of
# obstacles in the
# robot
# 's workspace, resulting in constraints on the possible values of $q$.
#
# > [!TIP]
# > If
# the
# math
# feels
# dense, focus
# on
# the
# mapping: FK
# answers
# "where is the hand given the joints?", IK
# asks
# "what joints reach that hand position?".The
# rest
# of
# the
# unit
# shows
# why
# the
# IK
# direction
# becomes
# hard in realistic
# settings.
#
# ### Differential Inverse Kinematics
#
# When
# IK is hard
# to
# solve
# directly, we
# can
# often
# make
# progress
# by
# working
# with small motions (velocities) instead of absolute positions.
#
# Let $J(q)$ denote
# the
# Jacobian
# matrix
# of(partial)
# derivatives
# of
# the
# FK - function.Then, one
# can
# apply
# the
# chain
# rule
# to
# any $p(q)$, deriving $\dot
# {p} = J(q) \dot
# {q}$, and thus finally relating
# variations in the
# robot
# configurations
# to
# variations in pose.
#
# Given
# a
# desired
# end - effector
# trajectory, differential
# IK
# finds $\dot
# {q}(t)$ solving
# for joints' *velocities* instead of *configurations*:
#
# $$\dot
# {q}(t) = \arg\min_\nu \ | J(q(t)) \nu - \dot
# {p} ^
# *(t)\ | _2 ^ 2$$
#
# This
# often
# admits
# the
# closed - form
# solution $\dot
# {q} = J(q) ^ + \dot
# {p} ^ * $, where $J ^ +(q)$ denotes
# the
# Moore - Penrose
# pseudo - inverse
# of $J(q)$.
#
# ### Adding Feedback Loops
#
# While
# very
# effective
# when
# a
# goal
# trajectory
# has
# been
# well
# specified, the
# performance
# of
# differential
# IK
# can
# degrade
# significantly in the
# presence
# of
# modeling / tracking
# errors, or in the
# presence
# of
# non - modeled
# dynamics in the
# environment.
#
# To
# mitigate
# the
# effect
# of
# modeling
# errors, sensing
# noise and other
# disturbances, classical
# pipelines
# indeed
# do
# augment
# differential
# IK
# with feedback control looping back quantities of interest.In practice, following a trajectory with a closed feedback loop might consist in backwarding the error between the target and measured pose, $\Delta p = p ^ * - p(q)$, hereby modifying the control applied to $\dot{q} = J(q) ^ + (\dot{p} ^ * + k_p \Delta p)$, with $k_p$ defined as the (proportional) gain.
#
# More
# advanced
# techniques
# for control consisting in feedback linearization, PID control, Linear Quadratic Regulator (LQR) or Model-Predictive Control (MPC) can be employed to stabilize tracking and reject moderate perturbations.
#
# ## Limitations of Dynamics-based Robotics
#
# This
# brings
# us
# to
# the “so
# what?”: where
# do
# these
# classical
# tools
# struggle in practice, and why
# does
# that
# motivate
# learning?
#
# Despite
# the
# last
# 60 + years
# of
# robotics
# research, autonomous
# robots
# are
# still
# largely
# incapable
# of
# performing
# tasks
# at
# human - level
# performance in the
# physical
# world
# generalizing
# across(1)
# robot
# embodiments(different
# manipulators, different
# locomotion
# platforms, etc.) and (2)
# tasks(tying
# shoe - laces, manipulating
# a
# diverse
# set
# of
# objects).
#
# Dynamics - based
# approaches
# to
# robotics
# suffer
# from several limitations: (1)
# orchestrating
# multiple
# components
# poses
# integration
# challenges;
# (2)
# the
# need
# to
# develop
# custom
# processing
# pipelines
# for the sensing modalities and tasks considered hinders scalability; (3) simplified analytical models of physical phenomena limit real-world performance.Lastly, (4) dynamics-based methods overlook trends in the availability and growth of robotics data.
#
# ### Key Limitations
#
# ** 1.
# Integration
# Challenges **
# Dynamics - based
# robotics
# pipelines
# have
# historically
# been ** developed
# sequentially, engineering
# the
# different
# blocks ** now
# within
# most
# architectures
# for specific purposes.That is, sensing, state estimation, mapping, planning, (diff-)IK, and low-level control have been traditionally developed as distinct modules with fixed interfaces.Pipelining these specific modules proved error-prone, and brittleness emerges—alongside compounding errors—whenever changes incur.
#
# ** 2.
# Limited
# Scalability **
# Classical
# planners
# operate
# on
# compact, assumed - sufficient
# state
# representations;
# extending
# them
# to
# reason
# directly
# over
# raw, heterogeneous and noisy
# data
# streams is non - trivial.This
# results in a ** limited
# scalability
# to
# multimodal
# data and multitask
# settings **, as incorporating
# high - dimensional
# perceptual
# inputs(RGB, depth, tactile, audio)
# traditionally
# required
# extensive
# engineering
# efforts
# to
# extract
# meaningful
# features
# for control.
#
# ** 3.
# Modeling
# Limitations **
# Setting
# aside
# integration and scalability
# challenges: developing
# accurate
# modeling
# of
# contact, friction, and compliance
# for complicated systems remains difficult.Rigid-body approximations are often insufficient in the presence of deformable objects, and ** relying on approximated models hinders real-world applicability ** of the methods developed.
#
# ** 4.
# Overlooking
# Data
# Trends **
# Lastly, dynamics - based
# methods(naturally)
# overlook
# the
# rather
# recent ** increase in availability
# of
# openly - available
# robotics
# datasets **.The
# curation
# of
# academic
# datasets
# by
# large
# centralized
# groups
# of
# human
# experts in robotics is now
# increasingly
# complemented
# by
# a ** growing
# number
# of
# robotics
# datasets
# contributed in a
# decentralized
# fashion ** by
# individuals
# with varied expertise.
#
# Taken
# together, these
# limitations
# motivate
# the
# exploration
# of
# learning - based
# approaches
# that
# can:
# 1. ** Integrate
# perception and control
# more
# tightly **
# 2. ** Adapt
# across
# tasks and embodiments **
# with reduced expert modeling interventions
# 3. ** Scale
# gracefully in performance ** as more
# robotics
# data
# becomes
# available
#
# ## References
#
# For
# a
# full
# list
# of
# references, check
# out
# the[tutorial](https: // huggingface.co / spaces / lerobot / robot - learning - tutorial).
#
# - ** Modern
# Robotics: Mechanics, Planning, and Control ** (2017)
# Kevin
# M.Lynch and Frank
# C.Park
# A
# comprehensive
# textbook
# covering
# the
# foundations
# of
# classical
# robotics, including
# kinematics, dynamics, and control.Essential
# reading
# for understanding the traditional approaches discussed in this unit.
# [Book Website](http: // hades.mech.northwestern.edu / index.php / Modern_Robotics)
#
# - ** Springer
# Handbook
# of
# Robotics ** (2016)
# Edited
# by
# Bruno
# Siciliano and Oussama
# Khatib
# An
# authoritative
# reference
# covering
# all
# aspects
# of
# robotics,
# from classical control
#
# theory
# to
# emerging
# learning - based
# approaches.
# [DOI: 10.1007 / 978 - 3 - 319 - 32552 - 1](https: // doi.org / 10.1007 / 978-3-319-32552-1)
#
# # Understanding Robot Kinematics
#
# In this section, we'll build intuition for robot kinematics through a concrete, worked example. Kinematics describes the mathematical relationship between joint angles and end-effector positions - given the joint configuration, where does the robot's hand end up? We'll explore this fundamental concept by walking through a simplified but representative case.
#
# Let's examine how traditional robotics approaches robot control using a specific example you can follow step by step.
#
# ## From Complex to Simple: The SO-100 Example
#
# We'll start with a familiar robot platform and systematically simplify it to isolate the core kinematic principles without unnecessary complexity.
#
# The SO-100 arm simplified to a 2D planar manipulator by constraining some joints.
#
# The SO-100 is a 6-degree-of-freedom (6-DOF) robot arm. To understand the principles, let's simplify it to a **2-DOF planar manipulator** by constraining some joints.
#
# ## The Simplified Robot
#
# To keep the math clear, our simplified robot has:
# - **Two joints** with angles θ₁ and θ₂
# - **Two links** of equal length *l*
# - **Configuration** q = [θ₁, θ₂] ∈ [-π, +π]²
#
# ## Forward Kinematics: From Joints to Position
#
# **Question:** Given joint angles θ₁ and θ₂, where is the end-effector?
#
# **Answer:** We can calculate the end-effector position mathematically:
#
# $$p(q) = \begin{pmatrix} l \cos(\theta_1) + l \cos(\theta_1 + \theta_2) \\ l \sin(\theta_1) + l \sin(\theta_1 + \theta_2) \end{pmatrix}$$
#
# This is called **Forward Kinematics (FK)** - mapping from joint space to task space.
#
# > [!TIP]
# > **Understanding the Math:** This equation comes from basic trigonometry!
# > - First link: endpoint at $(l \cos(\theta_1), l \sin(\theta_1))$
# > - Second link: starts from first link's end, rotated by $\theta_1 + \theta_2$
# > - Final position: sum of both link contributions
# >
# > **Why it matters:** For a simple robot, FK is relatively easy - given joint angles, we can always compute where the robot's hand is. More complex robots - for instance, dexterous hands, are more challenging to model.
#
# ## Inverse Kinematics: From Position to Joints
#
# Now let’s try to invert the mapping: given a desired hand position, what joint configuration achieves it?
#
# **Question:** Given a desired end-effector position p*, what joint angles should we use?
#
# **Answer:** This is **Inverse Kinematics (IK)** - typically more intricate to solve, as it deals with the Jacobian matrix of the forward kinematics function!
#
# We need to solve: $p(q) = p^*$
#
# In general, this becomes an optimization problem:
# $$\min_{q \in \mathcal{Q}} \|p(q) - p^*\|_2^2$$
#
# > [!WARNING]
# > **Why IK is Hard:**
# > - **Multiple solutions** - Same end position can be reached with different joint angles
# >
# > - **Nonlinear equations** - Some functions make analytical solutions difficult
# > - **Constraints** - Joint limits and obstacles further complicate the problem
# >
# > This is why robotics engineers spend so much time on IK algorithms!
#
# ## The Challenge of Constraints
#
# *Free to move*
#
# *Constrained by floor*
#
# *Multiple obstacles*
#
# Real robots face **constraints**:
# - **Physical limits** - Can't move through the floor
# - **Obstacles** - Must avoid collisions
# - **Joint limits** - Finite range of motion
#
# These constraints make the feasible configuration space $\mathcal{Q}$ much more complex!
#
# ## Key Insight
#
# Even for this **simple 2-DOF robot**, solving IK with constraints is non-trivial. Real robots have:
# - **6+ degrees of freedom**
# - **Complex geometries**
# - **Dynamic environments**
# - **Uncertain models**
#
# Traditional approaches require **extensive mathematical modeling** and **expert knowledge** for each specific case.
#
# > [!TIP]
# > Mental model: FK is a direct calculator (q → p) and is usually easy; IK is a search (p → q) and becomes hard as soon as you add workspace limits, obstacles, or joint constraints. When IK gets brittle, we'll switch to differential reasoning (velocities) in the next step.
#
# ## References
#
# For a full list of references, check out the [tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial).
#
# - **Modern Robotics: Mechanics, Planning, and Control** (2017)
#   Kevin M. Lynch and Frank C. Park
#   Chapter 4 provides an in-depth treatment of forward kinematics with extensive examples, while Chapter 6 covers inverse kinematics with both analytical and numerical approaches.
#   [Book Website](http://hades.mech.northwestern.edu/index.php/Modern_Robotics)
#
# - **Introduction to Robotics: Mechanics and Control** (2005)
#   John J. Craig
#   A classic textbook with detailed coverage of robot kinematics, including the Denavit-Hartenberg notation and systematic approaches to solving forward and inverse kinematics problems.
#   [Publisher Link](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003519)
#
