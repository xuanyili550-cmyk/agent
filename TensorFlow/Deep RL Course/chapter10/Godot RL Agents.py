"""
================================================================================
 Deep RL Course · Chapter 10 · Godot RL Agents（学习笔记描述）
================================================================================
 一句话：开源 Godot 游戏引擎 + RL，给游戏 NPC 学复杂行为(2D/3D)。
 本章讲：
   ① Godot 引擎与 Python RL 框架(SB3/CleanRL/Ray)的桥接。
   ② 自建自定义环境 + AI 传感器 + 记忆型 agent(LSTM/注意力)。
   ③ pip install godot-rl 上手。
 要点：游戏引擎是低成本造 RL 环境的好途径。
 说明：需 Godot + godot-rl；参考为主。
================================================================================
"""

# # Godot RL Agents
#
# [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) is an Open Source package that allows video game creators, AI researchers, and hobbyists the opportunity **to learn complex behaviors for their Non Player Characters or agents**.
#
# The library provides:
#
# - An interface between games created in the [Godot Engine](https://godotengine.org/) and Machine Learning algorithms running in Python
# - Wrappers for four well known rl frameworks: [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/), [CleanRL](https://docs.cleanrl.dev/), [Sample Factory](https://www.samplefactory.dev/) and [Ray RLLib](https://docs.ray.io/en/latest/rllib-algorithms.html)
# - Support for memory-based agents with LSTM or attention based interfaces
# - Support for *2D and 3D games*
# - A suite of *AI sensors* to augment your agent's capacity to observe the game world
# - Godot and Godot RL Agents are **completely free and open source under a very permissive MIT license**. No strings attached, no royalties, nothing.
#
# You can find out more about Godot RL agents on their [GitHub page](https://github.com/edbeeching/godot_rl_agents) or their AAAI-2022 Workshop [paper](https://arxiv.org/abs/2112.03636). The library's creator, [Ed Beeching](https://edbeeching.github.io/), is a Research Scientist here at Hugging Face.
#
# Installation of the library is simple: `pip install godot-rl`
#
# ## Create a custom RL environment with Godot RL Agents
#
# In this section, you will **learn how to create a custom environment in the Godot Game Engine** and then implement an AI controller that learns to play with Deep Reinforcement Learning.
#
# The example game we create today is simple, **but shows off many of the features of the Godot Engine and the Godot RL Agents library**. You can then dive into the examples for more complex environments and behaviors.
#
# The environment we will be building today is called Ring Pong, the game of pong but the pitch is a ring and the paddle moves around the ring. The **objective is to keep the ball bouncing inside the ring**.
#
# ### Installing the Godot Game Engine
#
# The [Godot game engine](https://godotengine.org/) is an open source tool for the **creation of video games, tools and user interfaces**.
#
# Godot Engine is a feature-packed, cross-platform game engine designed to create 2D and 3D games from a unified interface. It provides a comprehensive set of common tools, so users **can focus on making games without having to reinvent the wheel**. Games can be exported in one click to a number of platforms, including the major desktop platforms (Linux, macOS, Windows) as well as mobile (Android, iOS) and web-based (HTML5) platforms.
#
# While we will guide you through the steps to implement your agent, you may wish to learn more about the Godot Game Engine. Their [documentation](https://docs.godotengine.org/en/latest/index.html) is thorough, and there are many tutorials on YouTube we would also recommend [GDQuest](https://www.gdquest.com/), [KidsCanCode](https://kidscancode.org/godot_recipes/4.x/) and [Bramwell](https://www.youtube.com/channel/UCczi7Aq_dTKrQPF5ZV5J3gg) as sources of information.
#
# In order to create games in Godot, **you must first download the editor**. Godot RL Agents supports the latest version of Godot, Godot 4.0.
#
# Which can be downloaded at the following links:
#
# - [Windows](https://downloads.tuxfamily.org/godotengine/4.0.1/Godot_v4.0.1-stable_win64.exe.zip)
# - [Mac](https://downloads.tuxfamily.org/godotengine/4.0.1/Godot_v4.0.1-stable_macos.universal.zip)
# - [Linux](https://downloads.tuxfamily.org/godotengine/4.0.1/Godot_v4.0.1-stable_linux.x86_64.zip)
#
# ### Loading the starter project
#
# We provide two versions of the codebase:
# - [A starter project, to download and follow along for this tutorial](https://drive.google.com/file/d/1C7xd3TibJHlxFEJPBgBLpksgxrFZ3D8e/view?usp=share_link)
# - [A final version of the project, for comparison and debugging.](https://drive.google.com/file/d/1k-b2Bu7uIA6poApbouX4c3sq98xqogpZ/view?usp=share_link)
#
# To load the project, in the Godot Project Manager click **Import**, navigate to where the files are located and load the **project.godot** file.
#
# If you press F5 or play in the editor, you should be able to play the game in human mode. There are several instances of the game running, this is because we want to speed up training our AI agent with many parallel environments.
#
# ### Installing the Godot RL Agents plugin
#
# The Godot RL Agents plugin can be installed from the Github repo or with the Godot Asset Lib in the editor.
#
# First click on the AssetLib and search for “rl”
#
# Then click on Godot RL Agents, click Download and unselect the LICENSE and README .md files. Then click install.
#
# The Godot RL Agents plugin is now downloaded to your machine. Now click on Project → Project settings and enable the addon:
#
# ### Adding the AI controller
#
# We now want to add an AI controller to our game. Open the player.tscn scene, on the left you should see a hierarchy of nodes that looks like this:
#
# Right click the **Player** node and click **Add Child Node.** There are many nodes listed here, search for AIController3D and create it.
#
# The AI Controller Node should have been added to the scene tree, next to it is a scroll. Click on it to open the script that is attached to the AIController. The Godot game engine uses a scripting language called GDScript, which is syntactically similar to python. The script contains methods that need to be implemented in order to get our AI controller working.
#
# ```python
# #-- Methods that need implementing using the "extend script" option in Godot --#
# func get_obs() -> Dictionary:
# 	assert(false, "the get_obs method is not implemented when extending from ai_controller")
# 	return {"obs":[]}
#
# func get_reward() -> float:
# 	assert(false, "the get_reward method is not implemented when extending from ai_controller")
# 	return 0.0
#
# func get_action_space() -> Dictionary:
# 	assert(false, "the get get_action_space method is not implemented when extending from ai_controller")
# 	return {
# 		"example_actions_continous" : {
# 			"size": 2,
# 			"action_type": "continuous"
# 		},
# 		"example_actions_discrete" : {
# 			"size": 2,
# 			"action_type": "discrete"
# 		},
# 		}
#
# func set_action(action) -> void:
# 	assert(false, "the get set_action method is not implemented when extending from ai_controller")
# # -----------------------------------------------------------------------------#
# ```
#
# In order to implement these methods, we will need to create a class that inherits from AIController3D. This is easy to do in Godot, and is called “extending” a class.
#
# Right click the AIController3D Node and click “Extend Script” and call the new script `controller.gd`. You should now have an almost empty script file that looks like this:
#
# ```python
# extends AIController3D
#
# # Called when the node enters the scene tree for the first time.
# func _ready():
# 	pass # Replace with function body.
#
# # Called every frame. 'delta' is the elapsed time since the previous frame.
# func _process(delta):
# 	pass
# ```
#
# We will now implement the 4 missing methods, delete this code, and replace it with the following:
#
# ```python
# extends AIController3D
#
# # Stores the action sampled for the agent's policy, running in python
# var move_action : float = 0.0
#
# func get_obs() -> Dictionary:
# 	# get the balls position and velocity in the paddle's frame of reference
# 	var ball_pos = to_local(_player.ball.global_position)
# 	var ball_vel = to_local(_player.ball.linear_velocity)
# 	var obs = [ball_pos.x, ball_pos.z, ball_vel.x/10.0, ball_vel.z/10.0]
#
# 	return {"obs":obs}
#
# func get_reward() -> float:
# 	return reward
#
# func get_action_space() -> Dictionary:
# 	return {
# 		"move_action" : {
# 			"size": 1,
# 			"action_type": "continuous"
# 		},
# 		}
#
# func set_action(action) -> void:
# 	move_action = clamp(action["move_action"][0], -1.0, 1.0)
# ```
#
# We have now defined the agent’s observation, which is the position and velocity of the ball in its local coordinate space. We have also defined the action space of the agent, which is a single continuous value ranging from -1 to +1.
#
# The next step is to update the Player’s script to use the actions from the AIController, edit the Player’s script by clicking on the scroll next to the player node, update the code in `Player.gd` to the following:
#
# ```python
# extends Node3D
#
# @export var rotation_speed = 3.0
# @onready var ball = get_node("../Ball")
# @onready var ai_controller = $AIController3D
#
# func _ready():
# 	ai_controller.init(self)
#
# func game_over():
# 	ai_controller.done = true
# 	ai_controller.needs_reset = true
#
# func _physics_process(delta):
# 	if ai_controller.needs_reset:
# 		ai_controller.reset()
# 		ball.reset()
# 		return
#
# 	var movement : float
# 	if ai_controller.heuristic == "human":
# 		movement = Input.get_axis("rotate_anticlockwise", "rotate_clockwise")
# 	else:
# 		movement = ai_controller.move_action
# 	rotate_y(movement*delta*rotation_speed)
#
# func _on_area_3d_body_entered(body):
# 	ai_controller.reward += 1.0
# ```
#
# We now need to synchronize between the game running in Godot and the neural network being trained in Python. Godot RL agents provides a node that does just that. Open the train.tscn scene, right click on the root node, and click “Add child node”. Then, search for “sync” and add a Godot RL Agents Sync node. This node handles the communication between Python and Godot over TCP.
#
# You can run training live in the editor, by first launching the python training with `gdrl`.
#
# In this simple example, a reasonable policy is learned in several minutes. You may wish to speed up training, click on the Sync node in the train scene, and you will see there is a “Speed Up” property exposed in the editor:
#
# Try setting this property up to 8 to speed up training. This can be a great benefit on more complex environments, like the multi-player FPS we will learn about in the next chapter.
#
# ### Export model
#
# [Reference doc](https://github.com/edbeeching/godot_rl_agents/tree/main?tab=readme-ov-file#exporting-and-loading-your-trained-agent-in-onnx-format)
#
# Let's put aside the Godot editor for now. We'll need to use terminals to run some commands in order to save models we trained.
#
# The latest version of the Godot RL library provides experimental support for onnx models with the Stable Baselines 3, rllib, and CleanRL training frameworks.
#
# For example, let's use the Stable Baselines 3 as the framework. Train your agent using the [sb3 example](https://github.com/edbeeching/godot_rl_agents/blob/main/examples/stable_baselines3_example.py) ([instructions for using the script](https://github.com/edbeeching/godot_rl_agents/blob/main/docs/ADV_STABLE_BASELINES_3.md#train-a-model-from-scratch)), enabling the option `--onnx_export_path=model.onnx`
#
# Below is an example command line to execute:
#
# ```bash
# cd  # go into this Godot project directory
# python stable_baselines3_example.py --timesteps=100_000 --onnx_export_path=model.onnx --save_model_path=model.zip --save_checkpoint_frequency=20_000 --experiment_name=exp1
# ```
#
# If things work correctly, you should see messages printed out in the terminal like below:
#
# ```
# No game binary has been provided, please press PLAY in the Godot editor
# waiting for remote GODOT connection on port 11008
# ```
#
# > If you encounter failures about import error in stable_baselines3_example script: "ImportError: cannot import name 'export_model_as_onnx' from 'godot_rl.wrappers.onnx.stable_baselines_export'", follow the answer in [this issue](https://github.com/edbeeching/godot_rl_agents/issues/203) here.
#
# Now it's time to switch back to the Godot editor, and hit PLAY on the top right corner. Once you hit that, the game scene will pop up showing AI training. In the meantime, the terminal will start to print out metrics. Wait for the training to finish, and if things work correctly, you should be able to find the file `model.onnx` in the Godot project directory.
#
# ### Apply AI in the game!
#
# Now let's apply this trained model to the game!
#
# In the Godot editor, find the Sync node in `train.tscn`:
#
# * change the control mode to `Onnx Inference` from the dropdown
# * set `Onnx Model Path` to the model file name, in our case here it's `model.onnx`
#
# To run this game, we need the mono version (i.e., the .NET version) of the Godot editor, you can download it from the Godot official page. We need to install [.NET](https://dotnet.microsoft.com/en-us/download) as well.
#
# Most likely you wil encounter errors in the first attempt. Below are the scenarios to help you resolve the errors.
#
# 1. issue about `Invalid Call. Nonexistent function 'new' in base 'CSharpScript'`: [solution](https://github.com/edbeeching/godot_rl_agents/blob/main/docs/TROUBLESHOOTING.md)
# 2. errors about `onnxruntime` on MacOS: [solution](https://github.com/microsoft/onnxruntime/issues/9707)
#
# ### There’s more!
#
# We have only scratched the surface of what can be achieved with Godot RL Agents, the library includes custom sensors and cameras to enrich the information available to the agent. Take a look at the [examples](https://github.com/edbeeching/godot_rl_agents_examples) to find out more!
#
# For the ability to export the trained model to .onnx so that you can run inference directly from Godot without the Python server, and other useful training options, take a look at the [advanced SB3 tutorial](https://github.com/edbeeching/godot_rl_agents/blob/main/docs/ADV_STABLE_BASELINES_3.md).
#
# ## Author
#
# This section was written by Edward Beeching
#


#Train our robot

# Getting started:
#
# To
# get
# started, download
# the
# project
# from
#
# [here](https: // huggingface.co / ivan267 / imitation - learning - tutorial - godot - project / tree / main) (click
#                                                                                                               on the download icon next to `GDRL-IL-Project.zip`).The
# zip
# file
# features
# both
# the “Starter” and “Complete” projects.
#
# The
# game
# code is already
# implemented in the
# starter
# project and the
# nodes
# are
# configured.We
# will
# focus
# on:
#
# - Implementing
# the
# code
# for the AIController node,
# - Recording
# expert
# demonstrations,
# - Training
# the
# agent and exporting
# an.onnx
# file
# which
# we
# can
# use
# for inference in Godot.
#
# ### Open the starter project in Godot
#
# Extract
# the
# zip
# file, open
# Godot, click “Import” and navigate
# to
# the
# `Starter\Godot
# ` folder
# of
# the
# extracted
# archive.
#
# ### Open the robot scene
#
# You
# can
# search
# for “robot” in the FileSystem search.
#
# This
# scene
# contains
# a
# couple
# of
# different
# nodes, including
# the
# `robot`
# node, which
# contains
# the
# visual
# shape
# of
# the
# robot, `CameraXRotation`
# node
# which is used
# to
# rotate
# the
# camera “up - down” using
# the
# mouse in human
# control
# modes.The
# AI
# agent
# does
# not control
# this
# node
# since
# it is not necessary
# for learning the task.`RaycastSensors` node contains two Raycast sensors that help the agent to “sense” parts of the game world, including walls, floors, etc.
#
# ### Click on the scroll next to AIController3D to open the script for editing
#
# You
# might
# have
# to
# collapse
# the “robot” branch
# to
# find
# it
# more
# easily, or you
# can
# type
# `aicontroller` in the
# Filter
# box
# above
# the
# `Robot`
# node.
#
# ### Replace the `get_obs()` and `get_reward()` methods with the implementation below:
#
# ```python
# func
# get_obs() -> Dictionary:
# var
# observations: Array[float] = []
# for raycast_sensor in raycast_sensors:
#     observations.append_array(raycast_sensor.get_observation())
#
# var
# level_size = 16.0
#
# var
# chest_local = to_local(chest.global_position)
# var
# chest_direction = chest_local.normalized()
# var
# chest_distance = clampf(chest_local.length(), 0.0, level_size)
#
# var
# lever_local = to_local(lever.global_position)
# var
# lever_direction = lever_local.normalized()
# var
# lever_distance = clampf(lever_local.length(), 0.0, level_size)
#
# var
# key_local = to_local(key.global_position)
# var
# key_direction = key_local.normalized()
# var
# key_distance = clampf(key_local.length(), 0.0, level_size)
#
# var
# raft_local = to_local(raft.global_position)
# var
# raft_direction = raft_local.normalized()
# var
# raft_distance = clampf(raft_local.length(), 0.0, level_size)
#
# var
# player_speed = player.global_basis.inverse() * player.velocity.limit_length(5.0) / 5.0
#
# (
#     observations
#     .append_array(
#         [
#             chest_direction.x,
#             chest_direction.y,
#             chest_direction.z,
#             chest_distance,
#             lever_direction.x,
#             lever_direction.y,
#             lever_direction.z,
#             lever_distance,
#             key_direction.x,
#             key_direction.y,
#             key_direction.z,
#             key_distance,
#             raft_direction.x,
#             raft_direction.y,
#             raft_direction.z,
#             raft_distance,
#             raft.movement_direction_multiplier,
#             float(player._is_lever_pulled),
#             float(player._is_chest_opened),
#             float(player._is_key_collected),
#             float(player.is_on_floor()),
#             player_speed.x,
#             player_speed.y,
#             player_speed.z,
#         ]
#     )
# )
# return {"obs": observations}
#
# func
# get_reward() -> float:
# return reward
# ```
#
# In
# `get_obs()`, we
# first
# get
# the
# obs
# from the two
#
# Raycast
# sensors
# added
# to
# the
# `AIController3D`
# node in the
# inspector, and add
# them
# to
# the
# obs, then
# we
# get
# the
# relative
# position
# vectors
# to
# chest, lever, key, and raft, which
# we
# separate
# into
# directions and distances, and then
# we
# add
# them
# to
# the
# obs as well.
#
# We
# also
# add
# other
# game
# state
# info
# to
# the
# obs:
#
# - has
# the
# lever
# has
# been
# pulled,
# - was
# the
# key
# collected,
# - was
# the
# chest
# opened,
# - is the
# player
# on
# floor(also
# determines
# whether
# the
# player
# can
# jump),
# - the
# normalized
# local
# velocity
# of
# the
# player.
#
# We
# convert
# boolean
# values
# such as `_is_lever_pulled`
# to
# floats(0 or 1).
#
# In
# `get_reward()`, we
# only
# need
# to
# return the
# current
# reward.
#
# ### Replace the `_physics_process()` and `reset()` methods with the implementation below:
#
# ```python
# func
# _physics_process(delta: float) -> void:
# # Reset on timeout, this is implemented in parent class to set needs_reset to true,
# # we are re-implementing here to call player.game_over() that handles the game reset.
# n_steps += 1
# if n_steps > reset_after:
#     player.game_over()
#
# # In training or onnx inference modes, this method will be called by sync node with actions provided,
# # For expert demo recording mode, it will be called without any actions (as we set the actions based on human input),
# # For human control mode the method will not be called, so we call it here without any actions provided.
# if control_mode == ControlModes.HUMAN:
#     set_action()
#
# # Reset the game faster if the lever is not pulled.
# steps_without_lever_pulled += 1
# if steps_without_lever_pulled > 200 and (not player._is_lever_pulled):
#     player.game_over()
#
# func
# reset():
# super.reset()
# steps_without_lever_pulled = 0
# ```
#
# ### **Replace the `get_action_space()`, `get_action()`, and `set_action()` methods with the implementation below:**
#
# ```python
# # Defines the actions for the AI agent ("size": 2 means 2 floats for this action)
# func
# get_action_space() -> Dictionary:
# return {
#     "movement": {"size": 2, "action_type": "continuous"},
#     "rotation": {"size": 1, "action_type": "continuous"},
#     "jump": {"size": 1, "action_type": "continuous"},
#     "use_action": {"size": 1, "action_type": "continuous"}
# }
#
# # We return the action values in the same order as defined in get_action_space() (important), but all in one array
# # For actions of size 1, we return 1 float in the array, for size 2, 2 floats in the array, etc.
# # set_action is called just before get_action by the sync node, so we can read the newly set values
# func
# get_action():
# return [
#     # "movement" action values
#     player.requested_movement.x,
#     player.requested_movement.y,
#     # "rotation" action value
#     player.requested_rotation.x,
#     # "jump" action value (-1 if not requested, 1 if requested)
#     -1.0 + 2.0 * float(player.jump_requested),
#     # "use_action" action value (-1 if not requested, 1 if requested)
#     -1.0 + 2.0 * float(player.use_action_requested)
# ]
#
# # Here we set human control and AI control actions to the robot
# func
# set_action(action=null) -> void:
# # If there's no action provided, it means that AI is not controlling the robot (human control),
# if not action:
#     # Only rotate if the mouse has moved since the last set_action call
#     if previous_mouse_movement == mouse_movement:
#         mouse_movement = Vector2.ZERO
#
#     player.requested_movement = Input.get_vector(
#         "move_left", "move_right", "move_forward", "move_back"
#     )
#     player.requested_rotation = mouse_movement
#
#     var
#     use_action = Input.is_action_pressed("requested_action")
#     var
#     jump = Input.is_action_pressed("requested_jump")
#
#     player.use_action_requested = use_action
#     player.jump_requested = jump
#
#     previous_mouse_movement = mouse_movement
# else:
#     # If there is action provided, we set the actions received from the AI agent
#     player.requested_movement = Vector2(action.movement[0], action.movement[1])
#     # The agent only rotates the robot along the Y axis, no need to rotate the camera along X axis
#     player.requested_rotation = Vector2(action.rotation[0], 0.0)
#     player.jump_requested = bool(action.jump[0] > 0)
#     player.use_action_requested = bool(action.use_action[0] > 0)
# ```
#
# For
# `get_action()`(only
# needed if using
# the
# demo
# record
# mode), we
# need
# to
# provide
# the
# actions
# that
# we
# want
# the
# agent
# to
# send
# when
# it
# encounters
# the
# same
# state.It is important
# for the values to be in the correct range (`-1.0 to 1.0`), which is why we have the `-1 + 2 * variable` for boolean states, and in the correct order, as defined in `get_action_space()`.
#
# In
# demo
# record
# mode, `set_action()` is called
# without
# providing
# actions, as we
# need
# to
# set
# the
# action
# values
# based
# on
# human
# input.In
# training / inference
# modes, the
# method is called
# with an `action` argument containing values for all of the actions provided by the RL model, so we have an ` if / else ` to handle both cases.
#
# More
# info is included in the
# code
# comments.
#
# ### Replace the `_input` method with the implementation below:
#
# ```python
# # Record mouse movement for human and demo_record modes
# # We don't directly rotate in input to allow for frame skipping (action_repeat setting) which
# # will also be applied to the AI agent in training/inference modes.
# func
# _input(event):
# if not (heuristic == "human" or heuristic == "demo_record"):
#     return
#
# if event is InputEventMouseMotion:
#     var
#     movement_scale: float = 0.005
#     mouse_movement.y = clampf(event.relative.y * movement_scale, -1.0, 1.0)
#     mouse_movement.x = clampf(event.relative.x * movement_scale, -1.0, 1.0)
# ```
#
# This
# code
# part
# records
# mouse
# movement in case
# of
# human
# control and demo
# record
# modes.
#
# ** Finally, save
# the
# script.We
# are
# ready
# for the next step.**
#
# ### Open the demo record scene, and click on AIController3D node
#
# You
# can
# search
# for “demo” in the FileSystem search, and you can search for “aicontroller” in the scene's filter box.
#
# You
# don’t
# need
# to
# make
# any
# changes as everything is preset, but
# let’s
# go
# over
# the
# things
# you
# would
# need
# to
# set in your
# own
# env:
#
# The
# scene
# contains
# modified
# `Level > Robot > AIController3D`
# node
# settings:
#
# - `Control
# Mode
# ` is set
# to
# `Record
# Expert
# Demos
# `
# - `Expert
# Demo
# Save
# Path
# ` is filled
# out
# - `Action
# Repeat
# ` is set
# to
# the
# same
# value as is set
# for the `Sync` node in `training_scene` and `onnx_inference_scene`.This means that every action set by the agent is repeated for 3 physics frames.The setting in `AIController` adds the same action repeat to the human input (which introduces some lag) to match the same behavior.This is a fairly low value which doesn’t introduce much lag.If you change this value, make sure to change it in all 3 places.
# - `Remove
# Last
# Episode
# ` key
# allows
# us
# to
# set
# a
# key
# that
# can
# be
# used
# to
# remove
# a
# failed
# episode
# during
# recording, without
# having
# to
# restart
# the
# entire
# session.E.g. if the
# robot
# falls in the
# water and the
# game
# resets, we
# can
# use
# this
# key
# to
# remove
# the
# previously
# recorded
# episode
# while recording the next one.It is set to `R`, but you can change it to any key by clicking on it, and then clicking on the `Configure` button.
#
# Another
# way
# to
# make
# episode
# recording
# easier in challenging
# environments is to
# slow
# down
# the
# environment
# during
# recording.This
# can
# easily
# be
# done
# by
# clicking
# on
# the
# `Sync`
# node in the
# scene, and adjusting
# the
# `Speed
# Up
# ` property(set
# to
# 1
# by
# default).
#
# ### Let’s record some demos:
#
# Note
# that
# the
# demos
# will
# only
# be
# saved if we
# have
# recorded
# at
# least
# one
# complete
# episode and closed
# the
# game
# window
# by
# clicking
# on
# "X" or pressing
# ALT + F4.Using
# the
# stop
# button in Godot
# editor
# will
# not save
# the
# demos.It’s
# best
# to
# try recording just one episode first, then check if you see "expert_demos.json" in the filesystem or in the Godot project folder.
#
# Make
# sure
# that
# you
# are
# still in the
# `demo_record_scene`, `press
# F6
# ` and the
# demo
# recording
# will
# start.
#
# Controls:
#
# - mouse
# controls
# the
# camera( if you
# need
# to
# adjust
# mouse
# sensitivity, open
# the
# `robot`
# scene, click
# on
# the
# `Robot`
# node and adjust
# the
# `Rotation
# Speed
# `, keep
# it
# the
# same
# value
# for recording demos, training and inference),
# - `WASD`
# controls
# the
# player
# movement,
# - `SPACE`
# jumps,
# - `E`
# activates
# the
# lever and opens
# the
# chest
#
# You
# can
# take
# a
# few
# practice
# first
# to
# get
# familiar
# with the env.If you wish to skip recording demos, you can also find the pre-recorded demos in the completed project and use the `expert_demos.json` file from there.
#
# The
# recorded
# demos
# should
# include
# at
# least
# 22 - 24
# complete
# successful
# episodes.Multiple
# demo
# files
# can
# also
# be
# used in the
# training
# stage, so
# you
# don’t
# have
# to
# record
# all
# demos in one
# go(you
# can
# change
# the
# file
# name
# using
# the
# `Expert
# Demo
# Save
# Path
# ` property
# mentioned
# before).
#
# Recording
# 23
# episodes
# took
# me
# ~10
# minutes( as the
# key
# has
# 2
# alternating
# spawning
# positions, 22 or 24
# would
# provide
# an
# equal
# distribution
# of
# key
# positions in the
# demos, but
# it is fairly
# close).When
# approaching
# the
# lever or chest, I
# pressed and held
# the
# `E`
# key
# slightly
# longer
# to
# ensure
# the
# action is recorded
# for multiple steps when near those objects.I also removed a couple of episodes that I didn’t complete successfully by pressing the `R` key during the following episode.
#
# Here’s
# a
# sped - up
# video
# of
# the
# demo
# recording
# process:
#
# ### Export the game for training:
#
# You
# can
# export
# the
# game
# from Godot using
#
# `Project > Export`.
#
