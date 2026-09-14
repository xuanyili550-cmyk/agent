from __future__ import annotations

import json
import re
import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        ROOT = _p
        break

import schemas as sch
from agents import (
    ChapterPlannerAgent,
    CharacterAgent,
    ChiefEditorAgent,
    DialogueBatch,
    EpisodeAgent,
    FileConversationStore,
    MockLLMProvider,
    PromptAgent,
    QCOfficerAgent,
    ScreenplayAgent,
    StoryAgent,
    StoryboardAgent,
    build_memory,
)
from planners import EpisodePlanner, SeasonArcPlanner
from workflows.pipeline import AgentBundle, build_graph

NAME_BY_ID = {"char_su_wanwan": "苏晚晚", "char_lu_jingchen": "陆景琛", "char_su_mengyao": "苏梦瑶"}

SCENE_FIXTURES = {
    1: dict(
        location_id="loc_engagement_hall",
        location_name="临江洲际酒店订婚宴会厅",
        time_of_day=sch.TimeOfDay.NIGHT,
        int_ext=sch.IntExt.INT,
        description="苏晚晚重生后提前抵达订婚宴会厅，与苏梦瑶针锋相对，气氛骤然紧张。",
        characters_present=["char_su_wanwan", "char_su_mengyao"],
        emotional_tone="紧张对峙、暗流涌动",
    ),
    2: dict(
        location_id="loc_su_mansion",
        location_name="苏氏公馆卧室",
        time_of_day=sch.TimeOfDay.MORNING,
        int_ext=sch.IntExt.INT,
        description="苏晚晚在公馆确认自己重生的事实，陆景琛的名字突然浮现在她脑海中，成为她计划的关键变量。",
        characters_present=["char_su_wanwan", "char_lu_jingchen"],
        emotional_tone="震惊、逐渐冷静的决绝",
    ),
}


def _story_bible_fixture() -> str:
    world = sch.WorldSetting(
        id="world_001",
        name="临江市豪门圈",
        description="虚构现代都市临江市，以修远集团、苏氏集团为首的多家家族企业盘踞其中。",
        time_period="架空现代都市",
        locations=[
            sch.Location(id="loc_engagement_hall", name="临江洲际酒店订婚宴会厅", description="豪门酒会地标", mood="奢华、压迫"),
            sch.Location(id="loc_su_mansion", name="苏氏公馆", description="苏晚晚从小长大的老宅", mood="表面温情、暗流涌动"),
        ],
        organizations=["修远集团", "苏氏集团"],
        social_rules=["豪门联姻是维系家族利益的常见手段"],
        tone="都市精英、冷峻、强反转",
    )
    bible = sch.StoryBible(
        id="story_001",
        title="重生之豪门逆袭",
        logline="豪门千金苏晚晚被继妹与未婚夫联手构陷惨死，重生回到三年前订婚夜展开复仇。",
        genre=[sch.Genre.REBIRTH, sch.Genre.REVENGE, sch.Genre.CEO_DRAMA],
        themes=["复仇", "重生", "豪门权斗"],
        tone="紧张、强爽感、多重反转",
        target_audience="18-35岁女性，短剧竖屏平台用户",
        main_conflict="苏晚晚与继母、继妹、前未婚夫之间的构陷与复仇对抗",
        unique_selling_point="开局即高能死亡重生+豪门商战+双强携手复仇",
        episode_count_planned=12,
        world=world,
        character_ids=["char_su_wanwan", "char_lu_jingchen", "char_su_mengyao"],
    )
    return bible.model_dump_json()


def _character_fixture(_system: str, user: str) -> str:
    if "女主角" in user:
        char = sch.Character(
            id="char_su_wanwan",
            name="苏晚晚",
            role=sch.CharacterRole.PROTAGONIST,
            gender=sch.Gender.FEMALE,
            age=26,
            occupation="苏氏集团原设计总监",
            appearance="鹅蛋脸，眼神冷静锐利",
            personality_traits=["表面温顺", "内心坚韧"],
            backstory="重生前被继妹与未婚夫联手构陷致死。",
            motivation="阻止悲剧重演，揪出所有幕后黑手",
            relationships=[sch.Relationship(character_id="char_su_mengyao", relation_type=sch.RelationType.RIVAL, description="继妹，最大威胁")],
        )
    elif "男主角" in user:
        char = sch.Character(
            id="char_lu_jingchen",
            name="陆景琛",
            role=sch.CharacterRole.DEUTERAGONIST,
            gender=sch.Gender.MALE,
            age=31,
            occupation="修远集团总裁",
            appearance="身形挺拔，气场冷峻",
            personality_traits=["城府深", "外冷内热"],
            backstory="白手起家的商业强者，厌恶背叛与算计。",
            motivation="被苏晚晚的坚韧打动，决定助她复仇",
            relationships=[sch.Relationship(character_id="char_su_wanwan", relation_type=sch.RelationType.ALLY, description="从合作到心动")],
        )
    else:
        char = sch.Character(
            id="char_su_mengyao",
            name="苏梦瑶",
            role=sch.CharacterRole.ANTAGONIST,
            gender=sch.Gender.FEMALE,
            age=24,
            occupation="苏氏集团市场部经理",
            appearance="甜美精致，笑容恰到好处地无辜",
            personality_traits=["嫉妒心强", "擅长伪装"],
            backstory="自幼觊觎苏晚晚的一切，联合他人策划构陷。",
            motivation="彻底取代苏晚晚，独占苏氏集团继承权",
            relationships=[sch.Relationship(character_id="char_su_wanwan", relation_type=sch.RelationType.RIVAL, description="表面姐妹实为对手")],
        )
    return char.model_dump_json()


EPISODE_FIXTURES = {
    1: dict(
        title="重生夜",
        synopsis="苏晚晚重生回到订婚宴当晚，提前抵达现场试探对手，却意外撞见陆景琛。",
        hook="开场3秒：苏晚晚浑身是血跌落订婚宴会厅台阶，画面骤然倒转回三年前同一晚。",
        cliffhanger="陆景琛忽然逼近质问：'你刚刚说的——我杀了你——是什么意思？'",
        emotional_arc="绝望死亡 → 震惊重生 → 警惕试探 → 悬念对峙",
    ),
    2: dict(
        title="交易",
        synopsis="苏晚晚向陆景琛摊牌'重生'一事，以未来三年的商业情报换取他的结盟，苏梦瑶暗中跟踪。",
        hook="开场3秒：陆景琛把一份并购合同拍在桌上——'证明你说的是真的，否则我现在就报警。'",
        cliffhanger="苏梦瑶在门外举起手机，屏幕上是两人密谈的照片，收件人写着'继母'。",
        emotional_arc="对峙 → 试探 → 结盟 → 被窥视的危机",
    ),
}


def _episode_fixture(_system: str, user: str) -> str:
    m = re.search(r"这是第 (\d+) 集", user)
    number = int(m.group(1)) if m else 1
    f = EPISODE_FIXTURES.get(number, EPISODE_FIXTURES[1])
    episode = sch.Episode(
        id=f"ep_{number:03d}",
        episode_number=number,
        season_id="season_001",
        duration_seconds=300,
        characters_featured=["char_su_wanwan", "char_lu_jingchen", "char_su_mengyao"],
        **f,
    )
    return episode.model_dump_json()


def _script_fixture(_system: str, user: str) -> str:
    m = re.search(r"episode_id='([^']+)'", user)
    episode_id = m.group(1) if m else "ep_001"
    m_v = re.search(r"version 填 (\d+)", user)
    script = sch.Script(
        id=f"script_{episode_id}",
        episode_id=episode_id,
        version=int(m_v.group(1)) if m_v else 1,
        title="重生夜",
        language="zh",
        content=(
            "【场景一：临江洲际酒店订婚宴会厅 · 夜】\n"
            "苏晚晚提前抵达空旷的宴会厅，与前来试探的苏梦瑶针锋相对。\n"
            "苏梦瑶：姐姐今天怎么来得这么早？\n"
            "苏晚晚：总要提前看清楚，谁在这场宴会里笑里藏刀。\n\n"
            "【场景二：苏氏公馆卧室 · 晨】\n"
            "苏晚晚确认自己回到了三年前，脑海中浮现出陆景琛这个关键人物。\n"
            "苏晚晚（自语）：这一次，我不会再让你们轻易得逞。"
        ),
    )
    return script.model_dump_json()


def _scene_fixture(_system: str, user: str) -> str:
    m = re.search(r"scene_number=(\d+)", user)
    scene_num = int(m.group(1))
    m_ep = re.search(r"episode_id='ep_(\d+)'", user)
    ep_num = int(m_ep.group(1)) if m_ep else 1
    f = SCENE_FIXTURES[scene_num]
    scene = sch.Scene(
        id=f"scene_{ep_num:03d}_{scene_num:02d}",
        episode_id=f"ep_{ep_num:03d}",
        scene_number=scene_num,
        location_id=f["location_id"],
        time_of_day=f["time_of_day"],
        int_ext=f["int_ext"],
        description=f["description"],
        characters_present=f["characters_present"],
        emotional_tone=f["emotional_tone"],
        shot_ids=[],
        dialogue=[],
    )
    return scene.model_dump_json()


def _shots_fixture(_system: str, user: str) -> str:
    m = re.search(r"scene_id='([^']+)'", user)
    scene_id = m.group(1)
    _, ep_str, scene_str = scene_id.split("_")
    ep_num, scene_num = int(ep_str), int(scene_str)
    f = SCENE_FIXTURES[scene_num]
    shots = []
    for i in (1, 2):
        shots.append(
            sch.Shot(
                id=f"shot_{ep_num:03d}_{scene_num:02d}_{i:02d}",
                episode=ep_num,
                scene=scene_num,
                shot=i,
                scene_id=scene_id,
                character=f["characters_present"][:1] if i == 1 else f["characters_present"],
                location=f["location_name"],
                action=f"第{scene_num}场第{i}个镜头：{f['emotional_tone']}的关键演出",
                emotion=sch.Emotion.DETERMINATION if i == 1 else sch.Emotion.SHOCK,
                camera=sch.CameraSpec(
                    shot_size=sch.ShotSize.CLOSE_UP if i == 1 else sch.ShotSize.MEDIUM_SHOT,
                    angle=sch.CameraAngle.EYE_LEVEL,
                    movement=sch.CameraMovement.DOLLY_IN if i == 1 else sch.CameraMovement.STATIC,
                ),
                duration=2.5,
            )
        )
    shots_file = sch.ShotsFile(shots=shots)
    return shots_file.model_dump_json()


def _dialogue_fixture(_system: str, user: str) -> str:
    shot_ids = re.findall(r"shot_\d{3}_\d{2}_\d{2}", user)
    chars_match = re.search(r"出场角色 id：([^\n]+)", user)
    chars = chars_match.group(1).split("、") if chars_match else ["char_su_wanwan"]
    lines: list[sch.DialogueLine] = []
    canned = [
        ("这一次，我不会再心软。", "This time, I won't hold back.", sch.Emotion.DETERMINATION),
        ("你以为你还能像上辈子一样赢吗？", "Do you think you can win like last time?", sch.Emotion.CONTEMPT),
    ]
    for idx, shot_id in enumerate(shot_ids[:2]):
        char_id = chars[idx % len(chars)]
        line_zh, line_en, emotion = canned[idx % len(canned)]
        lines.append(
            sch.DialogueLine(
                id=f"dlg_{shot_id}",
                shot_id=shot_id,
                character_id=char_id,
                character_name=NAME_BY_ID.get(char_id, char_id),
                order=idx + 1,
                line_zh=line_zh,
                line_en=line_en,
                emotion=emotion,
            )
        )
    return DialogueBatch(dialogue=lines).model_dump_json()


def _image_prompt_fixture(_system: str, user: str) -> str:
    m = re.search(r"shot_id='([^']+)'", user)
    shot_id = m.group(1)
    prompt = sch.ImagePrompt(
        id=f"img_{shot_id}",
        shot_id=shot_id,
        prompt_text=f"电影感竖屏短剧画面，对应镜头 {shot_id}，情绪张力拉满，9:16 画幅，写实摄影质感",
        negative_prompt="低分辨率, 变形肢体, 多余手指, 水印",
        style_tags=["cinematic", "vertical 9:16", "photorealistic"],
        aspect_ratio="9:16",
        reference_character_ids=[],
    )
    return prompt.model_dump_json()


def _video_prompt_fixture(_system: str, user: str) -> str:
    m_shot = re.search(r"shot_id='([^']+)'", user)
    m_img = re.search(r"image_prompt_id 填 '([^']+)'", user)
    shot_id = m_shot.group(1)
    image_prompt_id = m_img.group(1) if m_img else f"img_{shot_id}"
    prompt = sch.VideoPrompt(
        id=f"vid_{shot_id}",
        shot_id=shot_id,
        image_prompt_id=image_prompt_id,
        prompt_text=f"基于关键帧生成短剧镜头动画，对应 {shot_id}，运镜自然，情绪连贯，符合竖屏短剧节奏",
        motion_description="缓慢推进运镜，强化情绪张力",
        camera_movement=sch.CameraMovement.DOLLY_IN,
        duration=2.5,
        fps=24,
        model_target="generic",
        negative_prompt="画面抖动过度, 人物崩坏, 闪烁伪影",
    )
    return prompt.model_dump_json()


def _qc_verdict_fixture(_system: str, user: str) -> str:
    m = re.search(r"id 使用 '([^']+)'，episode_id 为 '([^']+)'", user)
    verdict = sch.QCVerdict(
        id=m.group(1) if m else "qc_ep_001",
        episode_id=m.group(2) if m else "ep_001",
        passed=True,
        score=86,
        issues=[
            sch.QCIssue(
                code="pacing",
                severity=sch.IssueSeverity.LOW,
                location="scene_001_02",
                description="第二场情绪转折略快",
                suggestion="可在苏晚晚自语前加一个 1.5 秒的沉默特写",
            )
        ],
        summary="钩子与悬念达标，人物一致，引用闭合；仅节奏有轻微可优化点。",
    )
    return verdict.model_dump_json()


def _editorial_fixture(_system: str, user: str) -> str:
    m = re.search(r"id 使用 '([^']+)'，episode_id 为 '([^']+)'", user)
    decision = sch.EditorialDecision(
        id=m.group(1) if m else "ed_ep_001",
        episode_id=m.group(2) if m else "ep_001",
        decision=sch.EditorialDecisionType.APPROVE,
        notes="开场 3 秒的坠落画面冲击力足够，结尾质问直接抛出下一集悬念，可以投产。",
        required_changes=[],
        commercial_score=82,
        audience_fit="契合 18-35 岁女性观众对重生复仇题材的爽感预期",
    )
    return decision.model_dump_json()


def _season_arc_fixture(_system: str, user: str) -> str:
    # 章节规划师（LLM 版）的 mock：返回和规则版一致的分幕，只是文案更具体
    arc = SeasonArcPlanner(num_episodes=12).plan(sch.StoryBible.model_validate_json(_story_bible_fixture()))
    beats = [
        b.model_copy(update={"description": f"{b.act}：{['苏晚晚重生并借订婚宴试探敌人', '结盟陆景琛，苏梦瑶与继母反扑', '揭开三年前死亡真相，家族清算'][i]}"})
        for i, b in enumerate(arc.beats)
    ]
    return arc.model_copy(update={"beats": beats}).model_dump_json()


def build_mock_provider() -> MockLLMProvider:
    return MockLLMProvider(
        {
            "StoryBible": _story_bible_fixture(),
            "SeasonArc": _season_arc_fixture,
            "Character": _character_fixture,
            "Episode": _episode_fixture,
            "Script": _script_fixture,
            "Scene": _scene_fixture,
            "ShotsFile": _shots_fixture,
            "DialogueBatch": _dialogue_fixture,
            "ImagePrompt": _image_prompt_fixture,
            "VideoPrompt": _video_prompt_fixture,
            "QCVerdict": _qc_verdict_fixture,
            "EditorialDecision": _editorial_fixture,
        }
    )


def main() -> None:
    provider = build_mock_provider()
    out_dir = ROOT / "02_STORY_ENGINE" / "demo_output"
    out_dir.mkdir(exist_ok=True)

    # 整条流水线共用一个会话记忆：六个 agent 的每一次成功调用都追加进同一份历史，
    # 历史落盘在 demo_output/conversations/<context_id>.json，重跑 demo 会接着上次继续。
    memory = build_memory(provider, context_id="drama_demo_story_001", store=FileConversationStore(out_dir / "conversations"))
    print(f"会话 {memory.context_id}：加载到 {memory.turn_count} 轮历史，上下文上限 {memory.max_context_tokens} tokens")

    bundle = AgentBundle(
        story_agent=StoryAgent(provider, memory=memory),
        character_agent=CharacterAgent(provider, memory=memory),
        episode_agent=EpisodeAgent(provider, memory=memory),
        screenplay_agent=ScreenplayAgent(provider, memory=memory),
        storyboard_agent=StoryboardAgent(provider, memory=memory),
        prompt_agent=PromptAgent(provider, memory=memory),
        season_planner=ChapterPlannerAgent(provider, memory=memory, num_episodes=12),
        episode_planner=EpisodePlanner(),
        qc_officer=QCOfficerAgent(provider, memory=memory),
        chief_editor=ChiefEditorAgent(provider, memory=memory),
    )
    # 连续生成两集：第 2 集的 EpisodeAgent 会拿到第 1 集的"前情提要"
    graph = build_graph(bundle, num_characters=3, num_scenes=2, episode_numbers=[1, 2])
    result = graph.invoke({"idea": "豪门千金重生复仇，携手冷峻总裁夺回家族企业", "target_duration_seconds": 300})

    print("=== Story Bible ===")
    print(result["story_bible"].title, "-", result["story_bible"].logline)
    print("\n=== Season Arc ===")
    print(result["season_arc"].title, "| beats:", [b.act for b in result["season_arc"].beats])
    print("\n=== Characters ===")
    for c in result["characters"]:
        print(f"- {c.name} ({c.role.value})")
    print("\n=== Episodes ===")
    for ep in result["episodes"]:
        print(f"第{ep.episode_number}集《{ep.title}》| hook: {ep.hook}")
        print(f"  cliffhanger: {ep.cliffhanger}")
    print("\n=== Editorial (质检官 + 总编审) ===")
    for ep_id, rec in result["editorial_by_episode"].items():
        print(f"{ep_id}: status={rec['status']} qc_score={rec['qc']['score']} decision={rec['editorial']['decision']} rounds={rec['revision_rounds']}")
    ep = result["episode"]
    print("\n=== Script (first 120 chars) ===")
    print(result["script"].content[:120].replace("\n", " "))
    print("\n=== Scenes / Shots / Dialogue ===")
    total_dialogue = sum(len(s.dialogue) for s in result["scenes"])
    print(f"scenes={len(result['scenes'])} shots={len(result['shots'])} dialogue_lines={total_dialogue}")
    print("\n=== Prompts ===")
    print(f"image_prompts={len(result['image_prompts'])} video_prompts={len(result['video_prompts'])}")
    print("\n=== Conversation Memory ===")
    print(f"本次 LLM 调用 {len(provider.calls)} 次，历史累计 {memory.turn_count} 轮，最后一次调用携带 {provider.calls[-1]['history_len']} 条历史消息")

    story_file = sch.StoryFile(story_bible=result["story_bible"], season_arcs=[result["season_arc"]])
    characters_file = sch.CharactersFile(characters=result["characters"])
    episodes_file = sch.EpisodesFile(episodes=result["episodes"])
    scenes_file = sch.ScenesFile(scenes=result["scenes"])
    shots_file = sch.ShotsFile(shots=result["shots"])
    prompts_file = sch.PromptsFile(image_prompts=result["image_prompts"], video_prompts=result["video_prompts"])

    for name, model in [
        ("story.json", story_file),
        ("characters.json", characters_file),
        ("episodes.json", episodes_file),
        ("scenes.json", scenes_file),
        ("shots.json", shots_file),
        ("prompts.json", prompts_file),
    ]:
        (out_dir / name).write_text(json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n结构化数据已写入: {out_dir}")


if __name__ == "__main__":
    main()
