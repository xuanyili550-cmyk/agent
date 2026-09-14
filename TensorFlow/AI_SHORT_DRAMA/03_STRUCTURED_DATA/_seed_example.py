from __future__ import annotations

import json
from pathlib import Path

from schemas import (
    CameraAngle,
    CameraMovement,
    CameraSpec,
    Character,
    CharacterRole,
    CharactersFile,
    DialogueLine,
    Emotion,
    Episode,
    EpisodesFile,
    Gender,
    Genre,
    ImagePrompt,
    IntExt,
    Location,
    PromptsFile,
    Relationship,
    RelationType,
    Scene,
    ScenesFile,
    SeasonArc,
    Shot,
    ShotsFile,
    ShotSize,
    StoryBeat,
    StoryBible,
    StoryFile,
    TimeOfDay,
    VideoPrompt,
    WorldSetting,
)

HERE = Path(__file__).parent

world = WorldSetting(
    id="world_001",
    name="临江市豪门圈",
    description="虚构现代都市临江市，以修远集团、苏氏集团为首的多家家族企业盘踞其中，联姻、股权与舆论是权力更迭的三大战场。",
    time_period="架空现代都市",
    locations=[
        Location(
            id="loc_engagement_hall",
            name="临江洲际酒店订婚宴会厅",
            description="全市顶级豪门举办婚庆与商务酒会的地标场地，水晶吊灯与落地窗环绕",
            mood="奢华、压迫、众目睽睽",
        ),
        Location(id="loc_su_mansion", name="苏氏公馆", description="苏晚晚从小长大的老宅，欧式庭院配中式回廊", mood="表面温情、暗流涌动"),
        Location(id="loc_xingyuan_tower", name="修远集团总部大厦", description="陆景琛的商业帝国总部，顶层为其私人办公室", mood="冷峻、权力感"),
        Location(id="loc_hall_balcony", name="宴会厅露台", description="宴会厅外的露台，可俯瞰临江夜景", mood="私密、紧张对峙"),
    ],
    organizations=["修远集团", "苏氏集团", "陈氏地产"],
    social_rules=["豪门联姻是维系家族利益的常见手段", "舆论与股权投票权同样能左右一个人的命运"],
    tone="都市精英、冷峻、强反转",
)

story_bible = StoryBible(
    id="story_001",
    title="重生之豪门逆袭",
    logline="豪门千金苏晚晚被继母与未婚夫联手构陷、含冤惨死，一朝重生回到三年前订婚夜，这一次她要让所有伤害过她的人血债血偿。",
    genre=[Genre.REBIRTH, Genre.REVENGE, Genre.CEO_DRAMA],
    themes=["复仇", "重生", "豪门权斗", "逆风翻盘"],
    tone="紧张、强爽感、多重反转",
    target_audience="18-35岁女性，短剧竖屏平台用户",
    main_conflict="苏晚晚与继母王丽芬、继妹苏梦瑶、前未婚夫陈皓之间的构陷与复仇对抗",
    unique_selling_point="开局即高能死亡重生+豪门商战+双强携手复仇，每集三秒强钩子、结尾强悬念",
    episode_count_planned=12,
    world=world,
    character_ids=[
        "char_su_wanwan",
        "char_lu_jingchen",
        "char_su_mengyao",
        "char_chen_hao",
        "char_wang_lifen",
        "char_lin_xi",
    ],
)

season_arc = SeasonArc(
    id="season_001",
    season_number=1,
    title="血债血偿",
    synopsis="苏晚晚重生回到订婚夜前夕，携手修远集团总裁陆景琛，从舆论、股权、亲情三条战线逐步瓦解继母与继妹的构陷网络，最终夺回苏氏集团并揭穿三年前的真相。",
    central_question="苏晚晚能否在重蹈覆辙之前识破所有构陷，并让加害者付出代价？",
    resolution="苏晚晚联合陆景琛揭穿王丽芬与陈皓伪造遗嘱、构陷杀人的证据，夺回苏氏集团控制权，苏梦瑶与陈皓身败名裂。",
    beats=[
        StoryBeat(act="第一幕：重生与结盟", description="苏晚晚重生并与陆景琛因误会结识，识破当晚的第一场构陷", episode_ids=["ep_001", "ep_002", "ep_003"]),
        StoryBeat(
            act="第二幕：商战反击",
            description="双方在股权与舆论战场多次交手，苏晚晚步步夺回话语权",
            episode_ids=["ep_004", "ep_005", "ep_006", "ep_007", "ep_008"],
        ),
        StoryBeat(
            act="第三幕：真相与清算",
            description="尘封的死亡真相浮出水面，苏晚晚完成复仇并与陆景琛确认心意",
            episode_ids=["ep_009", "ep_010", "ep_011", "ep_012"],
        ),
    ],
    episode_ids=["ep_001", "ep_002", "ep_003", "ep_004", "ep_005", "ep_006", "ep_007", "ep_008", "ep_009", "ep_010", "ep_011", "ep_012"],
)

characters = [
    Character(
        id="char_su_wanwan",
        name="苏晚晚",
        aliases=["苏大小姐"],
        role=CharacterRole.PROTAGONIST,
        gender=Gender.FEMALE,
        age=26,
        occupation="苏氏集团原设计总监（重生前）",
        appearance="鹅蛋脸，眼尾微挑，常年一身利落职业装，重生后眼神比从前更冷静锐利",
        personality_traits=["表面温顺", "内心坚韧", "极强的观察力", "重情但不再天真"],
        backstory="苏氏集团嫡女，生母早逝，父亲续弦后处处忍让继母与继妹，订婚当晚被继妹与未婚夫联手构陷侵吞公款，众叛亲离后坠楼身亡。",
        motivation="阻止悲剧重演，揪出所有幕后黑手，夺回本该属于自己的一切",
        relationships=[
            Relationship(character_id="char_su_mengyao", relation_type=RelationType.RIVAL, description="名义上的继妹，实为夺权的最大威胁"),
            Relationship(character_id="char_chen_hao", relation_type=RelationType.ENEMY, description="前未婚夫，重生后果断划清界限"),
            Relationship(character_id="char_lu_jingchen", relation_type=RelationType.ALLY, description="重生夜意外结识，后成为复仇路上最重要的盟友"),
            Relationship(character_id="char_wang_lifen", relation_type=RelationType.FAMILY, description="继母，表面慈爱实则主谋之一"),
        ],
        arc_summary="从隐忍受害者成长为冷静掌控全局的复仇者与商业强者",
        voice_style="平静克制，情绪爆发时字字清晰、不歇斯底里",
    ),
    Character(
        id="char_lu_jingchen",
        name="陆景琛",
        aliases=["陆总"],
        role=CharacterRole.DEUTERAGONIST,
        gender=Gender.MALE,
        age=31,
        occupation="修远集团总裁",
        appearance="身形挺拔，常年一身深色西装，气场冷峻，笑容极少",
        personality_traits=["城府深", "重规则", "外冷内热", "护短"],
        backstory="白手起家接手家族企业并将其扩张为临江第一集团，因幼年家变而极度厌恶背叛与算计。",
        motivation="起初只是被苏晚晚的反常举动引起商业警觉，逐渐被她的坚韧打动，决定助她复仇",
        relationships=[
            Relationship(character_id="char_su_wanwan", relation_type=RelationType.ALLY, description="从合作到心动的复杂关系"),
            Relationship(character_id="char_lin_xi", relation_type=RelationType.EMPLOYEE, description="陆景琛最信任的私人秘书"),
        ],
        arc_summary="从疏离克制到主动敞开心扉，成为苏晚晚复仇路上的坚实后盾",
        voice_style="低沉简练，惯用反问句，情绪外露极少",
    ),
    Character(
        id="char_su_mengyao",
        name="苏梦瑶",
        aliases=["梦瑶"],
        role=CharacterRole.ANTAGONIST,
        gender=Gender.FEMALE,
        age=24,
        occupation="苏氏集团市场部经理（挂名）",
        appearance="甜美精致的妆容，笑容总是恰到好处地无辜",
        personality_traits=["嫉妒心强", "擅长伪装", "手段狠辣"],
        backstory="王丽芬带来的继女，自幼觊觎苏晚晚的一切，联合母亲与陈皓策划构陷苏晚晚侵吞公款并致其坠楼。",
        motivation="彻底取代苏晚晚，独占苏氏集团继承权与陈皓",
        relationships=[
            Relationship(character_id="char_su_wanwan", relation_type=RelationType.RIVAL, description="表面姐妹实为夺权对手"),
            Relationship(character_id="char_wang_lifen", relation_type=RelationType.FAMILY, description="母女同谋"),
            Relationship(character_id="char_chen_hao", relation_type=RelationType.ROMANTIC, description="暗中勾结的地下恋人"),
        ],
        arc_summary="从伪装完美到步步露馅，最终身败名裂",
        voice_style="声音甜腻，情绪激动时会破音露出真实的尖利",
    ),
    Character(
        id="char_chen_hao",
        name="陈皓",
        aliases=[],
        role=CharacterRole.ANTAGONIST,
        gender=Gender.MALE,
        age=28,
        occupation="陈氏地产少东家",
        appearance="斯文清秀，笑容温和具有欺骗性",
        personality_traits=["软弱", "贪婪", "易受操纵"],
        backstory="苏晚晚名义上的未婚夫，被苏梦瑶以感情与利益双重诱惑拉拢，参与构陷苏晚晚侵吞公款。",
        motivation="借苏家与陈家联姻巩固地产生意，同时贪恋苏梦瑶的顺从与美貌",
        relationships=[
            Relationship(character_id="char_su_wanwan", relation_type=RelationType.ENEMY, description="曾经的未婚夫，重生后被彻底看清"),
            Relationship(character_id="char_su_mengyao", relation_type=RelationType.ROMANTIC, description="地下恋情，实际上被其利用"),
        ],
        arc_summary="从伪装深情到彻底暴露懦弱与贪婪的本质",
        voice_style="语气温吞，心虚时习惯性重复对方的话",
    ),
    Character(
        id="char_wang_lifen",
        name="王丽芬",
        aliases=["王夫人"],
        role=CharacterRole.ANTAGONIST,
        gender=Gender.FEMALE,
        age=52,
        occupation="苏氏集团董事（继母）",
        appearance="端庄雍容，说话总是慢条斯理",
        personality_traits=["城府极深", "表面慈爱", "冷酷算计"],
        backstory="苏晚晚的继母，长年伪造账目与遗嘱证据，企图将苏氏集团彻底转移到亲生女儿苏梦瑶名下。",
        motivation="确保苏梦瑶继承苏氏集团全部资产，掩盖多年的财务造假",
        relationships=[
            Relationship(character_id="char_su_wanwan", relation_type=RelationType.FAMILY, description="表面继母实为构陷主谋"),
            Relationship(character_id="char_su_mengyao", relation_type=RelationType.FAMILY, description="亲生母女"),
        ],
        arc_summary="从幕后操盘手到证据链曝光后彻底崩溃",
        voice_style="语速缓慢，用词客气但暗藏威胁",
    ),
    Character(
        id="char_lin_xi",
        name="林溪",
        aliases=["林秘书"],
        role=CharacterRole.SUPPORTING,
        gender=Gender.FEMALE,
        age=27,
        occupation="陆景琛私人秘书",
        appearance="干练短发，职业套装，永远带着平板电脑",
        personality_traits=["细心", "忠诚", "反应快"],
        backstory="跟随陆景琛多年的得力助手，是他调查苏氏集团内幕的重要执行者。",
        motivation="协助陆景琛完成商业布局，同时逐渐认同苏晚晚的正直",
        relationships=[
            Relationship(character_id="char_lu_jingchen", relation_type=RelationType.EMPLOYER, description="多年上下级，高度信任"),
            Relationship(character_id="char_su_wanwan", relation_type=RelationType.ALLY, description="从旁观者变为暗中支持者"),
        ],
        arc_summary="从职业化的旁观者成长为坚定的同盟",
        voice_style="语速快，条理清晰，习惯用数据说话",
    ),
]

episode_1 = Episode(
    id="ep_001",
    episode_number=1,
    season_id="season_001",
    title="重生夜",
    synopsis="苏晚晚在订婚宴会上被继妹与未婚夫联手构陷坠楼惨死，灵魂重生回到三年前同一个夜晚，她提前抵达宴会厅，却意外撞见了陆景琛，说漏了一句不该说的话。",
    hook="开场3秒：苏晚晚浑身是血从订婚宴会厅台阶跌落，眼睁睁看着陈皓与苏梦瑶在众人面前联手羞辱她致死——画面骤然倒转，她在三年前同一晚睁开双眼。",
    cliffhanger="陆景琛忽然出现在她面前，眼神骤冷：'苏小姐，你我素不相识，你刚刚说的那句——我杀了你——是什么意思？'",
    duration_seconds=420,
    scene_ids=["scene_001", "scene_002", "scene_003", "scene_004"],
    characters_featured=["char_su_wanwan", "char_lu_jingchen", "char_su_mengyao", "char_chen_hao", "char_wang_lifen"],
    emotional_arc="绝望死亡 → 震惊重生 → 警惕试探 → 悬念对峙",
)

episodes = [episode_1]

scenes: list[Scene] = []
shots: list[Shot] = []
image_prompts: list[ImagePrompt] = []
video_prompts: list[VideoPrompt] = []


def add_shot(scene_id, ep, sc, sh_no, character, location, action, emotion, camera, duration, dialogue_ids=None, notes=""):
    sh = Shot(
        id=f"shot_{ep:03d}_{sc:02d}_{sh_no:02d}",
        episode=ep,
        scene=sc,
        shot=sh_no,
        scene_id=scene_id,
        character=character,
        location=location,
        action=action,
        emotion=emotion,
        camera=camera,
        duration=duration,
        dialogue_ids=dialogue_ids or [],
        notes=notes,
    )
    shots.append(sh)
    return sh


# Scene 1: 订婚宴会厅 - 惨死高潮（重生前，作为强钩子序幕）
scene1_dialogue = [
    DialogueLine(
        id="dlg_001_01",
        shot_id="shot_001_01_02",
        character_id="char_su_mengyao",
        character_name="苏梦瑶",
        order=1,
        line_zh="姐姐，你贪污公款的证据都在这里了，你还有什么好狡辩的？",
        line_en="Sister, here's the proof you embezzled company funds. What excuse do you have left?",
        emotion=Emotion.CONTEMPT,
        voice_direction="表面痛心，实则得意，尾音带着一丝颤抖的表演感",
    ),
    DialogueLine(
        id="dlg_001_02",
        shot_id="shot_001_01_03",
        character_id="char_su_wanwan",
        character_name="苏晚晚",
        order=2,
        line_zh="陈皓，你也相信这些？我们订婚才三个月。",
        line_en="Chen Hao, you believe this too? We've only been engaged three months.",
        emotion=Emotion.DESPAIR,
        voice_direction="声音发抖，带着最后一丝希望",
    ),
    DialogueLine(
        id="dlg_001_03",
        shot_id="shot_001_01_04",
        character_id="char_chen_hao",
        character_name="陈皓",
        order=3,
        line_zh="证据都摆在眼前了，晚晚，你还是自己收拾行李离开苏家吧。",
        line_en="The evidence speaks for itself, Wanwan. Just pack your things and leave the Su family.",
        emotion=Emotion.CONTEMPT,
        voice_direction="刻意维持温和语气，眼神却毫无温度",
    ),
]

scene1 = Scene(
    id="scene_001",
    episode_id="ep_001",
    scene_number=1,
    location_id="loc_engagement_hall",
    time_of_day=TimeOfDay.NIGHT,
    int_ext=IntExt.INT,
    description="订婚宴会高潮，苏梦瑶当众出示伪造的贪污证据，陈皓与王丽芬联手逼迫苏晚晚认罪，混乱中苏晚晚被推下台阶身亡。",
    characters_present=["char_su_wanwan", "char_su_mengyao", "char_chen_hao", "char_wang_lifen"],
    emotional_tone="绝望、羞辱、致命反转",
    shot_ids=["shot_001_01_01", "shot_001_01_02", "shot_001_01_03", "shot_001_01_04", "shot_001_01_05"],
    dialogue=scene1_dialogue,
)

add_shot(
    "scene_001",
    1,
    1,
    1,
    ["char_su_wanwan"],
    "临江洲际酒店订婚宴会厅",
    "苏晚晚身穿染血礼服跌坐在宴会厅台阶下，四周宾客窃窃私语后退",
    Emotion.DESPAIR,
    CameraSpec(shot_size=ShotSize.CLOSE_UP, angle=CameraAngle.HIGH_ANGLE, movement=CameraMovement.STATIC),
    2.0,
    notes="全剧开场第一个镜头，必须在3秒内建立强钩子",
)
add_shot(
    "scene_001",
    1,
    1,
    2,
    ["char_su_mengyao"],
    "临江洲际酒店订婚宴会厅",
    "苏梦瑶举着文件当众宣读，泪光中带着算计的笑意",
    Emotion.CONTEMPT,
    CameraSpec(shot_size=ShotSize.MEDIUM_CLOSE_UP, angle=CameraAngle.LOW_ANGLE, movement=CameraMovement.DOLLY_IN),
    3.5,
    dialogue_ids=["dlg_001_01"],
)
add_shot(
    "scene_001",
    1,
    1,
    3,
    ["char_su_wanwan"],
    "临江洲际酒店订婚宴会厅",
    "苏晚晚踉跄起身望向陈皓，眼神从愤怒转为绝望",
    Emotion.DESPAIR,
    CameraSpec(shot_size=ShotSize.CLOSE_UP, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.HANDHELD),
    3.0,
    dialogue_ids=["dlg_001_02"],
)
add_shot(
    "scene_001",
    1,
    1,
    4,
    ["char_chen_hao", "char_su_wanwan"],
    "临江洲际酒店订婚宴会厅",
    "陈皓别过脸，语气冷漠地下达逐客令，宾客哗然",
    Emotion.CONTEMPT,
    CameraSpec(shot_size=ShotSize.TWO_SHOT, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.STATIC),
    3.5,
    dialogue_ids=["dlg_001_03"],
)
add_shot(
    "scene_001",
    1,
    1,
    5,
    ["char_su_wanwan"],
    "临江洲际酒店订婚宴会厅",
    "混乱中苏晚晚被推搡跌下台阶，画面猛然定格再骤然反转为黑屏",
    Emotion.SHOCK,
    CameraSpec(shot_size=ShotSize.EXTREME_LONG_SHOT, angle=CameraAngle.BIRDS_EYE, movement=CameraMovement.CRANE),
    2.5,
    notes="接黑屏转场，进入重生瞬间",
)

# Scene 2: 重生瞬间 - 苏氏公馆卧室
scene2_dialogue = [
    DialogueLine(
        id="dlg_002_01",
        shot_id="shot_001_02_02",
        character_id="char_su_wanwan",
        character_name="苏晚晚",
        order=1,
        line_zh="三年前……订婚宴当天？我竟然……重生了。",
        line_en="Three years ago... the engagement day? I've actually... been reborn.",
        emotion=Emotion.SURPRISE,
        voice_direction="气息急促，语速由快转慢，带着难以置信的颤抖",
    ),
]

scene2 = Scene(
    id="scene_002",
    episode_id="ep_001",
    scene_number=2,
    location_id="loc_su_mansion",
    time_of_day=TimeOfDay.MORNING,
    int_ext=IntExt.INT,
    description="苏晚晚在苏氏公馆卧室惊醒，发现自己回到了三年前订婚宴当天的清晨，桌上台历与手机日期印证了重生的事实。",
    characters_present=["char_su_wanwan"],
    emotional_tone="震惊、混乱、逐渐清醒的决绝",
    shot_ids=["shot_001_02_01", "shot_001_02_02", "shot_001_02_03"],
    dialogue=scene2_dialogue,
)

add_shot(
    "scene_002",
    1,
    2,
    1,
    ["char_su_wanwan"],
    "苏氏公馆卧室",
    "苏晚晚猛然从床上坐起，大口喘息，双手下意识摸向自己完好无损的身体",
    Emotion.SHOCK,
    CameraSpec(shot_size=ShotSize.MEDIUM_SHOT, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.HANDHELD),
    2.5,
)
add_shot(
    "scene_002",
    1,
    2,
    2,
    ["char_su_wanwan"],
    "苏氏公馆卧室",
    "特写手机锁屏日期定格在三年前订婚宴当天，苏晚晚瞳孔骤缩",
    Emotion.SURPRISE,
    CameraSpec(shot_size=ShotSize.EXTREME_CLOSE_UP, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.ZOOM_IN),
    2.0,
    dialogue_ids=["dlg_002_01"],
)
add_shot(
    "scene_002",
    1,
    2,
    3,
    ["char_su_wanwan"],
    "苏氏公馆卧室",
    "苏晚晚起身走向梳妆镜，眼神从慌乱迅速转为冷静与杀意",
    Emotion.DETERMINATION,
    CameraSpec(shot_size=ShotSize.MEDIUM_CLOSE_UP, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.DOLLY_IN),
    3.0,
    notes="情绪转折点，确立本集人物目标",
)

# Scene 3: 宴会厅（三年前）- 提前抵达偶遇陆景琛
scene3_dialogue = [
    DialogueLine(
        id="dlg_003_01",
        shot_id="shot_001_03_02",
        character_id="char_su_wanwan",
        character_name="苏晚晚",
        order=1,
        line_zh="这一次，我不会再让你们这么轻易得逞。",
        line_en="This time, I won't let you get away with it so easily.",
        emotion=Emotion.VENGEFUL,
        voice_direction="低声自语，语气坚定不容置疑",
    ),
    DialogueLine(
        id="dlg_003_02",
        shot_id="shot_001_03_03",
        character_id="char_lu_jingchen",
        character_name="陆景琛",
        order=2,
        line_zh="这位小姐，你说的这句话，是在对谁下战书？",
        line_en="Miss, that declaration of yours — who exactly is it aimed at?",
        emotion=Emotion.CALM,
        voice_direction="低沉平静，带着审视的兴味",
    ),
]

scene3 = Scene(
    id="scene_003",
    episode_id="ep_001",
    scene_number=3,
    location_id="loc_engagement_hall",
    time_of_day=TimeOfDay.DUSK,
    int_ext=IntExt.INT,
    description="苏晚晚提前抵达宴会厅布置场地，趁四下无人对自己发誓复仇，却不知陆景琛已提前到场并听到了她的自言自语。",
    characters_present=["char_su_wanwan", "char_lu_jingchen"],
    emotional_tone="紧张、警惕、意外相遇",
    shot_ids=["shot_001_03_01", "shot_001_03_02", "shot_001_03_03", "shot_001_03_04"],
    dialogue=scene3_dialogue,
)

add_shot(
    "scene_003",
    1,
    3,
    1,
    ["char_su_wanwan"],
    "临江洲际酒店订婚宴会厅",
    "苏晚晚提前抵达空旷的宴会厅，环视四周布置，神情复杂",
    Emotion.ANXIETY,
    CameraSpec(shot_size=ShotSize.LONG_SHOT, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.TRACKING),
    3.0,
)
add_shot(
    "scene_003",
    1,
    3,
    2,
    ["char_su_wanwan"],
    "临江洲际酒店订婚宴会厅",
    "苏晚晚站在台阶前，望着三年前坠落的位置低声立誓",
    Emotion.VENGEFUL,
    CameraSpec(shot_size=ShotSize.CLOSE_UP, angle=CameraAngle.LOW_ANGLE, movement=CameraMovement.STATIC),
    3.5,
    dialogue_ids=["dlg_003_01"],
)
add_shot(
    "scene_003",
    1,
    3,
    3,
    ["char_lu_jingchen"],
    "临江洲际酒店订婚宴会厅",
    "陆景琛从阴影处走出，语气审视地开口询问",
    Emotion.CALM,
    CameraSpec(shot_size=ShotSize.MEDIUM_SHOT, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.STATIC),
    3.0,
    dialogue_ids=["dlg_003_02"],
)
add_shot(
    "scene_003",
    1,
    3,
    4,
    ["char_su_wanwan", "char_lu_jingchen"],
    "临江洲际酒店订婚宴会厅",
    "苏晚晚猛然回头，与陆景琛四目相对，空气瞬间凝固",
    Emotion.SHOCK,
    CameraSpec(shot_size=ShotSize.TWO_SHOT, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.ZOOM_IN),
    2.5,
)

# Scene 4: 宴会厅露台 - 结尾悬念对峙
scene4_dialogue = [
    DialogueLine(
        id="dlg_004_01",
        shot_id="shot_001_04_02",
        character_id="char_lu_jingchen",
        character_name="陆景琛",
        order=1,
        line_zh="苏小姐，你我素不相识，你刚刚说的那句——我杀了你——是什么意思？",
        line_en="Miss Su, you and I are strangers. That thing you said — 'I killed you' — what did you mean?",
        emotion=Emotion.SHOCK,
        voice_direction="语气骤冷，尾音压低带着审讯般的压迫感",
    ),
]

scene4 = Scene(
    id="scene_004",
    episode_id="ep_001",
    scene_number=4,
    location_id="loc_hall_balcony",
    time_of_day=TimeOfDay.NIGHT,
    int_ext=IntExt.EXT,
    description="苏晚晚被陆景琛跟到露台质问，慌乱中脱口而出重生前的记忆碎片，陆景琛神情骤变，本集在悬而未决的对峙中结束。",
    characters_present=["char_su_wanwan", "char_lu_jingchen"],
    emotional_tone="紧张对峙、悬念拉满",
    shot_ids=["shot_001_04_01", "shot_001_04_02", "shot_001_04_03"],
    dialogue=scene4_dialogue,
)

add_shot(
    "scene_004",
    1,
    4,
    1,
    ["char_su_wanwan", "char_lu_jingchen"],
    "宴会厅露台",
    "苏晚晚快步走向露台想要独处，陆景琛紧随其后拦住去路",
    Emotion.ANXIETY,
    CameraSpec(shot_size=ShotSize.MEDIUM_LONG_SHOT, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.TRACKING),
    2.5,
)
add_shot(
    "scene_004",
    1,
    4,
    2,
    ["char_lu_jingchen"],
    "宴会厅露台",
    "陆景琛逼近一步，居高临下地质问，夜风吹动他的衣角",
    Emotion.SHOCK,
    CameraSpec(shot_size=ShotSize.CLOSE_UP, angle=CameraAngle.LOW_ANGLE, movement=CameraMovement.DOLLY_IN),
    3.5,
    dialogue_ids=["dlg_004_01"],
    notes="全集结尾悬念台词，需在此处硬切黑屏并叠加下集预告条",
)
add_shot(
    "scene_004",
    1,
    4,
    3,
    ["char_su_wanwan"],
    "宴会厅露台",
    "苏晚晚脸色骤白，瞳孔剧烈收缩，画面定格叠加'下集见'字样后硬切黑屏",
    Emotion.SHOCK,
    CameraSpec(shot_size=ShotSize.EXTREME_CLOSE_UP, angle=CameraAngle.EYE_LEVEL, movement=CameraMovement.ZOOM_IN),
    2.0,
    notes="强悬念收尾镜头，驱动用户点击下一集",
)

scenes.extend([scene1, scene2, scene3, scene4])

IMAGE_STYLE_TAGS = ["cinematic", "vertical 9:16", "short-drama lighting", "photorealistic", "high contrast"]

for sh in shots:
    chars = "、".join(sh.character)
    img = ImagePrompt(
        id=f"img_{sh.id}",
        shot_id=sh.id,
        prompt_text=(
            f"{sh.location}，{chars}，{sh.action}，情绪：{sh.emotion.value}，"
            f"景别：{sh.camera.shot_size.value}，机位角度：{sh.camera.angle.value}，竖屏短剧质感，电影级布光"
        ),
        negative_prompt="低分辨率, 变形肢体, 多余手指, 水印, 卡通风格",
        style_tags=IMAGE_STYLE_TAGS,
        aspect_ratio="9:16",
        reference_character_ids=sh.character,
    )
    image_prompts.append(img)
    vid = VideoPrompt(
        id=f"vid_{sh.id}",
        shot_id=sh.id,
        image_prompt_id=img.id,
        prompt_text=(
            f"基于关键帧生成{sh.duration}秒短剧镜头：{sh.action}，运镜：{sh.camera.movement.value}，"
            f"人物情绪保持{sh.emotion.value}，画面节奏紧凑符合竖屏短剧观感"
        ),
        motion_description=f"{sh.camera.movement.value} 运镜，{sh.duration}秒内完成动作弧线",
        camera_movement=sh.camera.movement,
        duration=sh.duration,
        fps=24,
        model_target="generic",
        negative_prompt="画面抖动过度, 人物崩坏, 闪烁伪影",
    )
    video_prompts.append(vid)

story_file = StoryFile(story_bible=story_bible, season_arcs=[season_arc])
characters_file = CharactersFile(characters=characters)
scenes_file = ScenesFile(scenes=scenes)
episodes_file = EpisodesFile(episodes=episodes)
shots_file = ShotsFile(shots=shots)
prompts_file = PromptsFile(image_prompts=image_prompts, video_prompts=video_prompts)


def _dump(model: BaseModelLike, filename: str) -> None:
    (HERE / filename).write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


from pydantic import BaseModel as BaseModelLike  # noqa: E402

if __name__ == "__main__":
    _dump(story_file, "story.json")
    _dump(characters_file, "characters.json")
    _dump(scenes_file, "scenes.json")
    _dump(episodes_file, "episodes.json")
    _dump(shots_file, "shots.json")
    _dump(prompts_file, "prompts.json")
    print("seeded 6 json files")
    print("shots:", len(shots), "dialogue lines:", sum(len(s.dialogue) for s in scenes))
