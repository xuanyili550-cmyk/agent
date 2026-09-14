"""质检官：在进入生产前用 LLM 审一集的剧本 + 分镜 + 对白，输出 QCVerdict。

两层检查：
1. ``rule_check()`` —— 纯代码的硬校验（引用一致性、时长范围、钩子/悬念非空），零成本、
   结果确定，先跑；发现 blocker 直接判不通过，不浪费一次 LLM 调用。
2. ``review()`` —— LLM 按 prompt 里的 7 条清单做语义检查（人物是否走形、钩子够不够强）。
两层结果合并成一份 QCVerdict。
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        break

import schemas as sch

from .base import PROMPTS_DIR, BaseAgent, LLMProvider
from .memory import ConversationMemory

MIN_SHOT_SECONDS = 1.0
MAX_SHOT_SECONDS = 8.0
DURATION_TOLERANCE = 0.5  # 分镜总时长与目标时长偏差超过 50% 视为节奏问题


def rule_check(
    episode: sch.Episode,
    scenes: list[sch.Scene],
    shots: list[sch.Shot],
    characters: list[sch.Character],
) -> list[sch.QCIssue]:
    """不需要 LLM 的硬校验。"""
    issues: list[sch.QCIssue] = []
    char_ids = {c.id for c in characters}
    scene_ids = {s.id for s in scenes}
    shot_ids = {s.id for s in shots}

    if not episode.hook.strip():
        issues.append(sch.QCIssue(code="hook_too_slow", severity=sch.IssueSeverity.HIGH, location=episode.id, description="缺少开场钩子"))
    if not episode.cliffhanger.strip():
        issues.append(sch.QCIssue(code="weak_cliffhanger", severity=sch.IssueSeverity.HIGH, location=episode.id, description="缺少结尾悬念"))

    for scene in scenes:
        if scene.episode_id != episode.id:
            issues.append(
                sch.QCIssue(
                    code="reference_mismatch",
                    severity=sch.IssueSeverity.BLOCKER,
                    location=scene.id,
                    description=f"scene.episode_id={scene.episode_id} 不是本集 {episode.id}",
                )
            )
        for cid in scene.characters_present:
            if cid not in char_ids:
                issues.append(
                    sch.QCIssue(code="reference_mismatch", severity=sch.IssueSeverity.BLOCKER, location=scene.id, description=f"出场角色 {cid} 不在角色表里")
                )
        for line in scene.dialogue:
            if line.character_id not in char_ids:
                issues.append(
                    sch.QCIssue(
                        code="reference_mismatch",
                        severity=sch.IssueSeverity.BLOCKER,
                        location=line.id,
                        description=f"对白角色 {line.character_id} 不在角色表里",
                    )
                )
            if line.shot_id not in shot_ids:
                issues.append(
                    sch.QCIssue(
                        code="reference_mismatch", severity=sch.IssueSeverity.BLOCKER, location=line.id, description=f"对白指向不存在的镜头 {line.shot_id}"
                    )
                )

    total = 0.0
    for shot in shots:
        total += shot.duration
        if shot.scene_id not in scene_ids:
            issues.append(
                sch.QCIssue(
                    code="reference_mismatch", severity=sch.IssueSeverity.BLOCKER, location=shot.id, description=f"镜头指向不存在的场次 {shot.scene_id}"
                )
            )
        for cid in shot.character:
            if cid not in char_ids:
                issues.append(
                    sch.QCIssue(code="reference_mismatch", severity=sch.IssueSeverity.BLOCKER, location=shot.id, description=f"镜头角色 {cid} 不在角色表里")
                )
        if not (MIN_SHOT_SECONDS <= shot.duration <= MAX_SHOT_SECONDS):
            issues.append(
                sch.QCIssue(
                    code="pacing",
                    severity=sch.IssueSeverity.MEDIUM,
                    location=shot.id,
                    description=f"镜头时长 {shot.duration}s 超出 {MIN_SHOT_SECONDS}-{MAX_SHOT_SECONDS}s",
                    suggestion="拆分或合并镜头",
                )
            )
        if not shot.action.strip() or not shot.location.strip():
            issues.append(sch.QCIssue(code="ungenerable_shot", severity=sch.IssueSeverity.MEDIUM, location=shot.id, description="镜头缺少地点或动作描述"))

    if shots and episode.duration_seconds and abs(total - episode.duration_seconds) / episode.duration_seconds > DURATION_TOLERANCE:
        issues.append(
            sch.QCIssue(
                code="pacing",
                severity=sch.IssueSeverity.MEDIUM,
                location=episode.id,
                description=f"分镜总时长 {total:.1f}s 与目标 {episode.duration_seconds}s 偏差过大",
                suggestion="按目标时长增减镜头数量",
            )
        )
    return issues


def has_blocking_issue(issues: list[sch.QCIssue]) -> bool:
    return any(i.severity in (sch.IssueSeverity.BLOCKER, sch.IssueSeverity.HIGH) for i in issues)


class QCOfficerAgent(BaseAgent):
    SYSTEM_PROMPT_FILE = PROMPTS_DIR / "qc_officer_agent_system.txt"

    def __init__(self, provider: LLMProvider, max_retries: int = 3, memory: ConversationMemory | None = None):
        super().__init__(provider, self.SYSTEM_PROMPT_FILE.read_text(encoding="utf-8"), max_retries=max_retries, memory=memory)

    def review(
        self,
        episode: sch.Episode,
        script: sch.Script,
        scenes: list[sch.Scene],
        shots: list[sch.Shot],
        characters: list[sch.Character],
        *,
        skip_llm_on_blocker: bool = True,
    ) -> sch.QCVerdict:
        rule_issues = rule_check(episode, scenes, shots, characters)
        verdict_id = f"qc_{episode.id}"
        if skip_llm_on_blocker and any(i.severity == sch.IssueSeverity.BLOCKER for i in rule_issues):
            return sch.QCVerdict(
                id=verdict_id,
                episode_id=episode.id,
                passed=False,
                score=0,
                issues=rule_issues,
                summary="硬校验发现引用错误（blocker），未进入语义检查。",
            )

        cast = "\n".join(f"- {c.id} {c.name}（{c.role.value}）：{c.appearance}；性格 {'、'.join(c.personality_traits)}" for c in characters)
        shot_lines = "\n".join(
            f"- {s.id} scene={s.scene_id} {s.duration}s [{s.camera.shot_size.value}/{s.camera.angle.value}/{s.camera.movement.value}] "
            f"角色={','.join(s.character)} 地点={s.location} 动作={s.action} 情绪={s.emotion.value}"
            for s in shots
        )
        dialogue_lines = "\n".join(f"- {d.id} shot={d.shot_id} {d.character_name}({d.character_id})：{d.line_zh}" for sc in scenes for d in sc.dialogue)
        user_prompt = (
            f"待检查集：{episode.id}《{episode.title}》目标时长 {episode.duration_seconds}s\n"
            f"hook：{episode.hook}\ncliffhanger：{episode.cliffhanger}\n梗概：{episode.synopsis}\n\n"
            f"角色表：\n{cast}\n\n剧本正文：\n{script.content[:3000]}\n\n"
            f"分镜：\n{shot_lines}\n\n对白：\n{dialogue_lines or '（无）'}\n\n"
            f"代码硬校验已发现 {len(rule_issues)} 条问题：{[i.code for i in rule_issues]}。"
            f"请按检查清单完成语义检查，id 使用 '{verdict_id}'，episode_id 为 '{episode.id}'。"
        )
        verdict = self.generate(user_prompt, sch.QCVerdict)
        merged = rule_issues + [i for i in verdict.issues if i not in rule_issues]
        passed = verdict.passed and not has_blocking_issue(merged)
        return verdict.model_copy(update={"id": verdict_id, "episode_id": episode.id, "issues": merged, "passed": passed})
