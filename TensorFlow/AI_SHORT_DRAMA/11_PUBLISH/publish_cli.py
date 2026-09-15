"""发布命令行：用 --dry-run 演练每个 PublishClient，不需要网络和真实凭据就能验证请求体组装与校验。

用法：python publish_cli.py --episode ../10_EPISODES/EP001/episode_manifest.json --dry-run
生产里真正的发布由 13_INFRA 的任务触发，这个 CLI 主要用于本地检查平台配置和 manifest 字段是否齐全。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# 本目录（找 base / generic_http_adapter）和 10_EPISODES（找 episode_manifest）都要进 sys.path
PUBLISH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PUBLISH_ROOT))
sys.path.insert(0, str(PUBLISH_ROOT.parent / "10_EPISODES"))

from episode_manifest import EpisodeManifest  # type: ignore  # noqa: E402


def _load_client_module(platform: str):
    """按文件路径加载 <platform>/client.py。

    每个平台目录都有一个同名的 client.py，如果用普通 import 会在 sys.modules 里互相覆盖，
    所以用 importlib 以 "<platform>_client" 这样唯一的模块名加载。
    """
    spec = importlib.util.spec_from_file_location(f"{platform}_client", PUBLISH_ROOT / platform / "client.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_clients() -> dict:
    """实例化全部五个平台客户端。TikTok 传假 token 是为了 dry-run 时不因缺 TIKTOK_ACCESS_TOKEN 而无法构造。"""
    youtube_mod = _load_client_module("youtube")
    tiktok_mod = _load_client_module("tiktok")
    reelshort_mod = _load_client_module("reelshort")
    dramabox_mod = _load_client_module("dramabox")
    goodshort_mod = _load_client_module("goodshort")

    return {
        "youtube": youtube_mod.YouTubePublishClient(),
        "tiktok": tiktok_mod.TikTokPublishClient(access_token="fake-token-for-dry-run"),
        "reelshort": reelshort_mod.ReelShortPublishClient(),
        "dramabox": dramabox_mod.DramaBoxPublishClient(),
        "goodshort": goodshort_mod.GoodShortPublishClient(),
    }


def main() -> None:
    """解析参数，加载 manifest，对选中的平台逐个调 upload；单个平台出错不影响其他平台继续演练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--platform", default="all", help="youtube|tiktok|reelshort|dramabox|goodshort|all")
    args = parser.parse_args()

    episode = EpisodeManifest.load(args.episode)
    clients = build_clients()
    targets = clients if args.platform == "all" else {args.platform: clients[args.platform]}

    for name, client in targets.items():
        try:
            result = client.upload(episode, dry_run=args.dry_run)
            print(f"[{name}] status={result.status.value} dry_run={result.dry_run} payload_keys={list((result.payload or {}).keys())}")
        except Exception as exc:  # noqa: BLE001 - surface adapter/config errors in the CLI
            print(f"[{name}] ERROR: {exc}")  # 配置缺失 / 校验失败在 CLI 里直接打印，方便一眼看出哪个平台没配好


if __name__ == "__main__":
    main()
