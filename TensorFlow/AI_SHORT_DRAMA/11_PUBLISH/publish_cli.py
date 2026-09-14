# Demonstrates every PublishClient with --dry-run so payload building/validation
# can be exercised without network access or real credentials.
# Usage: python publish_cli.py --episode ../10_EPISODES/EP001/episode_manifest.json --dry-run
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PUBLISH_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PUBLISH_ROOT))
sys.path.insert(0, str(PUBLISH_ROOT.parent / "10_EPISODES"))

from episode_manifest import EpisodeManifest  # type: ignore  # noqa: E402


def _load_client_module(platform: str):
    # each platform dir has its own client.py; load under a unique module name
    # so they don't collide with each other in sys.modules.
    spec = importlib.util.spec_from_file_location(f"{platform}_client", PUBLISH_ROOT / platform / "client.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_clients() -> dict:
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
            print(f"[{name}] ERROR: {exc}")


if __name__ == "__main__":
    main()
