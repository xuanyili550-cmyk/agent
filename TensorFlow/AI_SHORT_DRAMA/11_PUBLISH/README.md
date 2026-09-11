# 11_PUBLISH

各平台发布客户端，统一实现 `base.PublishClient.upload(episode_manifest, dry_run=False) -> PublishResult`。

- `youtube/client.py`：YouTube Data API v3（google-api-python-client + OAuth2），有官方公开 API。
- `tiktok/client.py`：TikTok Content Posting API v2（requests 直连 HTTP），有官方公开 API。
- `reelshort/`、`dramabox/`、`goodshort/`：没有官方公开 API，使用 `generic_http_adapter.py`
  提供的可配置通用 HTTP 上传骨架，见各自目录下的 README.md。

`--dry-run` 模式下只调用 `build_payload` + `validate_payload`，不发起任何网络请求，
用 `publish_cli.py` 可以直接跑通校验逻辑（见该文件顶部注释）。
