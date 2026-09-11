# DramaBox adapter

DramaBox 目前没有公开发布的官方开发者 API。

这里提供的 `DramaBoxPublishClient`（`client.py`）只是一个**可配置的通用 HTTP 上传适配器骨架**，
基于 `../generic_http_adapter.py` 的 `GenericHTTPAdapter`：

- endpoint / headers / payload 字段模板全部来自 `config.yaml`（复制 `config.example.yaml` 后按需修改，
  或者通过 `DRAMABOX_CONFIG_PATH` 环境变量指定路径）。
- 鉴权 token 通过 `token_env_var`（默认 `DRAMABOX_API_TOKEN`）指定的环境变量读取，不写死在代码或配置里。
- `--dry-run` 模式下只会渲染并校验 payload，不会发起真实网络请求。

**要真正对接 DramaBox，需要平台方直接提供私有的上传接口文档/合作协议**，
在拿到真实的 endpoint、鉴权方式和字段定义之前，`config.example.yaml`
里的 endpoint 是占位符（`REPLACE_WITH_DRAMABOX_UPLOAD_ENDPOINT`），
`validate_payload` 会在检测到未替换的占位符时主动报错，防止误以为已经可用。
