# CI 工作流

`github-workflow.ai_short_drama.yml` 是 GitHub Actions 工作流（lint + 全量测试 + Alembic 从零迁移 + Docker 构建）。

GitHub 只识别仓库根目录下的 `.github/workflows/`，而推送这个文件需要 token 带 `workflow` 权限
（本次自动推送用的 gh OAuth token 没有这个权限，被 GitHub 拒绝）。启用方式二选一：

```bash
# 方式一：本机 gh 补一次 workflow 权限后，把文件挪到仓库根目录再推
gh auth refresh -h github.com -s workflow
git mv TensorFlow/AI_SHORT_DRAMA/ci/github-workflow.ai_short_drama.yml .github/workflows/ai_short_drama.yml
git commit -m "启用 AI_SHORT_DRAMA CI" && git push

# 方式二：在 GitHub 网页上 Actions -> New workflow -> set up a workflow yourself，把文件内容粘进去
```

文件里的 `working-directory` / `paths` 已经按"本仓库里 AI_SHORT_DRAMA 位于 TensorFlow/ 子目录"写好，不需要改。
