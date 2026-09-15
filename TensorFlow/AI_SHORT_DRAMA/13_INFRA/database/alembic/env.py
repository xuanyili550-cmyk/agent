"""alembic 迁移环境：告诉 alembic 用哪份 metadata 做 autogenerate、连哪个库跑迁移。

在整个平台里的位置：生产环境的表结构变更不走 ``Base.metadata.create_all``（那只给开发/测试用），
而是 ``alembic upgrade head`` 按 versions/ 里的迁移脚本逐步演进。本文件由 ``alembic init`` 生成，
只改了两处：
1. 把项目根加进 sys.path 并用 importlib 加载 ``13_INFRA.database.models``（包名数字开头，普通 import 语句不行）；
2. 环境变量 ``DATABASE_URL`` 优先于 alembic.ini 里的 sqlalchemy.url，这样 Docker/CI 不用改 ini 文件。
"""

import importlib
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# alembic 的 Config 对象，可以读到当前使用的 .ini 文件里的各项配置
config = context.config

# 按 .ini 文件配置 Python logging（基本上就是把 logger 设置好）
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# "13_INFRA" 不是合法标识符，普通 `import` 语句到不了它；importlib.import_module 可以处理数字开头的包名，
# 前提是 AI_SHORT_DRAMA 根目录（13_INFRA 的上一级）在 sys.path 上。
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 只 import models（不 import session）：拿 metadata 做 autogenerate 对比即可，不要顺带按 settings 建 engine
models = importlib.import_module("13_INFRA.database.models")
target_metadata = models.Base.metadata

# 环境变量里的 DATABASE_URL 覆盖 alembic.ini 的 sqlalchemy.url，和应用本身读同一个变量
database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

# env.py 需要的其它配置项也可以从 config 里取，例如：
# my_important_option = config.get_main_option("my_important_option")
# ... 等等


def run_migrations_offline() -> None:
    """离线模式跑迁移：只用 URL 配置 context，不建 Engine（给了 Engine 也行）。

    跳过 Engine 之后甚至不需要 DBAPI 可用；此模式下 context.execute() 不真正执行 SQL，
    而是把语句输出成脚本（``alembic upgrade head --sql``），适合交给 DBA 审核后手工执行。
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,  # 参数直接内联进 SQL 文本，生成的脚本才能独立执行
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式跑迁移：建 Engine、拿一条连接绑到 context 上，直接对数据库执行。

    用 NullPool：迁移是一次性进程，不需要连接池，跑完即断开。
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


# alembic 命令行按是否带 --sql 决定走离线还是在线
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
