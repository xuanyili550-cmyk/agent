"""ORM 声明基类：所有表模型都继承这里的 ``Base``，共享同一份 ``metadata``。

单独放一个文件而不是放进 models.py，是为了让 session.py（``Base.metadata.create_all`` 建表）
和 models.py 之间不产生循环 import：session 只依赖 base，models 也只依赖 base。
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 风格的声明基类；本身不定义任何列，只提供共享的 metadata / registry。"""

    pass
