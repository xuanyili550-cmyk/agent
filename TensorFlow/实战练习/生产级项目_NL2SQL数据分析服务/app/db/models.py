# 【构建 4/16 · 数据】无内部依赖，先定表结构
"""SQLAlchemy ORM 模型 + 表的中文元数据(供 NL2SQL 检索用)。"""
from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(32))
    price: Mapped[float] = mapped_column(Float)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    qty: Mapped[int] = mapped_column(Integer)
    city: Mapped[str] = mapped_column(String(32))
    amount: Mapped[float] = mapped_column(Float)


# 表元数据(中文说明)：NL2SQL 检索相关表时用，也拼进 LLM 提示
TABLE_METADATA = {
    "products": {
        "desc": "商品表",
        "columns": {"id": "商品ID", "name": "商品名", "category": "类目", "price": "单价"},
        "keywords": ["商品", "类目", "单价", "价格", "产品"],
    },
    "orders": {
        "desc": "订单表",
        "columns": {"id": "订单ID", "product_id": "商品ID", "qty": "数量", "city": "城市", "amount": "金额"},
        "keywords": ["订单", "销量", "数量", "金额", "城市", "销售", "卖"],
    },
}

SEED_PRODUCTS = [
    Product(id=1, name="手机", category="数码", price=3999),
    Product(id=2, name="耳机", category="数码", price=299),
    Product(id=3, name="T恤", category="服装", price=99),
    Product(id=4, name="跑鞋", category="服装", price=499),
]
SEED_ORDERS = [
    Order(id=1, product_id=1, qty=2, city="北京", amount=7998),
    Order(id=2, product_id=2, qty=5, city="上海", amount=1495),
    Order(id=3, product_id=3, qty=10, city="北京", amount=990),
    Order(id=4, product_id=4, qty=3, city="广州", amount=1497),
    Order(id=5, product_id=1, qty=1, city="上海", amount=3999),
    Order(id=6, product_id=3, qty=8, city="广州", amount=792),
]
