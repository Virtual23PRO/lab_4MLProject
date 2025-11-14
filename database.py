from sqlalchemy.engine import URL
from sqlalchemy import Integer, String
from typing import List
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector 
from sqlalchemy import create_engine


db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",    
    host="localhost",
    port=5555,              
    database="similarity_search_service_db",
)

class Base(DeclarativeBase):
    __abstract__ = True

class Images(Base):
    __tablename__ = "images"
    VECTOR_LENGTH = 512

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String(256))
    image_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))


engine = create_engine(db_url)
Base.metadata.create_all(engine)
