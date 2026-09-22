from sqlalchemy import create_engine, Column, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    url = Column(String, unique=True)
    source = Column(String)
    published_at = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    content = Column(Text)
    sentiment_score = Column(Float)
    sentiment_label = Column(String)

# Create tables if they don't exist
Base.metadata.create_all(engine)

def get_session():
    return Session()