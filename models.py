from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from database import Base

class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # store as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
