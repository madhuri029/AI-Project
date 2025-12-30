from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Note
import json
import numpy as np
import os
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",   
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set")


Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Notes Manager")

# Schemas 
class NoteCreate(BaseModel):
    title: str
    content: str

class SearchQuery(BaseModel):
    query: str

# Utils 
def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Routes ----------
@app.post("/notes")
def create_note(note: NoteCreate):
    db: Session = SessionLocal()

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=note.content
    ).data[0].embedding

    db_note = Note(
        title=note.title,
        content=note.content,
        embedding=json.dumps(embedding)
    )

    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    db.close()

    return db_note


@app.get("/notes")
def get_notes():
    db: Session = SessionLocal()
    notes = db.query(Note).all()
    db.close()
    return notes


@app.post("/search")
def search_notes(query: SearchQuery):
    db: Session = SessionLocal()
    notes = db.query(Note).all()
    db.close()

    if not notes:
        return {"message": "No notes found"}

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query.query
    ).data[0].embedding

    best_note = max(
        notes,
        key=lambda n: cosine(json.loads(n.embedding), query_embedding)
    )

    return best_note


