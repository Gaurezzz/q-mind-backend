from fastapi import FastAPI
from db.config import engine
from db import models
from api import auth, labels, material

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router)
app.include_router(labels.router)
app.include_router(material.router)