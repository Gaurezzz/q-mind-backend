from fastapi import FastAPI
from db.config import engine
from db import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()