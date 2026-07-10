from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from models import Base   # imports Article and Chunk table definitions

load_dotenv()

database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)   # the connection to Postgres

Base.metadata.create_all(engine)       # creates all tables defined in models.py

print("Tables created successfully.")