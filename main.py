from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from datetime import date, datetime

db = create_engine("postgresql://postgres:admin@localhost:5432/awt")

Session = sessionmaker(db)
session = Session()

import statsmodel 
