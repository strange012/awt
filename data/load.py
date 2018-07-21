import csv
import re
import time
import urllib
import urllib2
from datetime import date, datetime

from bs4 import BeautifulSoup
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import psycopg2


def get_url(port, year, interval):
    return "https://awt.cbp.gov/Home/OpenExcel?port={}&rptFrom=07%2F01%2F{}&rptTo=06%2F30%2F{}".format(port, year, year + interval)


def get_data(port, year, interval):
    html = urllib2.urlopen(get_url(port, year, interval)).read()
    soup = BeautifulSoup(html, "html.parser")
    return [[td.text.encode("utf-8") for td in row.find_all("td")]
            for row in soup.find("table").find_all("tr") if row.find("td")]


def data_to_db(data):
    for row in data:
        wait_time = WaitTime(
            airport=row[0],
            terminal=row[1],
            date=datetime.strptime(row[2], '%m/%d/%Y'),
            hour=int(row[3][:2]),
            us_av_time=int(row[4]),
            us_max_time=int(row[5]),
            nonus_av_time=int(row[6]),
            nonus_max_time=int(row[7]),
            all_av_time=int(row[8]),
            all_max_time=int(row[9]),
            pass_interval_15=int(row[10]),
            pass_interval_30=int(row[11]),
            pass_interval_45=int(row[12]),
            pass_interval_60=int(row[13]),
            pass_interval_90=int(row[14]),
            pass_interval_120=int(row[15]),
            pass_interval_plus=int(row[16]),
            excluded=int(row[17]),
            total=int(row[18]),
            flights=int(row[19]),
            booths=int(row[20])
        )
        session.add(wait_time)
    session.commit()
    return


def data_to_csv(data):
    with open("out.csv", "wb") as f:
        wr = csv.writer(f)
        wr.writerows(data)


soup = BeautifulSoup(urllib2.urlopen(
    "https://awt.cbp.gov/").read(), "html.parser")
ports = [str(x['value'])
         for x in soup.find(id="Airport").find_all('option', text=re.compile("^[A-Z0-9]"))]

print(ports)


db_config = json.load(open('db_config.json'))
db = create_engine("postgresql://{}:{}@{}:{}/{}".format(db_config['name'], db_config['user'],
                                                             db_config['host'], db_config['port'], db_config['db']))
base = declarative_base()


class WaitTime(base):
    __tablename__ = 'awt'
    id = Column(Integer, primary_key=True)
    airport = Column(String, nullable=False)
    terminal = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    hour = Column(Integer, nullable=False)
    us_av_time = Column(Integer, nullable=False)
    us_max_time = Column(Integer, nullable=False)
    nonus_av_time = Column(Integer, nullable=False)
    nonus_max_time = Column(Integer, nullable=False)
    all_av_time = Column(Integer, nullable=False)
    all_max_time = Column(Integer, nullable=False)
    pass_interval_15 = Column(Integer, nullable=False)
    pass_interval_30 = Column(Integer, nullable=False)
    pass_interval_45 = Column(Integer, nullable=False)
    pass_interval_60 = Column(Integer, nullable=False)
    pass_interval_90 = Column(Integer, nullable=False)
    pass_interval_120 = Column(Integer, nullable=False)
    pass_interval_plus = Column(Integer, nullable=False)
    excluded = Column(Integer, nullable=False)
    total = Column(Integer, nullable=False)
    flights = Column(Integer, nullable=False)
    booths = Column(Integer, nullable=False)


Session = sessionmaker(db)
session = Session()
base.metadata.create_all(db)


init = time.time()
for port in ports[30:]:
    for year in range(2008, 2018):
        start = time.time()
        data_to_db(get_data(port, year, 1))
        print "{} : {} year, {} s".format(port, year, time.time() - start)
    print "{}.{} : total time {} s".format(ports.index(port), port, time.time() - init)
