import argparse
import re

import pandas as pd
import psycopg2
from  ensemble_compilation.physical_db import DBConnection
from tqdm import tqdm

parser = argparse.ArgumentParser(description='get_truecard')

parser.add_argument('--version', type=str, help='datasets_dir', default='power')

args = parser.parse_args()
version = args.version

# db_connection = DBConnection(db='postgres',db_user='postgres',db_host="/var/run/postgresql")  # modify
# db_connection = DBConnection(db='autocard', db_password="jintao2020", db_user='jintao', db_host="localhost")  # modify
# ai4db
db_connection = DBConnection(db='ai4db', db_password="linux123", db_user='user1', db_host="localhost")
# ai4db
fschema = open('/home/hdd/user1/oblab/CardinalityEstimationTestbed/Overall/train-test-data/forest_power-data-sql/schema_power.sql')
schemasql = fschema.read()
dropsql = 'DROP TABLE ' + version + ';'

try:
    db_connection.submit_query(dropsql)  # Clear the table with the current name
except Exception as e:
    pass

try:
    db_connection.submit_query(schemasql)  # establish schema
except Exception as e:
    pass

# csvsql = r'\copy ' + version + ' from ./csvdata_sql/' + version + '.csv with csv header;'
# db_connection.submit_query(csvsql)
# os.system('psql -U postgres')
# os.system(csvsql)
df = pd.read_csv('/home/hdd/user1/oblab/CardinalityEstimationTestbed/Overall/train-test-data/forest_power-data-sql/power.csv', sep=';', escapechar='\\', encoding='utf-8', low_memory=False,
                 quotechar='"')
columns = tuple(df.columns)
# connection = psycopg2.connect(user='postgres', host="/var/run/postgresql", database='postgres')
# connection = psycopg2.connect(user='jintao', host="localhost", database='autocard', password="jintao2020")
connection = db_connection.get_connection()
cur = connection.cursor()
file = open('/home/hdd/user1/oblab/CardinalityEstimationTestbed/Overall/train-test-data/forest_power-data-sql/no_head/power.csv', 'r')  # Read a file without a header
cur.copy_from(file, version, sep=';')
connection.commit()