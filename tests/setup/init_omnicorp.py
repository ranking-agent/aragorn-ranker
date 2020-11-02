"""Initialize omnicorp testing instance."""
import os
from dotenv import load_dotenv
import psycopg2

file_path = os.path.dirname(os.path.realpath(__file__))
dotenv_path = os.path.abspath(os.path.join(file_path, '..', '.env'))
load_dotenv(dotenv_path=dotenv_path)

conn = psycopg2.connect(
    dbname=os.environ['OMNICORP_DB'],
    user=os.environ['OMNICORP_USER'],
    host=os.environ['OMNICORP_HOST'],
    port=os.environ['OMNICORP_PORT'],
    password=os.environ['OMNICORP_PASSWORD'])

cur = conn.cursor()

statement = f"CREATE SCHEMA IF NOT EXISTS omnicorp;\n"
statement += f"CREATE TABLE IF NOT EXISTS omnicorp.mondo (curie TEXT, pubmedid INTEGER);\n"
statement += f"""COPY omnicorp.mondo (curie,pubmedid)
FROM '/data/omnicorp_mondo.csv' DELIMITER ',' CSV HEADER;\n
"""
statement += f"CREATE TABLE IF NOT EXISTS omnicorp.hgnc (curie TEXT, pubmedid INTEGER);\n"
statement += f"""COPY omnicorp.hgnc (curie,pubmedid)
FROM '/data/omnicorp_hgnc.csv' DELIMITER ',' CSV HEADER;
"""

cur.execute(statement)
cur.close()
conn.commit()
conn.close()
