"""Initialize omnicorp testing instance."""
import os
from dotenv import load_dotenv
import psycopg2

file_path = os.path.dirname(os.path.realpath(__file__))
dotenv_path = os.path.abspath(os.path.join(file_path, '..', '.env'))
load_dotenv(dotenv_path=dotenv_path)

print('Connecting')

conn = psycopg2.connect(
    dbname=os.environ['OMNICORP_DB'],
    user=os.environ['OMNICORP_USER'],
    host=os.environ['OMNICORP_HOST'],
    port=os.environ['OMNICORP_PORT'],
    password=os.environ['OMNICORP_PASSWORD'])

print('Connected'
      )
cur = conn.cursor()

statement = f"CREATE SCHEMA IF NOT EXISTS omnicorp;\n"

curie_types: list = ['mesh', 'mondo', 'ncbigene', 'ncbitaxon', 'chebi', '"chembl.compound"']

for item in curie_types:
    print(f"working: {item}")
    statement += f"CREATE TABLE IF NOT EXISTS omnicorp.{item} (pubmedid varchar(255), curie varchar(255));\n"
    statement += f"CREATE INDEX ON omnicorp.{item} (pubmedid);\n"
    statement += f"CREATE INDEX ON omnicorp.{item} (curie);\n"
    new_item = item.replace('"', '')
    statement += f"COPY omnicorp.{item} (curie,pubmedid) FROM '/data/omnicorp_{new_item}.csv' DELIMITER ',' CSV HEADER;\n"



cur.execute(statement)
cur.close()
conn.commit()
conn.close()
