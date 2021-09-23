#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../helpers
docker-compose up -d

echo "Waiting for Postgres to start..."
until echo $(docker logs omnicorp_postgres 2>&1) | grep -q "ready to accept connections"; do sleep 1; done
echo "Postgres started."

docker exec omnicorp_postgres mkdir -p /data
docker cp ../InputJson_1.2/omnicorp_mesh.csv omnicorp_postgres:/data/omnicorp_mesh.csv
docker cp ../InputJson_1.2/omnicorp_mondo.csv omnicorp_postgres:/data/omnicorp_mondo.csv
docker cp ../InputJson_1.2/omnicorp_ncbigene.csv omnicorp_postgres:/data/omnicorp_ncbigene.csv
docker cp ../InputJson_1.2/omnicorp_ncbitaxon.csv omnicorp_postgres:/data/omnicorp_ncbitaxon.csv
docker cp ../InputJson_1.2/omnicorp_chebi.csv omnicorp_postgres:/data/omnicorp_chebi.csv
docker cp ../InputJson_1.2/omnicorp_chembl.compound.csv omnicorp_postgres:/data/omnicorp_chembl.compound.csv

python3 ../setup/init_omnicorp.py
echo "Postgres initialized."
