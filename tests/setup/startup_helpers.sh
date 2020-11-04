#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../helpers
docker-compose up -d

echo "Waiting for Postgres to start..."
until echo $(docker logs omnicorp_postgres 2>&1) | grep -q "ready to accept connections"; do sleep 1; done
echo "Postgres started."
docker exec omnicorp_postgres mkdir -p /data
docker cp ../data/omnicorp_mondo.csv omnicorp_postgres:/data/omnicorp_mondo.csv
docker cp ../data/omnicorp_hgnc.csv omnicorp_postgres:/data/omnicorp_hgnc.csv
python ../setup/init_omnicorp.py
echo "Postgres initialized."
