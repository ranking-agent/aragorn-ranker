#!/bin/bash

docker run -d --name omnicorp_redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

echo "Waiting for redisgraph to start..."
until echo $(docker logs omnicorp_redis 2>&1) | grep -q "Ready to accept connections"; do sleep 1; done
echo "redisgraph started."

cd ../OmnicorpTestData
redisgraph-bulk-insert OMNICORP -N CURIE curie_to_pmids.txt -R cooccurs curie_pairs.txt -o $'\t'

echo "Omnicorp initialized."
