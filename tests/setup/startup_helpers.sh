#!/bin/bash

docker run -d --name omnicorp_redis -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
#Need redis-cli for loading
sudo apt-get install -y redis-tools

echo "Waiting for redisgraph to start..."
until echo $(docker logs omnicorp_redis 2>&1) | grep -q "Ready to accept connections"; do sleep 1; done
echo "redisgraph started."

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../OmnicorpTestData
cat redis_curies | redis-cli -p 6379
cat redis_curie_pairs | redis-cli -p 6379

echo "Omnicorp initialized."
