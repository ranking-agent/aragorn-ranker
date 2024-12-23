"""Cache module."""

import logging
import os
import pickle
import redis
from lru import LRU

logger = logging.getLogger(__name__)


class PickleCacheSerializer:
    """Use Python's default serialization."""

    def dumps(self, obj):
        """Return stringified object."""
        return pickle.dumps(obj)

    def loads(self, string):
        """Load object from string."""
        return pickle.loads(string)


class Cache:
    """Cache objects by configurable means."""

    def __init__(
        self,
        cache_path="cache",
        serializer=PickleCacheSerializer,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password="",
        enabled=True,
    ):
        """Connect to cache."""
        self.enabled = enabled
        try:
            if redis_password:
                self.redis = redis.StrictRedis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True,
                )
            else:
                self.redis = redis.StrictRedis(
                    host=redis_host, port=redis_port, db=redis_db, decode_responses=True
                )

            self.redis.get("x")
            logger.debug(
                "Cache connected to redis at %s:%s/%s", redis_host, redis_port, redis_db
            )
        except redis.exceptions.ConnectionError:
            self.redis = None
            logger.error(
                "Failed to connect to redis at %s:%s/%s",
                redis_host,
                redis_port,
                redis_db,
            )
        self.cache_path = cache_path
        if not os.path.exists(self.cache_path):
            try:
                os.makedirs(self.cache_path)
            except PermissionError:
                self.cache_path = None
        self.cache = LRU(1000)
        self.serializer = serializer()

    def curie_query(self, keys):
        pipeline = self.redis.pipeline()
        for key in keys:
            pipeline.hgetall(key)
        result = pipeline.execute()
        return {key: value for key, value in zip(keys, result)}

    def shared_count_query(self, keys):
        pipeline = self.redis.pipeline()
        for key in keys:
            pipeline.get(key)
        result = pipeline.execute()
        return {key: value for key, value in zip(keys, result)}

    def get(self, key):
        """Get a cached item by key."""
        result = None
        if not self.enabled:
            return result
        if key in self.cache:
            result = self.cache[key]
        elif self.redis:
            rec = self.redis.get(key)
            if rec is not None:
                result = self.serializer.loads(rec)
            else:
                result = None
            self.cache[key] = result
        elif self.cache_path is not None:
            path = os.path.join(self.cache_path, key)
            if os.path.exists(path):
                with open(path, "rb") as stream:
                    result = self.serializer.loads(stream.read())
                    self.cache[key] = result
        return result

    def mget(self, *keys):
        """Get multiple cached items by key."""
        result = None
        if not self.enabled:
            return result
        result = []
        if self.redis:
            values = self.redis.mget(keys)
            for rec in values:
                if rec is not None:
                    result.append(self.serializer.loads(rec))
                else:
                    result.append(None)
        elif self.cache_path is not None:
            for key in keys:
                path = os.path.join(self.cache_path, key)
                if os.path.exists(path):
                    with open(path, "rb") as stream:
                        result.append(self.serializer.loads(stream.read()))
                else:
                    result.append(None)
        return result

    def set(self, key, value):
        """Add an item to the cache."""
        if not self.enabled:
            return
        if self.redis:
            if value is not None:
                self.redis.set(key, self.serializer.dumps(value))
                self.cache[key] = value
        elif self.cache_path is not None:
            path = os.path.join(self.cache_path, key)
            with open(path, "wb") as stream:
                stream.write(self.serializer.dumps(value))
            self.cache[key] = value
