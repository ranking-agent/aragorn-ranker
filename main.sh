#!/usr/bin/env bash

# export $(egrep -v '^#' .env | xargs)

gunicorn ranker.server:APP --bind 0.0.0.0:4868 -w 4 -k uvicorn.workers.UvicornWorker -t 600 --log-level=debug
