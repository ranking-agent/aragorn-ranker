#!/usr/bin/env python
from gunicorn.app.wsgiapp import WSGIApplication

# --bind 0.0.0.0:8080 -w 1 -k uvicorn.workers.UvicornWorker -t 600 ranker.server:APP
app = WSGIApplication()

app.run()
