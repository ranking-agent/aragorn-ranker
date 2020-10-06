#!/usr/bin/env python

"""Run Redis-REST with referencing."""
import argparse

from gunicorn.app.wsgiapp import WSGIApplication

# ranker.server:APP --bind 0.0.0.0:4868 -w 4 -k uvicorn.workers.UvicornWorker -t 600
app = WSGIApplication()
app.run()
