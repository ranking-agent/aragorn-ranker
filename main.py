#!/usr/bin/env python
# from gunicorn.app.wsgiapp import WSGIApplication
#
# --bind localhost:5003 -w 1 -k uvicorn.workers.UvicornWorker -t 600 ranker.server:APP
# app = WSGIApplication()
#
# app.run()

import uvicorn

class App:
    ...

app = App()

if __name__ == "__main__":
    uvicorn.run("ranker.server:APP", host="localhost", port=5003, log_level="info")