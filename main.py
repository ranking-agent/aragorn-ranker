#!/usr/bin/env python
# from gunicorn.app.wsgiapp import WSGIApplication
#
# # --bind 0.0.0.0:8080 -w 1 -k uvicorn.workers.UvicornWorker -t 600 ranker.server:APP
# app = WSGIApplication()
#
# app.run()

import uvicorn


class App: ...


app = App()

if __name__ == "__main__":
    uvicorn.run("ranker.server:APP", host="127.0.0.1", port=5003, log_level="info")
