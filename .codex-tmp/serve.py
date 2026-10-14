import threading, os, sys
sys.path.insert(0, "src")
from http.server import ThreadingHTTPServer
from pca_model_builder import web_model_results as ui
srv = ThreadingHTTPServer(("127.0.0.1", 8899), ui.ModelResultsHandler)
print("READY", flush=True)
srv.serve_forever()
