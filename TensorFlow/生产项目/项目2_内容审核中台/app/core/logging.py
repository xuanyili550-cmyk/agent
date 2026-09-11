import json, logging, sys, time
class JsonFormatter(logging.Formatter):
    def format(self, r):
        p={"ts":time.strftime("%Y-%m-%dT%H:%M:%S",time.localtime(r.created)),"level":r.levelname,"logger":r.name,"msg":r.getMessage()}
        if r.exc_info: p["exc"]=self.formatException(r.exc_info)
        return json.dumps(p,ensure_ascii=False)
def setup_logging(debug=False):
    h=logging.StreamHandler(sys.stdout);h.setFormatter(JsonFormatter())
    root=logging.getLogger();root.handlers[:]=[h];root.setLevel(logging.DEBUG if debug else logging.INFO)
def get_logger(n): return logging.getLogger(n)
