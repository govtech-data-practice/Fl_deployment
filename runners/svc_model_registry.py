#!/usr/bin/env python3
"""Model Registry — versioned model storage microservice."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, time, hashlib

STORE = os.environ.get('MODEL_STORE', '/tmp/models')
os.makedirs(STORE, exist_ok=True)

class ModelHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
        elif self.path == '/models':
            models = []
            for f in sorted(os.listdir(STORE)):
                if f.endswith('.json'):
                    with open(os.path.join(STORE, f)) as fh:
                        models.append(json.load(fh))
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps(models).encode())
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path == '/models':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            body['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
            body['id'] = hashlib.sha256(json.dumps(body).encode()).hexdigest()[:12]
            with open(os.path.join(STORE, body['id'] + '.json'), 'w') as f:
                json.dump(body, f)
            self.send_response(201); self.end_headers()
            self.wfile.write(json.dumps(body).encode())
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8083'))
    print(f'Model Registry listening on :{port}')
    HTTPServer(('0.0.0.0', port), ModelHandler).serve_forever()
