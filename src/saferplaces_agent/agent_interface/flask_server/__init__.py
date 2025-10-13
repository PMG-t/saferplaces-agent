from flask import Flask
from flask_cors import CORS

from .. import __GRAPH_REGISTRY__ 


def create_app():
    app = Flask(__name__)
    
    # DOC: Enable CORS for all routes
    CORS(app)
    
    # DOC: If there will be a DB.. here we would initialize it
    app.__GRAPH_REGISTRY__ = __GRAPH_REGISTRY__

    with app.app_context():
        from . import routes
    
    return app