import os
import re
import uuid
import json

import geopandas as gpd

from markupsafe import escape

from flask import Response, request, jsonify, session, current_app as app

from .. import GraphInterface
from ... import utils, s3_utils


app.secret_key = "The session is unavailable because no secret key was set. Set the secret_key on the application to something unique and secret."     # DOC: ahahah 


# @app.before_request
# def assegna_session_id():
#     print(request.endpoint)
#     # DOC: Handle new GraphInterface session
#     if request.endpoint == 'start':
#         print('Create new graph interface session')
#         gi: GraphInterface = app.__GRAPH_REGISTRY__.register(
#             thread_id=str(uuid.uuid4()),
#             user_id='flask_usr_000',
#             project_id='project_000',
#             map_handler=None  # DOC: Default map handler, can be changed later
#         )
#         session["session_id"] = gi.thread_id
    
    
        

@app.route('/')
def index():
    return jsonify("Welcome to the SaferPlaces Agent Interface!"), 200


@app.route('/user', methods=['POST'])
def user():
    data = request.get_json(silent=True) or {}
    print(data)
    user_id = data.get('user_id', None)
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    user_bucket_files = s3_utils.list_s3_files(f's3://{os.getenv("BUCKET_NAME", "saferplaces.co")}/{os.getenv("BUCKET_OUT_DIR", "SaferPlaces-Agent/dev")}/user={user_id}')
    user_project = sorted(list(set([re.search(r'project=(dev-\d+)', p).group(1) for p in user_bucket_files if 'project=' in p])))
    
    return jsonify({
        "user_id": user_id,
        "projects": user_project
    }), 200
    

@app.route('/t', methods=['GET', 'POST'])
def start():
    if request.method == 'GET':
        gi: GraphInterface = app.__GRAPH_REGISTRY__.register(
            thread_id = str(uuid.uuid4()),
            user_id = 'flask_usr_000',
            project_id = 'project_000',
            map_handler = None
        )
        session["session_id"] = gi.thread_id
    
    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        thread_id = data.get('thread_id', None) or str(uuid.uuid4())
        
        gi: GraphInterface = app.__GRAPH_REGISTRY__.get(thread_id)
        if not gi:
            gi = app.__GRAPH_REGISTRY__.register(
                thread_id = thread_id,
                user_id = data.get('user_id', 'flask_usr_000'),
                project_id = data.get('project_id', 'project_000'),
                map_handler = None
            )
            
        session["session_id"] = gi.thread_id
        
    else:
        return jsonify({"error": f"Method {request.method} not allowed"}), 405
    
    return jsonify({
        "thread_id": gi.thread_id,
        "user_id": gi.user_id,
        "project_id": gi.project_id
    }), 200


@app.route('/t/<thread_id>', methods=['POST'])
def prompt(thread_id):
    
    gi: GraphInterface = app.__GRAPH_REGISTRY__.get(thread_id)
    if not gi:
        return jsonify({"error": "GraphInterface not found"}), 404
    
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    prompt = escape(data['prompt'])
    
    stream_mode = data.get('stream', False)
    
    if stream_mode:
        def generate():
            for e in gi.user_prompt(prompt=prompt, state_updates={'avaliable_tools': []}):
                yield json.dumps(gi.conversation_handler.chat2json(chat=e)) + "\n"
        
        return Response(generate(), mimetype='text/plain')
    
    else:
        gen = (
            gi.conversation_handler.chat2json(chat=e)
            for e in gi.user_prompt(prompt=prompt, state_updates={'avaliable_tools': []})
        )
    
        return jsonify(list(gen)), 200
    
    
@app.route('/t/<thread_id>/layers', methods=['POST'])
def layers(thread_id):
    gi: GraphInterface = app.__GRAPH_REGISTRY__.get(thread_id)
    if not gi:
        return jsonify({"error": "GraphInterface not found"}), 404
    
    layers = gi.get_state('layer_registry', [])
    
    return jsonify(layers), 200


@app.route('/render', methods=['POST'])
def render_layer():
    data = request.get_json(silent=True) or {}
    layer_src = data.get('src', None)
    
    if not layer_src:
        return jsonify({"error": "Layer source is required"}), 400
    
    layer_type = data.get('type', None)
    if not layer_type:
        return jsonify({"error": "Layer type is required"}), 400
    
    if layer_type == 'raster':
        return jsonify({"error": "Raster layer rendering is not implemented yet"}), 501
    
    elif layer_type == 'vector':
        layer_render_src = utils.vector_to_geojson4326(layer_src)
            
    else:
        return jsonify({"error": f"Layer type '{layer_type}' is not supported"}), 400
        
    return jsonify({'src': layer_render_src}), 200