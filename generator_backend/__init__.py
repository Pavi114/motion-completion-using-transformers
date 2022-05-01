import json
import os

from flask import Flask, request
from flask_cors import CORS

from constants import DEVICE
from generator_backend.model import Model

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    CORS(app)

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    model = Model('2e_2d_2h_30k_linear')

    print(model)

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return f'Hello, World! {DEVICE}'

    @app.post('/generate')
    def generate():
        print("Generate called")

        body = json.loads(request.data)

        z_x, in_x = model.generate(body['gpos'])
        
        res = {
            'z_x': z_x,
            'in_x': in_x
        }

        return json.dumps(res)

    return app