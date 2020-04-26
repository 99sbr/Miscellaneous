from flask import Flask
from .apis import blueprint as api
from waitress import serve

app = Flask('Restplus API')
app.register_blueprint(api,url_prefix='/api/v1')
app.config["SWAGGER_UI_JSONEDITOR"]=True
app.run(port=1234,threaded=False)
# serve(app,port=1234)