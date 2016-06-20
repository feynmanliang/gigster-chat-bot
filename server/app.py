from flask import Flask
app = Flask(__name__)

@app.route('/analyze')
def hello_world():
    return 'Hello world!'
