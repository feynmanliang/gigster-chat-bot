import cherrypy

import json

from chatbot.bot import load_chatbot
from classifiers.marketplace_vs_social import load_classification_pipeline, CleanTextTransformer

class ModelServer(object):
    exposed = True

    messages = []
    clf = load_classification_pipeline()
    chatbot = load_chatbot()

    @cherrypy.tools.accept(media='text/plain')
    def GET(self, message):
        if message == 'RESET':
            ModelServer.messages = []
            response = 'Resetting conversation state.'
            pred = [0, 0]
        else:
            if not ModelServer.messages:
                ModelServer.messages = []
            ModelServer.messages.append(message)
            pred = ModelServer.clf.predict_proba(['\n'.join(ModelServer.messages)])[0].tolist()
            #response = 'What do you mean by: ' + message
            response = ModelServer.chatbot.get_response(message).text
        return json.dumps({
            'message': response,
            'predictions': pred
        })

def CORS():
    cherrypy.response.headers["Access-Control-Allow-Origin"] = "http://localhost:8081"

if __name__ == '__main__':
    conf = {
        '/': {
                    'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
                    'tools.CORS.on': True,
                }
    }
    cherrypy.tools.CORS = cherrypy.Tool('before_handler', CORS)
    cherrypy.quickstart(ModelServer(), '/', conf)
