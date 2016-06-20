import cherrypy

import sys
from os.path import dirname
sys.path.append(dirname(__file__))

from classifiers.marketplace_vs_social import load_classification_pipeline, CleanTextTransformer


class ModelServer(object):
    clf = load_classification_pipeline()
    messages = []

    @cherrypy.expose
    def predict(self, message):
        if message == 'RESET':
            ModelServer.messages = []
            return 'Resetting belief states'
        else:
            if not ModelServer.messages:
                ModelServer.messages = []
            ModelServer.messages.append(message)
            return str(ModelServer.clf), str(ModelServer.messages)

if __name__ == '__main__':
    from classifiers.marketplace_vs_social import load_classification_pipeline, CleanTextTransformer
    cherrypy.quickstart(ModelServer(), '/')
