import cherrypy

import json
from classifiers.marketplace_vs_social import load_classification_pipeline, CleanTextTransformer

class ModelServer(object):
    clf = load_classification_pipeline()
    messages = []

    @cherrypy.expose
    def predict(self, message):
        if message == 'RESET':
            ModelServer.messages = []
            response = 'Resetting conversation state.'
            pred = [0, 0]
        else:
            if not ModelServer.messages:
                ModelServer.messages = []
            ModelServer.messages.append(message)
            pred = ModelServer.clf.predict_proba(['\n'.join(ModelServer.messages)])[0].tolist()
            response = 'What do you mean by: ' + message
        return json.dumps({
            'message': response,
            'predictions': pred
        })

if __name__ == '__main__':
    from classifiers.marketplace_vs_social import load_classification_pipeline, CleanTextTransformer
    cherrypy.quickstart(ModelServer(), '/')
