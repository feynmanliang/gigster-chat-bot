#!/usr/bin/env python

import pickle

from chatterbot import ChatBot
from chatterbot.training.trainers import ListTrainer, ChatterBotCorpusTrainer

from scripts.chat_data_clean import load_chat_histories

CHATBOT_PKL_PATH = 'chatbot/model.pkl'

def load_chatbot():
    try:
        chatbot = pickle.load(open(CHATBOT_PKL_PATH, 'rb'))
    except FileNotFoundError:
        print('Chatbot not found at {}, regenerating'.format(CHATBOT_PKL_PATH))
        chatbot = ChatBot("Machine Salesman")

        chatbot.set_trainer(ChatterBotCorpusTrainer)
        chatbot.train('chatterbot.corpus.english')

        # This is taking too long... skip for now
        # chatbot.set_trainer(ListTrainer)
        # for chat in load_chat_histories():
        #     chatbot.train([message.text for message in chat.messages])

        pickle.dump(chatbot, open(CHATBOT_PKL_PATH, 'wb'))
    return chatbot

if __name__ == '__main__':
    chatbot = load_chatbot()
