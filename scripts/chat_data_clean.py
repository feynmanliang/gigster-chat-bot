#!/usr/bin/env python

CHATS_PKL_PATH = 'data/chat_histories.pkl'

from collections import namedtuple
import pickle
import json
from jsonpath_rw import jsonpath, parse
from pprint import pprint

def import_chats_data():
    with open('./data/chats.json') as fp:
        chats_data = json.load(fp)
    return chats_data

ChatHistory = namedtuple('ChatHistory', ['gigId', 'messages'])
ParsedMessage = namedtuple('ParsedMessage', ['text', 'toClient'])

def parse_chats_messages(chats_data):
    for gigId, raw_messages in chats_data['messages'].items():
        parsed_chat_history = []
        for match in parse('* . text').find(raw_messages):
            raw_message = match.context.value
            parsed_message = ParsedMessage(raw_message['text'], raw_message['toClient'])
            parsed_chat_history.append(parsed_message)
        if len(parsed_chat_history) == 0:
            continue
        else:
            yield ChatHistory(gigId, parsed_chat_history)

def pickle_chat_histories():
    """Serializes chat_histories to disk."""
    chats_data = import_chats_data()
    chat_histories = parse_chats_messages(chats_data)
    with open(CHATS_PKL_PATH, 'wb') as out_fd:
        pickle.dump(list(chat_histories), out_fd)

def load_chat_histories():
    with open(CHATS_PKL_PATH, 'rb') as fd:
        try:
            chats = pickle.load(fd)
        except FileNotFoundError:
            print('Chat histories not found at {}, regenerating'.format(CHATS_PKL_PATH))
            pickle_chat_histories(CHATS_PKL_PATH)
            chats = pickle.load(fd)
    return chats

if __name__ == "__main__":
    pickle_chat_histories()

