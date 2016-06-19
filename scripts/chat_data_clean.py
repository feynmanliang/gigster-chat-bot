#!/usr/bin/env python

from collections import namedtuple
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

if __name__ == "__main__":
    # prints 5 parsed chat messages
    chats_data = import_chats_data()
    chat_histories = parse_chats_messages(chats_data)
    for _ in range(5):
        pprint(next(chat_histories))

