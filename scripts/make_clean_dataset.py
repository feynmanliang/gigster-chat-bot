#!/usr/bin/env python

from collections import namedtuple
import pickle
from pprint import pprint

from scripts.chat_data_clean import pickle_chat_histories
from scripts.gig_data_clean import parse_gigs

GigInstance = namedtuple('GigInstance', ['gigId', 'messages', 'features', 'templates'])
def make_clean_dataset(fp = 'data/clean_dataset.pkl', chats_pkl_path = 'data/chat_histories.pkl'):
    with open(chats_pkl_path, 'rb') as fd:
        try:
            chats = pickle.load(fd)
        except EOFError:
            print('Chat histories not found at {}, regenerating'.format(chats_pkl_path))
            pickle_chat_histories(chats_pkl_path)
            chats = pickle.load(fd)
    gigs = parse_gigs()

    # convert to dicts keyed by gigId for hash-join
    chats = dict(map(lambda x: (x.gigId, x), chats))
    gigs = dict(map(lambda x: (x.gigId, x), gigs))

    # only consider gigs which have both:
    #  (1) a chat history, filtered in scripts/chat_data_clean
    #  (2) at least one feature or template, filtered in scripts/gig_data_clean
    valid_gigIds = set(chats.keys()) & set(gigs.keys())

    dataset = []
    for id in valid_gigIds:
        gigInstance = GigInstance(gigId = id,
                                  messages = chats[id].messages,
                                  features = gigs[id].features,
                                  templates = gigs[id].templates)
        dataset.append(gigInstance)

    # dump dataset
    with open(fp, 'wb') as out_fd:
        pickle.dump(dataset, out_fd)

if __name__ == '__main__':
    make_clean_dataset()
