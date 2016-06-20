#!/usr/bin/env python

from collections import namedtuple
from pprint import pprint
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.sales_ai

parsed_gigs_data = []
Gig = namedtuple('Gig', ['gigId', 'name', 'features', 'templates'])
def parse_gigs():
    "Parses gigs.json"
    for gig in db.gigs.find({ 'calculator': { '$exists': True } }):
        parsed_gig = Gig(gigId = str(gig['_id']),
			 name = gig['name'],
                         features = gig['calculator'].get('features', []),
                         templates = gig['calculator'].get('templates', []))
        if len(parsed_gig.features) + len(parsed_gig.templates) > 0:
            parsed_gigs_data.append(parsed_gig)
    return parsed_gigs_data

if __name__ == "__main__":
    print(parse_gigs()[:5])
