#!/usr/bin/env zsh

python -m nltk.downloader all
python -m spacy.en.download
./scripts/mongo_start.zsh &

cd chat_client && npm install
