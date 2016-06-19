#!/usr/bin/env zsh

running=$(ps aux | grep "mongod " | grep -v grep)
if [ -z "$running" ]; then
  mongod --config /usr/local/etc/mongod.conf &
fi

tail -f /usr/local/var/log/mongodb/mongo.log
