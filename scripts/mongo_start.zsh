#!/usr/bin/env zsh
#
if [ "$(ps aux | grep "mongod " | awk '{print $11}')" == "grep" ]; then
  mongod --config /usr/local/etc/mongod.conf &
fi

tail -f /usr/local/var/log/mongodb/mongo.log
