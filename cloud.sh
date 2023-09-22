#!/bin/sh

unset LD_LIBRARY_PATH

nohup python3.8 -u service_demo.py > service.log 2>&1 &

sleep 3

echo 'service on cloud is initialized'

nohup python3.8 -u query_manager.py > query.log 2>&1 &

sleep 3

echo 'query manager on cloud is initialized'