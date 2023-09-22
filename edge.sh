#!/bin/sh


nohup python3 -u service_demo.py > service.log 2>&1 &

sleep 3

echo 'service on edge is initialized'

nohup python3 -u job_manager.py > job.log 2>&1 &

sleep 3

echo 'job manager on edge is initialized'

nohup python3 -u camera_simulation.py > camera.log 2>&1 &

sleep 3

echo 'camera on edge is initialized'

# shellcheck disable=SC2164
cd expr

nohup python3 -u headup_detect_delay_test_new.py > test.log 2>&1 &

sleep 3

echo 'test script on edge is initialized'



