curl -X POST http://141.61.33.12:8888/start_profile
python req.py prompt.txt 16
curl -X POST http://141.61.33.12:8888/stop_profile
