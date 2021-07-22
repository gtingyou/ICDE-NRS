date
mkdir -p ~/DailyNews/Logs/setn_$(date "+%Y-%m-%d") &
~/anaconda3/envs/NLP/bin/python ~/DailyNews/setn.py >~/DailyNews/Logs/setn_$(date "+%Y-%m-%d")/$(date "+%H:%M:%S").log &
date