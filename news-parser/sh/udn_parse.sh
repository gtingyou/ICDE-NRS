date
mkdir -p ~/DailyNews/Logs/udn_$(date "+%Y-%m-%d") &
~/anaconda3/envs/NLP/bin/python ~/DailyNews/udn.py >~/DailyNews/Logs/udn_$(date "+%Y-%m-%d")/$(date "+%H:%M:%S").log &
date
