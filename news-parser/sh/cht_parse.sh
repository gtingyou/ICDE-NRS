date
mkdir -p ~/DailyNews/Logs/cht_$(date "+%Y-%m-%d") &
~/anaconda3/envs/NLP/bin/python ~/DailyNews/cht.py >~/DailyNews/Logs/cht_$(date "+%Y-%m-%d")/$(date "+%H:%M:%S").log &
date