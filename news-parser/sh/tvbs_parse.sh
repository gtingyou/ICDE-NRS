date
mkdir -p ~/DailyNews/Logs/tvbs_$(date "+%Y-%m-%d") &
~/anaconda3/envs/NLP/bin/python ~/DailyNews/tvbs.py >~/DailyNews/Logs/tvbs_$(date "+%Y-%m-%d")/$(date "+%H:%M:%S").log &
date