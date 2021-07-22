date
mkdir -p ~/DailyNews/Logs/lbt_$(date "+%Y-%m-%d") &
~/anaconda3/envs/NLP/bin/python ~/DailyNews/lbt.py >~/DailyNews/Logs/lbt_$(date "+%Y-%m-%d")/$(date "+%H:%M:%S").log &
date