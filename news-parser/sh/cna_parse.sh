date
mkdir -p ~/DailyNews/Logs/cna_$(date "+%Y-%m-%d") &
~/anaconda3/envs/NLP/bin/python ~/DailyNews/cna.py >~/DailyNews/Logs/cna_$(date "+%Y-%m-%d")/$(date "+%H:%M:%S").log &
date