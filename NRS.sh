#!/bin/sh

today_date1=$(date +'%Y-%m-%d')
today_date2=$(date +'%Y%m%d')
echo "$today_date1"

time_version=18
echo "$time_version"


python cross_news_network6.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db' --time_version ${time_version}
python gcn.py --today_date ${today_date1} --time_version ${time_version}









# python exp-cross_news_network6.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db'
# python exp-recommend_via_gcn.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db' --exp_w 0.1 --exp_d 1
# python exp-recommend_via_gcn.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db' --exp_w 0.01 --exp_d 1
# python exp-recommend_via_gcn.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db' --exp_w 0.5 --exp_d 1
# python exp-recommend_via_gcn.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db' --exp_w 0.1 --exp_d 3
# python exp-recommend_via_gcn.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db' --exp_w 0.1 --exp_d 5
# python exp-final_recommend_list.py --today_date ${today_date1} --db_path './db/NewsNetwork_ch_'${today_date2}'.db'