# ICDE-NRS

## References
- GCN: https://github.com/tkipf/pygcn
- Sentence-BERT: https://github.com/UKPLab/sentence-transformers
- IsoRankN: http://cb.csail.mit.edu/cb/mna/

## Usage
- `single_news_network.py` : construct interaction-edges 
- `run_isorank.py` : execute IsoRankN to construct inter-edges
- `cross_news_network6.py` : construct the proposed cross-media news network
- `gcn.py` : GCN model
- `recommendation.py` : personalized recommendations
- `user_network.py` : user modeling
- `utils.py` : some utils functions


## Data
http://acolab.ie.nthu.edu.tw/news_network/ICDE-db.rar

Due to the privacy issue, the user browsing history record is not attached.
You can sample your own browsing history record from the news data.


## Model
modified Sentence-BERT: http://acolab.ie.nthu.edu.tw/news_network/sbert-model.rar
