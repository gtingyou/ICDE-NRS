# ICDE-NRS

## References
- GCN: https://github.com/tkipf/pygcn
- Sentence-BERT: https://github.com/UKPLab/sentence-transformers

## Usage
- `cross_news_network6.py` : construct the proposed cross-media news network
  - `single_news_network.py` : construct interaction-edges 
  - `run_isorank.py` : execute IsoRankN to construct inter-edges
- `gcn.py` : GCN model
- `recommendation.py` : personalized recommendations
  - `user_network.py` : user modeling
- `utils.py` : some utils functions
