import sqlite3
import datetime
import time
from tqdm import tqdm

def select_table(database_name, table_name, columns, conditions):
    conn = sqlite3.connect(database_name)
    cur = conn.cursor()
    myresult = cur.execute('''SELECT {1}
                            FROM {0} WHERE {2}'''
                           .format(table_name,columns,conditions) )
    result = []
    for row in myresult:
        result.append(row)
    conn.close()    
    return result


class SingleNewsNetwork():
    def __init__(self, media_name, DB_PATH, TODAY_DATE, NUM_DAYS):
        self.name = 'SingleNewsNetwork'
        self.media_name = media_name
        self.edge_table_name = self.media_name+'connection'
        self.node_table_name = self.media_name+'news'
        self.today_date = TODAY_DATE
        self.week_ago_date = (datetime.datetime.strptime(TODAY_DATE, "%Y-%m-%d") - datetime.timedelta(days=NUM_DAYS)).strftime("%Y-%m-%d")
        self.DB_PATH = DB_PATH
        print('\n\n[INFO] Construct %s single news network' %(media_name))
        
        
    def get_edges_from_db(self):
        print('[INFO] Getting %s edges from database...' %(self.media_name))
        edges = select_table(self.DB_PATH, 
                             self.edge_table_name, 
                             'News1, News2', 
                             "ParseDate>='%s' " %(self.week_ago_date))
        edges = list(set(edges))
        print('[INFO] Get %d edges which ParseDate>=%s' %(len(edges), self.week_ago_date))
        return edges
        
    def get_nodes_unique(self, edges):
        print('[INFO] Getting unique nodes...')
        nodes = list(set([k[0] for k in edges] + [k[1] for k in edges]))
        print('[INFO] Get %d nodes which ParseDate>=%s' %(len(nodes), self.week_ago_date))
        return nodes
    
    def filter_nodes(self, nodes):
        ''' 把日期太久以前的 node 刪掉 '''
        print('[INFO] Filtering nodes which NewsDate<=%s...' %(self.week_ago_date))
        filtered_nodes = []
        ''' 先把所有 NewsDate>=WEEKAGO 的新聞全部抓出來(assess比較快) '''
        available_nodes = select_table(self.DB_PATH, 
                                       self.node_table_name, 
                                       '*', 
                                       "NewsDate>='%s' and NewsDate!='None' " %(self.week_ago_date))
        available_nodes_NewsIndex = [k[0] for k in available_nodes]
#         print('[INFO] %d nodes later then %s' %(len(available_nodes), self.week_ago_date))
        for NewsIndex in nodes:
            if NewsIndex in available_nodes_NewsIndex:
                idx = available_nodes_NewsIndex.index(NewsIndex)
                filtered_nodes.append(available_nodes[idx])
        print('[INFO] Get %d filtered nodes which ParseDate>=%s' %(len(filtered_nodes), self.week_ago_date))
        return filtered_nodes
    
    def filter_edges(self, edges, filtered_nodes):
        ''' 把日期太久以前的 edge 刪掉 '''
        print('[INFO] Filtering edges which NewsDate<=%s...' %(self.week_ago_date))
        filtered_nodes_NewsIndex = [k[0] for k in filtered_nodes]
        filtered_edges = []
        for edge in edges:
            n1 = edge[0]
            n2 = edge[1]
            # 若兩個 node 都沒有被因為 "NewsDate太久以前" 被 filter掉，才保留 edge
            if n1 in filtered_nodes_NewsIndex and n2 in filtered_nodes_NewsIndex:
                if n1<n2:
                    filtered_edges.append((n1,n2))
                else:
                    filtered_edges.append((n2,n1))
        filtered_edges = list(set(filtered_edges))
        print('[INFO] Get %d filtered edges which ParseDate>=%s' %(len(filtered_edges), self.week_ago_date))
        return filtered_edges
    
    def construct_news_network(self):
        edges = self.get_edges_from_db()
        nodes = self.get_nodes_unique(edges)
        filtered_nodes = self.filter_nodes(nodes)
        filtered_edges = self.filter_edges(edges, filtered_nodes)
        return sorted(filtered_nodes, key=lambda x: x[0]), sorted(filtered_edges)