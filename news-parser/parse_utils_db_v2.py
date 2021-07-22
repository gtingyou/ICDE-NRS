import sqlite3
import time

def create_nodes_table(database_name, table_name):
    con = sqlite3.connect(database_name)
    con.execute('''CREATE TABLE {0}
                 (NewsIndex TEXT,
                 ParseDate TEXT,
                 NewsURL TEXT,
                 NewsDate TEXT,
                 NewsTitle TEXT,
                 NewsContext TEXT);'''
                 .format(table_name) )
    con.close()
    
def create_edges_table(database_name, table_name):
    con = sqlite3.connect(database_name)
    con.execute('''CREATE TABLE {0}
                 (News1 TEXT,
                 News2 TEXT,
                 ParseDate TEXT);'''
                 .format(table_name) )
    con.close()

def select_nodes_table(database_name, table_name, columns, conditions):
    con = sqlite3.connect(database_name)
    cur = con.cursor()
    myresult = cur.execute('''SELECT {1}
                            FROM {0} WHERE {2}'''
                           .format(table_name,columns,conditions) )
    result = []
    for row in myresult:
        result.append(row)
    con.close()    
    return result

def select_edges_table(database_name, table_name, columns, conditions):
    con = sqlite3.connect(database_name)
    cur = con.cursor()
    myresult = cur.execute('''SELECT {1}
                            FROM {0} WHERE {2}'''
                           .format(table_name,columns,conditions) )
    result = []
    for row in myresult:
        result.append(row)
    con.close()        
    return result

def insert_nodes_table(database_name, table_name, nodes):
    con = sqlite3.connect(database_name)
    cur = con.cursor()
    for n in nodes:
        cur.execute('''INSERT INTO {0} (NewsIndex,ParseDate,NewsURL,NewsDate,NewsTitle,NewsContext) VALUES (?,?,?,?,?,?)'''.format(table_name), n)
    con.commit()
    con.close()

def insert_edges_table(database_name, table_name, edges):
    con = sqlite3.connect(database_name)
    cur = con.cursor()
    for e in edges:
        cur.execute('''INSERT INTO {0} (News1,News2,ParseDate) VALUES (?,?,?)'''.format(table_name), e)
    con.commit()
    con.close()


if __name__ == '__main__':
    database_name = './NewsNetwork_ch.db'
    MEDIA = ['CHT','LBT']
    for media_name in MEDIA:
        print(media_name)
        table_name1 = media_name+'news'
        table_name2 = media_name+'connection'
        create_nodes_table(database_name, table_name1)
        create_edges_table(database_name, table_name2)