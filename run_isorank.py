import os
import subprocess
import datetime


class IsoRank():
    def __init__(self, TODAY_DATE, ALLnodes, sparse_document_matrix):
        self.name = 'IsoRank'
        self.current_path = os.getcwd()
        self.output_folder_name = 'IsoRank-' + TODAY_DATE
        
        self.output_path = self.current_path + '/IsoRank/' + self.output_folder_name
        print('[INFO] IsoRank files path %s' %(self.output_path))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        self.dense_edges = self.sparse_matrix_to_dense_edges(ALLnodes, sparse_document_matrix)

    def sparse_matrix_to_dense_edges(self, ALLnodes, sparse_matrix):
        matrix_index = [k[0] for k in ALLnodes]
        data = sparse_matrix.data
        indptr = sparse_matrix.indptr
        indices = sparse_matrix.indices
        dense_edges = []
        for i in range(sparse_matrix.shape[0]):
            # 第 i 列
            d = data[indptr[i]: indptr[i+1]] # 第 i 列所有非零元素
            col = indices[indptr[i]:indptr[i+1]] # 第 i 列所有非零元素的 行index
            for idx, j in enumerate(col):
                # 第 i 列，第 j 行
                if i<=j: # symmetric matrix，只取上半三角形
                    dense_edges.append([matrix_index[i], matrix_index[j], d[idx]])
        return sorted(dense_edges)
            
    def create_evals_files(self, media_name1, media_name2):
        # 按照 A-Z 排序 media names
        names = sorted([media_name1, media_name2])
        media_name1 = names[0]
        media_name2 = names[1]
        
        dense_edges = self.dense_edges
        
        media11 = [k for k in dense_edges if media_name1 in k[0] and media_name1 in k[1]]
        self.output_eval_file(media_name1, media_name1, media11)
        
        media22 = [k for k in dense_edges if media_name2 in k[0] and media_name2 in k[1]]
        self.output_eval_file(media_name2, media_name2, media22)
        
        media12 = [k for k in dense_edges if media_name1 in k[0] and media_name2 in k[1]]
        self.output_eval_file(media_name1, media_name2, media12)
        
    def output_eval_file(self, media_name1, media_name2, similarity_score):
#         if os.path.exists("%s/%s-%s_news.evals" %(self.output_path, media_name1, media_name2)): return 
        print('[INFO] Output %s/%s-%s_news.evals' %(self.output_path, media_name1, media_name2) )
        f = open("%s/%s-%s_news.evals" %(self.output_path, media_name1, media_name2), "w")
        for s in similarity_score:
            f.write(str(s[0]) + "    " + str(s[1]) + "    " + str(s[2]) + "\n")
        f.close()
        
    def create_tab_files(self, media_name, edges):
#         if not os.path.exists('%s/%s_news.tab' %(self.output_path, media_name)):
        print('[INFO] Output %s/%s_news.tab' %(self.output_path, media_name) )
        f = open('%s/%s_news.tab' %(self.output_path, media_name), "w")
        f.write("INTERACTOR_A    INTERACTOR_B" + "\n")
        for e in edges:
            f.write(str(e[0]) + "    " + str(e[1]) + "\n")
        f.close()
    
    def create_inp_file(self, media_names):
        print('[INFO] Output %s/three_sc.inp' %(self.output_path) )
        f = open('%s/three_sc.inp' %(self.output_path), "w")
        f.write("%s\n" %(self.output_path) )
        f.write("_news\n")
        f.write("%s\n" %(str(len(media_names))) )
        for i in range( len(media_names) ):
            f.write("%s\n" %(media_names[i]) )
        f.close()
        
    def create_sh_file(self, ALPHA):
        command = '''date
%s/isorankn_src_NRS.exe --K 5 --threadid 6 --thresh 1e-3 --alpha %s --maxveclen 100000 -- %s/three_sc.inp
date''' %(self.current_path, str(ALPHA), self.output_path)
        print('[INFO] Output %s/run_isorankn.sh' %(self.output_path) )
        f = open('%s/run_isorankn.sh' %(self.output_path), "w")
        f.write(command)
        f.close()
        
    def run_isorank_exe(self):
        print('[INFO] Running IsoRank...')
        subprocess.call(['sh', 
                         self.output_path + '/run_isorankn.sh'])
        subprocess.call(['mv', 
                         self.current_path + '/tmp_match-score.txt', 
                         self.output_path + '/tmp_match-score.txt' ])
#         subprocess.call(['mv', 
#                          self.current_path + '/tmp_output_cluster.txt', 
#                          self.output_path + '/tmp_output_cluster.txt' ])



# if __name__ == '__main__':
    # isorank = IsoRank()

    # isorank.create_evals_files('CHT', 'LBT', ALLnodes, sparse_document_matrix)
    # isorank.create_tab_files('CHT', CHTedges)
    # isorank.create_tab_files('LBT', LBTedges)
    # isorank.create_inp_file(['CHT', 'LBT'])
    # isorank.create_sh_file(ALPHA=0.1)

    # isorank.run_isorank_exe()