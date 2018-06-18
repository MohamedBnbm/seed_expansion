from funcs import *


dic_seeds = {'politique':['userId:19438626','userId:11694252','userId:266215308'],
                'f1':['userId:138041655','userId:213969309','userId:28297965'],
                'ai':['userId:33836629','userId:48008938','userId:471550563']}


k = 5
unwanted_lists = biglists(500)
for champ in ['ai2']:
    g = graph(0, K = k, champ = champ, dic_champs = dic_seeds, unwanted_lists = unwanted_lists)
    g.totxt()
    start = time.clock()
    dc, dlc = g.louvain_communities()
    cl = g.clusters_kws()
