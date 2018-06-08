import time
import os
import numpy as np
import elasticsearch as es
from elasticsearch import Elasticsearch
from gensim.summarization import keywords
from collections import Counter
import gexf
import redis
from igraph import *



'''
Main class representing similarity graph extracted from seeds
    
    Attributes: integer k : min # common lists to create edge
                string encoding the desired output in the dictionary of seeds
                list of seeds
                dictionary of edges with valences
                set of vertices ids
                dictionary of frequency of visits of each vertice
                dictionary of Louvain communities membership
                dictionary of list ids of each community
                dictionary of keywords of lists in each community
                dictionary of information about vertices (frequency, name, cluster membership)
    
    Methods:__init__ : Either connects to redis to grow the graph from the seeds or reads
                        a precomputed graph
            totxt : writes the edges of the graph and their valences (dict of dicts)
                    writes the frequency of visits of each vertice by the random walk
            louvain_communities : Either computes the communities using Louvain algo and
                            connect to redis to grab list ids in communities or reads
                            precomputed files
            clusters_kws : Connects to elasticsearch and calculate word frequency inside
                            each cluster
            to_gexf : export graph in gexf format
            get_users_info : connects to elasticsearch to get screennames of vertices and
                            summarizes information about users in gaph
'''
class graph:
    K = 0
    champ = ''
    seeds = []
    edges = {}
    vertices = set()
    vertices_frequency = {}
    vertices_communities = {}
    communities_lists = {}
    communities_kws = {}
    user_names_info = {}


    def __init__(self, *args, **kwargs):
        K = kwargs['K']
        champ = kwargs['champ']
        dic_champs= kwargs['dic_champs']
        unwanted_lists = kwargs['unwanted_lists']
        self.K = K
        self.champ = champ
        seeds = dic_champs[champ]
        self.seeds = seeds
        if args[0] == 0:
            print('Initialisation {} : {} seeds k = {}'.format(champ,len(seeds),K))
            reduced_graphs = {}
            nodes_intensity = {}
            for seed in seeds:
                print('Starting with {}'.format(seed))
                d, dc1 = get_corona(seed, unwanted_lists, K)
                d2 = get_corona_2(dc1, unwanted_lists, K)
                final_dict, dic, dic_i = merge_dict(d, d2, seed)
                size = len(final_dict.keys())
                matrix = build_matrix(final_dict, size)
                rp, dic_paths = get_paths(matrix, dic_i[seed[7:]], dic)
                rg = reduced_graph(dic_paths.keys(), final_dict, dic)
                reduced_graphs[seed] = rg
                nodes_intensity[seed] = dic_paths
            
            self.edges = merge_graphs(reduced_graphs, seeds)
            #self.vertices = final_graph.keys()
            final_intensities = merge_intensity(nodes_intensity, seeds)
            self.vertices = set(final_intensities.keys())
            self.vertices_frequency = final_intensities
        else:
            print('Reading graph {} : k = {}'.format(champ, K))
            self.edges, self.vertices_frequency = read_graph_freq(champ, K)
            self.vertices = set(self.edges.keys())

    def totxt(self):
        write_dict(self.edges, self.vertices_frequency, self.champ, self.K)

    def louvain_communities(self, done = False):
        if done:
            dic_clusters, dic_lists_clusters = read_com(self.champ, self.K)
        else:
            l_edges, weights = edgelist(self.edges)
            l_vertices = self.vertices
            l_vertices = list(set(l_vertices))
            l_vertices = sorted([int(t) for t in l_vertices])
            l_vertices = [str(t) for t in l_vertices]
            #print('1')
            g = Graph(directed = False)
            g.add_vertices(l_vertices)
            #print('2')
            g.add_edges(l_edges)
            g.es['weight'] = weights
            #print('3')
            p = g.community_multilevel(weights = weights)
            dic_lists_clusters = {}
            dic_clusters = {}
            r = redis.StrictRedis(host='localhost')
            for i,cl in enumerate(p):
                lists = set()
                ids = set()
                for v in cl:
                    v = g.vs[v]['name']
                    ids.add(v)
                    for listname in r.smembers('userId:' + v):
                        lists.add(listname.decode('utf-8'))
                dic_lists_clusters[i] = lists
                dic_clusters[i] = ids

            write_clusters(dic_clusters, dic_lists_clusters, self.champ, self.K)
            #print(g.modularity(p))
        self.vertices_communities = dic_clusters
        self.communities_lists = dic_lists_clusters
        return dic_clusters, dic_lists_clusters
        #print([g.vs[t]['name'] for t in p[0]], p[1])
        #print(q)
        #print(g.vcount())
        #print(g.ecount())

    def clusters_kws(self, done = False):
        if done:
            c_kws = read_kws(self.champ, self.K)
        else:
            es_t = Elasticsearch([{'local': 'localhost'}])
            c_kws = {}
            d_names = {}
            for k, v in self.communities_lists.items():
                #maybe a list to take into account # of occurences
                names = set()
                #print(str(k))
                for l in v:
                    nm = get_list_name(es_t,l)
                    names.add(nm)
                d_names[k] = names
                txt = ' '.join(list(names))
                wds = min(5,len(names))
                #print('--------------doing kws')
                #kws = keywords(txt, words = wds)
                ws = txt.split(' ')
                unwanted_words = ['/','et','&', '-', '***NoListName***', 'de', ',', '_', '.', 'the', 'of', 'les', 'le', 'la', 'and', 'list','my']
                ws = [x.strip() for x in ws if len(x)>0]
                ws = [x.lower() for x in ws]
                ws = [x for x in ws if x not in unwanted_words]
                kws = Counter(ws).most_common(wds)
                print(kws)
                print('\n')
                c_kws[k] = kws
                #print('--------------done with kws')
                #c_kws[k] = [t.encode('ascii', 'ignore') for t in kws.split('\n')]
            write_names(d_names, self.champ, self.K)
            write_kws(c_kws, self.champ, self.K)
        self.communities_kws = c_kws
        return c_kws

    def to_gexf(self):
        edges = set()
        champ = self.champ
        k = self.K
        gexf_file = gexf.Gexf('Graph similarity',champ + str(k))
        gph = gexf_file.addGraph('gr',"undirected", "static")
        intensAtt = gph.addNodeAttribute("intensity","0.0","float")
        clussAtt = gph.addNodeAttribute("cluster","100","int")

        with open('../data/graph_{}_{}/user_names_info.txt'.format(champ, str(k)), 'r') as f:
            for line in f:
                line = line[:-1]
                line = line.split(',')
                n = gph.addNode(line[0], line[1])
                #, attributes={'intensity':line[2], 'cluster':line[3]})
                n.addAttribute(intensAtt, line[2])
                n.addAttribute(clussAtt, line[3])

        with open('../data/graph_{}_{}/graph.txt'.format(champ,str(k)), 'r') as f:
            eid = 0
            for line in f:
                line = line[:-1]
                line = line.split(';')
                u1 = line[0]
                for u2, val in [el.split(',') for el in line [1:]]:
                    if (u1, u2) not in edges and (u2, u1) not in edges:
                        eid += 1
                        edges.add((u1, u2))
                        gph.addEdge(eid, u1, u2, weight=val)

        file = open('./graph_{}_{}/graphe.gexf'.format(champ, str(k)), 'wb')
        gexf_file.write(file)

    def get_users_info(self, done=False):
        if done:
            dic_names_info = read_user_info(self.champ, self.K)
        else:
            dic_names_freq = {}
            es_t = Elasticsearch([{'local': 'localhost'}])
            for k, v in self.vertices_frequency.items():
                try:
                    name = es_t.search(index='twitter', body={'query': {'match': {'userId': k}}})['hits']['hits'][0]['_source']['screenName'].encode('ascii', 'ignore')
                except:
                    name = '**noName**'
                dic_names_freq[k] = [name, v]
            dic_names_cluster = {}
            for k, v in self.vertices_communities.items():
                for u in v:
                    dic_names_cluster[u] = k
            dic_names_info = {}
            for k, v in dic_names_freq.items():
                dic_names_info[k] = v + [dic_names_cluster[k]]
        self.user_names_info = dic_names_info
        write_user_info(dic_names_info, self.champ, self.K)
        return dic_names_info

    def print_cluster_info(self):
        for k in self.vertices_communities.keys():
            kws = ''
            for kw in self.communities_kws[k]:
                kws = kws + ',' + kw
            print(str(k) + ',' + str(len(self.vertices_communities[k])) + kws)

    def get_desired_vertices(self, d_c):
        directory = '../data/graph_{}_{}/'.format(self.champ, self.K)
        with open(directory + 'desired_vertices.txt', 'w') as f:
            for k,v in self.vertices_communities.items():
                if k in d_c:
                    for uid in v:
                        f.write(uid + '\n')



def get_vertices_lists(champ, k, biglist):
    users = set()
    directory = '../data/graph_{}_{}/'.format(champ,k)
    with open(directory + 'desired_vertices.txt', 'r') as f:
        for line in f:
            users.add(line[:-1])
    r = redis.StrictRedis(host='localhost')
    dic_users_lists = lists_of_users(users, r, biglist)
    with open(directory + 'vertices_lists.txt', 'w') as f:
        for k,v in dic_users_lists.items():
            f.write(k)
            for li in v:
                f.write(',' + li)
            f.write('\n')

# Write file with frequencies of visit, screenames and clusters of users
def write_user_info(dico, champ, k):
    directory = '../data/graph_{}_{}/'.format(champ, k)
    with open(directory + 'user_names_info.txt', 'w') as f:
        for k, v in dico.items():
            f.write(k + ',' + v[0] + ',' + str(v[1]) + ',' + str(v[2]) + '\n')


# Reads info about users and stores it in dictionary of lists
def read_user_info(champ, k):
    dic_names_info = {}
    directory = '../data/graph_{}_{}/'.format(champ, k)
    try:
        with open(directory + 'user_names_info.txt', 'r') as f:
            for line in f:
                line = line[:-1]
                line = line.split(',')
                dic_names_info[line[0]] = [line[1], line[2], line[3]]
        return dic_names_info
    except:
        print('No such file user_names_info')
        return False


# Write names of lists of users by clusters
def write_names(dic, champ, k):
    directory = '../data/graph_{}_{}/'.format(champ, k)
    with open(directory + 'cluster_names.txt', 'w') as f:
        for k, v in dic.items():
            f.write(str(k))
            for name in v:
                f.write(',' + name)
            f.write('\n')


# Query elasticsearch to get listname
def get_list_name(es_con, listid):
    try:
        name = es_con.search(index='twitter', body={"query": {"match": {'listId':listid}}})['hits']['hits'][0]['_source']['listName'].encode('ascii', 'ignore')
    except:
        name = '***NoListName***'
    if name != '':
        if name[-1] == '\n':
            name = name[:-1]
    return name


# Write keywords of listnames by clusters
def write_kws(dic, champ, k):
    directory = '../data/graph_{}_{}/'.format(champ, k)
    with open(directory + 'clusters_kws.txt', 'w') as f:
        for k, v in dic.items():
            f.write(str(k))
            for kw in v:
                f.write(',' + kw[0])
            f.write('\n')


# Read file of keywords of clusters and store in dictionary
def read_kws(champ, k):
    c_kws = {}
    directory = '../data/graph_{}_{}/'.format(champ, k)
    try:
        with open(directory + 'clusters_kws.txt', 'r') as f:
            for line in f:
                line = line.split(',')
                kws = []
                for kw in line[1:]:
                    if kw[-1] == '\n':
                        kw = kw[:-1]
                    kws.append(kw)
                c_kws[int(line[0])] = kws
        return c_kws
    except:
        print('No such file')
        return False


# Writes file with clusters membership of users 
#        file with lists occuring in each cluster
def write_clusters(dic_cl, dic_l_cl, champ, k):
    directory = '../data/graph_{}_{}/'.format(champ, k)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + 'clusters.txt', 'w') as f:
        for k, v in dic_cl.items():
            f.write(str(k))
            for u in v:
                f.write(',' + u)
            f.write('\n')
    with open(directory + 'lists_clusters.txt', 'w') as f:
        for k, v in dic_l_cl.items():
            f.write(str(k))
            for u in v:
                f.write(',' + u)
            f.write('\n')


# Returns a set of tuples representing edges and a lists of corresponding weights
def edgelist(dic_edges):
    el = set()
    w = []
    for k, v in dic_edges.items():
        #print('0')
        for n, val in v.items():
            if tuple([k, n]) not in el and tuple([n, k]) not in el:
                el.add(tuple([k, n]))
                w.append(val)
    return el, w


# Write a file representing the graph and the frequency of visits of each vertice
def write_dict(dic, dic_intensity, champ, k):
    directory = '../data/graph_{}_{}/'.format(champ, k)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + 'graph.txt', 'w') as f:
        for key, val in dic.items():
            f.write(str(key))
            for k, v in val.items():
                f.write(';' + str(k) + ',' + str(v))
            f.write('\n')
    with open(directory + 'intensity.txt', 'w') as f:
        for key, val in dic_intensity.items():
            f.write(str(key) + ',' + str(val) + '\n')


# Returns aggregate dictionary containing the graphs obtained from each seed
def merge_graphs(dic_graphs, seeds):
    full_dic = dic_graphs[seeds[0]].copy()
    for _ in range(1, len(seeds)):
        for user, dic_neis in dic_graphs[seeds[_]].items():
            if user in full_dic.keys():
                full_dic[user] = dict(set(full_dic[user].items()) | set(dic_neis.items()))
            else:
                full_dic[user] = dic_neis
    return full_dic


# Aggregation of frequency of visits of vertices starting from the different seeds
def merge_intensity(dic_imp,seeds):
    full_dic = dic_imp[seeds[0]].copy()
    for _ in range(1,len(seeds)):
        for user, imp in dic_imp[seeds[_]].items():
            if user in full_dic.keys():
                full_dic[user] += imp
            else:
                full_dic[user] = imp
    return full_dic
        

# Filters out lists containing more than 'max_members' users
def biglists(max_members):
    big_lists = set()
    with open('../lists_cardinality.csv','r') as f:
        next(f)
        for line in f:
            line = line.split(',')
            if int(line[2][:-1]) > max_members:
                big_lists.add(line[1][7:])
    return big_lists


# Return list ids containing given user 
def lists_of_user(user, r_con, biglists):
    print('Getting lists of seed.')
    lists = set(r_con.smembers(user))
    return {t.decode('utf-8') for t in lists if t.decode('utf-8') not in biglists}


# Return user ids belonging to a set of lists
def users_in_set_of_lists(lists, r_con, n):
    print('Getting users belonging to set of lists.')
    pipe = r_con.pipeline()
    if n == 1:
        for l in lists:
            pipe.smembers('listId:' + l)
    else:
        for l in lists:
            pipe.smembers('listId:' + l.decode('utf-8'))
    users = {j for i in pipe.execute() for j in i}
    return {t.decode('utf-8') for t in users}


# Return list ids containing each user of given set
def lists_of_users(users, r_con, biglists):
    print('Getting lists of set of users.')
    pipe = r_con.pipeline()
    list_users = []
    for u in users:
        list_users.append(u)
        pipe.smembers('userId:'+u)
    list_lists = pipe.execute()
    dic_corona_lists = {}
    for i,n in enumerate(list_users):
        dic_corona_lists[n] = set([t for t in list_lists[i] if t.decode('utf-8') not in biglists])
    return dic_corona_lists


# Forces the symmetry of a dictionary of edges
def sym_dict(dic):
    dico = {}
    for k,v in dic.items():
        for user,val in v.items():
            if k not in dico.keys():
                dico[k] = {}
            dico[k][user] = val
            if user not in dico.keys():
                dico[user] = {}
            dico[user][k] = val
    return dico            


# Computes intersections between sets of lists to get edges and valences
def intersection_pairs_users(dic_users,k):
    print('Calculating common lists between users.')
    dic_corona_links = {}
    for u, lu in dic_users.items():
        dic_nei_vals = {}
        for v, lv in dic_users.items():
            if v != u:
                inter = len(lu & lv)
                if inter >= k:
                    dic_nei_vals[v] = inter
        if dic_nei_vals != {}:
            dic_corona_links[u] = dic_nei_vals
    return sym_dict(dic_corona_links)


# Computes intersections between sets of lists to get edges and weights of the graph
def intersection_pairs_users_2(dic_users_1, dic_users_2, k):
    print('Calculating common lists between users.')
    dic_corona_links = {}
    for u, lu in dic_users_1.items():
        dic_nei_vals = {}
        for v, lv in dic_users_2.items():
            inter = len(lu & lv)
            if inter >= k:
                dic_nei_vals[v] = inter
        if dic_nei_vals != {}:
            dic_corona_links[u] = dic_nei_vals
    return sym_dict(dic_corona_links)


# Main function to get neighbours of a seed and connections between them
def get_corona(seed, blists, k):
    print('Corona I: ')
    START = time.clock()
    r = redis.StrictRedis(host='localhost')
    starting_lists = lists_of_user(seed, r, blists)
    #print('Seed is in {} lists'.format(len(starting_lists)))
    users_corona = users_in_set_of_lists(starting_lists, r, 1)
    #print('N users in corona and start : {}'.format(len(users_corona)))
    dic_corona_lists = lists_of_users(users_corona, r, blists)
    dic_corona_links = intersection_pairs_users(dic_corona_lists,k)
    print('Getting corona I took {} seconds\n'.format(time.clock()-START))
    try:
        del dic_corona_lists[seed[7:]]
    except:
        print("seed not in corona lists")
    print("test 1", len(dic_corona_links))
    print('Is seed in nodes of graph corona I : ',seed[7:] in dic_corona_links)
    return dic_corona_links, dic_corona_lists


# Main function to get neighbours of the neighbours of the seeds
def get_corona_2(lists_corona_1, blists, k):
    print('Corona II: ')
    START = time.clock()
    users_corona_1 = set(lists_corona_1.keys())
    starting_lists = set()
    for val in lists_corona_1.values():
        starting_lists = starting_lists | val
    r = redis.StrictRedis(host='localhost')
    users_corona = users_in_set_of_lists(starting_lists, r, 2)
    users_corona_2 = {t for t in users_corona if t not in users_corona_1}
    #print('Starting with {} users'.format(len(users_corona_2)))
    dic_corona_2 = lists_of_users(users_corona_2, r , blists)
    dic_corona_links_2 = intersection_pairs_users_2(lists_corona_1, dic_corona_2, k)
    print('Getting corona II took {} seconds'.format(time.clock()-START))
    print("test 2", len(dic_corona_links_2))  
    return dic_corona_links_2


# Create intermediate graph from one seed
def merge_dict(dic1, dic2, seed):
    nodes = set(dic1.keys()) | set(dic2.keys())
    print('Is seed in nodes of graph : ', seed[7:] in nodes)
    dic_nodes = {n:v for n,v in zip(range(len(nodes)),nodes)}
    dic_nodes_i = {n:v for n,v in zip(nodes,range(len(nodes)))}
    final_dict = {}
    for node in nodes:
        n_dict = {}
        if node in dic1.keys():
            for n,v in dic1[node].items():
                n_dict[dic_nodes_i[n]] = v
        if node in dic2.keys():
            for n,v in dic2[node].items():
                n_dict[dic_nodes_i[n]] = v
        final_dict[dic_nodes_i[node]] = n_dict
    return final_dict, dic_nodes, dic_nodes_i


# Return np matrix representing the graph
def build_matrix(dic, sz):
    matrix = np.zeros((sz, sz))
    for k, v in dic.items():
        for k2, val in v.items():
            matrix[int(k), int(k2)] = val
    return matrix


# Estimate the frequency of visits of each vertice with a rw starting from the seed
def get_paths(matrix, seed_ind, dic_users):
    paths2 = np.dot(matrix, matrix[seed_ind,:].T)
    v = np.ones(paths2.size)
    paths = np.dot(matrix, v) + paths2
    paths = (paths - np.mean(paths)) / np.std(paths)
    dic = {}
    for i, p in np.ndenumerate(paths):
        if p > 0:
            dic[dic_users[i[0]]] = p
    return paths, dic


# Eliminate vertices visited by the rw less than average
def reduced_graph(selected_users, dic, dic_users):
    red_graph = {}
    for k, v in dic.items():
        if dic_users[k] in selected_users:
            dico = {}
            for k2, val in v.items():
                if dic_users[k2] in selected_users:
                    dico[dic_users[k2]] = val
            red_graph[dic_users[k]] = dico
    return red_graph


# Read file representing the grph and the importance of nodes
def read_graph_freq(champ, k):
    graph = {}
    intensities = {}
    directory = '../data/graph_{}_{}/'.format(champ, k)
    try:
        with open(directory + 'graph.txt', 'r') as f:
            for line in f:
                line = line[:-1]
                line = line.split(';')
                dic = {}
                for pair in line[1:]:
                    n, val = pair.split(',')
                    dic[n] = int(val)
                graph[line[0]] = dic
        with open(directory + 'intensity.txt', 'r') as f:
            for line in f:
                v, imp = line.split(',')
                if imp[-1] == '\n':
                    imp = imp[:-1]
                intensities[v] = float(imp)
    except:
        print('No such file')
        return False
    return graph, intensities


def read_graph_uinfo(champ, k):
    graph = {}
    user_info = {}
    directory = '../data/graph_{}_{}/'.format(champ, k)
    try:
        with open(directory + 'graph.txt', 'r') as f:
            for line in f:
                line = line[:-1]
                line = line.split(';')
                dic = {}
                for pair in line[1:]:
                    n, val = pair.split(',')
                    dic[n] = int(val)
                graph[line[0]] = dic
        with open(directory + 'user_names_info.txt', 'r') as f:
            for line in f:
                line = line[:-1]
                v, name, imp, clu = line.split(',')
                user_info[v] =[name, float(imp)]
    except:
        print('No such file')
        return False
    return graph, user_info


def read_desired_users_and_lists(champ, k):
    desired_users = set()
    users_lists = {}
    directory = '../data/graph_{}_{}/'.format(champ, k)
    with open(directory + 'desired_vertices.txt', 'r') as f:
        for line in f:
            desired_users.add(line[:-1])
    with open(directory + 'vertices_lists.txt', 'r') as f:
        for line in f:
            line = line[:-1].split(',')
            users_lists[line[0]] = line[1:]
    return desired_users, users_lists




# Read files reprensenting node membership to clusters and lists of each cluster
def read_com(champ, k):
    clusters = {}
    lists_on_clusters = {}
    directory = '../data/graph_{}_{}/'.format(champ, k)
    try:
        with open(directory + 'clusters.txt', 'r') as f:
            for line in f:
                line = line[:-1]
                line = line.split(',')
                users = set()
                for user in line[1:]:
                    users.add(user)
                clusters[int(line[0])] = users
        with open(directory + 'lists_clusters.txt', 'r') as f:
            for line in f:
                line = line[:-1]
                line = line.split(',')
                lists = set()
                for l in line[1:]:
                    lists.add(l)
                lists_on_clusters[int(line[0])] = lists
    except:
        print('No such file')
        return False
    return clusters, lists_on_clusters


# Return direct neighborhood
def getReducedNeighbor(user, blids, k):
    START = time.clock()
    neighbor_dict = {}
    r = redis.StrictRedis(host='localhost')
    lists = [t for t in r.smembers(user) if t.decode('utf-8') not in blids]
    if len(lists) == 0:
        print('User {} is isolated.'.format(user))
        return neighbor_dict
    else:
        print('User {} is in {} lists.'.format(user, len(lists)))
        pipe = r.pipeline()
        for l in lists:
            listId = 'listId:' + l.decode('utf-8')
            pipe.smembers(listId)
        neighs = list(set([j for i in pipe.execute() for j in i]))
        pipe = r.pipeline()
        for n in neighs:
            ne = 'userId:' + n.decode('utf-8')
            pipe.sinter(user,ne)
        valences = pipe.execute()
        valences = [len([x for x in t if x not in blids]) for t in valences]
        print(valences[0])
        for u,val in zip(neighs, valences):
            if val > k:
                us = 'userId:' + u.decode('utf-8')
                neighbor_dict[us] = val
        print('Finding neighbors of {} took {} seconds.\n'.format(user,time.clock() - START))
        return neighbor_dict


# Alt grow gaph
def getGraph(seed, blists, k):
    dict_of_dicts = {}
    dict_of_dicts[seed] = getReducedNeighbor(seed, blists, k)
    keys = dict_of_dicts[seed].keys()
    print('Seed has {} neighbors.'.format(len(keys)))
    for el in keys:
        dict_of_dicts[el] = getReducedNeighbor(el, blists, k)
        print('User {} has {} neighbors.'.format(el,len(dict_of_dicts[el].keys())))
    dict_of_dicts = sym_dict(dict_of_dicts)
    dic_nodes = {n:v for n,v in zip(range(len(keys)+1),[seed]+list(keys))}
    dic_nodes_i = {n:v for n,v in zip([seed]+list(keys),range(len(keys)+1))}
    final_dict = {}
    for node,neis in dict_of_dicts.items():
        n_dict = {}
        for k,v in neis.items():
            n_dict[dic_nodes_i[k]] = v
        final_dict[dic_nodes_i[node]] = n_dict
    return final_dict, dic_nodes, dic_nodes_i


def construct_reduced_graph(c_p, uinfo, desired_users):
    l_edges, weights = edgelist(c_p)
    l_vertices = list(desired_users)
    l_vertices = sorted([int(t) for t in l_vertices])
    l_vertices = [str(t) for t in l_vertices]
    net = Graph(directed = False)
    net.add_vertices(l_vertices)
    w = []
    edges = []
    for i,edge in enumerate(l_edges):
        if edge[0] in desired_users and edge[1] in desired_users:
            #net.add_edge(edge)
            w.append(weights[i])
            edges.append(edge)
    net.add_edges(edges)
    net.es['weight'] = w
    p = net.community_multilevel(weights = w)
    user_full_info = {}
    for i,cl in enumerate(p):
        for v in cl:
            v = net.vs[v]['name']
            user_full_info[v] = uinfo[v] + [i]
    return edges, w, user_full_info
#def cut_graph():


def get_keywords(uinfo, ulists, champ, k):
    directory = '../data/graph_{}_{}/'.format(champ,k)
    es_t = Elasticsearch([{'local':'localhost'}])
    clu_lists = {}
    for k,v in ulists.items():
        clu = uinfo[k][2]
        if clu not in clu_lists.keys():
            clu_lists[clu] = set()
        else:
            clu_lists[clu] = clu_lists[clu] | set(v)
    c_kws = {}
    for k,v in clu_lists.items():
        names = set()
        for l in v:
            nm = get_list_name(es_t, l)
            names.add(nm)
        txt = ' '.join(list(names))
        wds = min(5, len(names))
        ws = txt.split(' ')
        unwanted_words = ['/','et','&', '-', '***NoListName***', 'de', ',', '_', '.', 'the', 'of', 'les', 'le', 'la', 'and', 'list','my']
        ws = [x.strip() for x in ws if len(x)>0]
        ws = [x.lower() for x in ws]
        ws = [x for x in ws if x not in unwanted_words]
        kws = Counter(ws).most_common(wds)
        print(kws)
        print('\n')
        c_kws[k] = kws
    with open(directory + 'reduced_clusters_kws.txt', 'w') as f:
        for k,v in c_kws.items():
            f.write(str(k))
            for pair in v:
                f.write(',{},{}'.format(pair[0],pair[1]))
            f.write('\n')
    return c_kws 



def write_gexf(r_g, w, uinfo, champ, k):
    edges = set()
    file = open('../data/graph_{}_{}/graphe_reduit.gexf'.format(champ, k), 'wb')
    gexf_file = gexf.Gexf('Reduced Graph similarity',champ + str(k))
    gph = gexf_file.addGraph("undirected", "static", 'gr')
    intensAtt = gph.addNodeAttribute("intensity","0.0","float")
    clusAtt = gph.addNodeAttribute("cluster","100","int")
    for k,v in uinfo.items():
        n = gph.addNode(k,v[0])
        n.addAttribute(intensAtt, str(v[1]))
        n.addAttribute(clusAtt, str(v[2]))
    eid = 0
    for (k,v) in r_g:
        if (k,v) not in edges and (v,k) not in edges:
            edges.add((k,v))
            gph.addEdge(eid, k, v, weight=w[eid])
        eid += 1
    gexf_file.write(file)