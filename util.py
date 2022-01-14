import networkx as nx
import numpy as np
import torch


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(x, y, Normalize=True):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    g_list = []
    label_dict = {}
    feat_dict = {}

    # adj = np.load('dataset/%s/adj.npy' % (dataset))

    # 矩阵总数
    n_g = len(x)
    # n_g = 50
    for i in range(n_g):
        n = x[i].shape[0]
        l = y[i]

        g = nx.Graph()
        node_tags = []
        node_features = []
        n_edges = 0
        for j in range(n):
            g.add_node(j)
            if j == n - 1:
                row = [j, 1, j - 1]
            else:
                row = [j, 1, j + 1]
            attr = x[i][j]
            g.add_node(j, att=attr)
            if not row[0] in feat_dict:
                mapped = len(feat_dict)
                feat_dict[row[0]] = mapped
            node_tags.append(feat_dict[row[0]])

            # if tmp > len(row):
            node_features.append(attr)

            n_edges += row[1]
            for k in range(2, len(row)):
                g.add_edge(j, row[k])

        if node_features != []:
            node_features = np.stack(node_features)
            node_feature_flag = True
        else:
            node_features = None
            node_feature_flag = False

        g_list.append(S2VGraph(g, l, node_tags))

    ##################
    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(g.g.nodes[0]['att']))
        for i in range(len(g.node_tags)):
            g.node_features[i] = torch.FloatTensor(g.g.nodes[i]['att'])

    ### Normalizing
    if (Normalize):
        X_concat = np.concatenate([graph.node_features.view(-1, graph.node_features.shape[1]) for graph in g_list])
        Min = torch.Tensor(np.min(X_concat, axis=0))[:-2]
        Ptp = torch.Tensor(np.ptp(X_concat, axis=0))[:-2]
        for g in g_list:
            g.node_features[:, :-2] = 2. * (g.node_features[:, :-2] - Min) / Ptp - 1

    for g in g_list:
        g.node_features2 = torch.zeros(len(g.node_tags), 2 * len(g.g.nodes[0]['att']))
        for i in range(len(g.node_tags)):
            if (i == 0):
                g.node_features2[i] = torch.cat([g.node_features[i], g.node_features[i]])
            else:
                g.node_features2[i] = torch.cat([g.node_features[i], g.node_features[i] - g.node_features[i - 1]])

    return g_list
