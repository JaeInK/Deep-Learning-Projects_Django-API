
# coding: utf-8

from operator import itemgetter
import pandas as pd
import networkx as nx

class NewsNetwork:
    def __init__(self):
        """constructor"""
        self.G = nx.Graph()
        self.new_G = None
        self.ps = dict() ### {key :[pol, Pp, Pn, Qp, Qp_2, Qn, Qn_2, Qneut, Qneut_2]} Pp : polarity_prop of Pos, Qp : quotation of largest Pos prop

    def read_file(self,df):
        """file read and generate network"""
        try:
            if type(df) is not pd.pandas.core.frame.DataFrame:
                raise TypeError

            before_art_num=df.iloc[0,0] #시작 article_num
            before_index = 0
            for row in df.itertuples():
                ### ps update
                if row.person not in self.ps.keys(): #생성
                    self.ps[row.person] = [1 if row.polarity_type == "Pos" and row.polarity_prop != 1
                                           else -1 if row.polarity_type == "Neg" and row.polarity_prop != 1
                                           else 0
                                        , row.polarity_prop if row.polarity_type == "Pos" and row.polarity_prop != 1 else 0
                                        , row.polarity_prop if row.polarity_type == "Neg" and row.polarity_prop != 1 else 0
                                        , row.quotation if row.polarity_type == "Pos" and row.polarity_prop != 1 else ''
                                        , '' # Qp_2
                                        , row.quotation if row.polarity_type == "Neg" and row.polarity_prop != 1 else ''
                                        , '' # Qb_2
                                        , row.quotation if row.polarity_type == "Neutral" else ''
                                        , '' # Qneut_2
                                          ]
                else: #업뎃
                    if row.polarity_type == "Pos" and row.polarity_prop != 1:
                        self.ps[row.person][0] +=1
                        if row.polarity_prop > self.ps[row.person][1]:
                            self.ps[row.person][4] = self.ps[row.person][3] # move Qp to Qp_2
                            self.ps[row.person][1] = row.polarity_prop
                            self.ps[row.person][3] = row.quotation #가장 큰 prop의 인용문
                    elif row.polarity_type == "Neg" and row.polarity_prop != 1:
                        self.ps[row.person][0] -= 1
                        if row.polarity_prop > self.ps[row.person][2]: #'Neg'여도 0.5보다 크다
                            self.ps[row.person][6] = self.ps[row.person][5] # move Qn to Qn_2
                            self.ps[row.person][2] = row.polarity_prop
                            self.ps[row.person][5] = row.quotation #Neg 일때도 가장 큰 prop의 인용문
                    elif row.polarity_type == "Neutral":
                        if len(row.quotation) < len(self.ps[row.person][5]): #작은길이로 업뎃
                            self.ps[row.person][8] = self.ps[row.person][7]
                            self.ps[row.person][7] = row.quotation

                ### gr update
                if row.group not in self.ps.keys():
                    self.ps[row.group] = [1 if row.polarity_type == "Pos" and row.polarity_prop != 1
                                           else -1 if row.polarity_type == "Neg" and row.polarity_prop != 1
                                           else 0
                                        , row.polarity_prop if row.polarity_type == "Pos" and row.polarity_prop != 1 else 0
                                        , row.polarity_prop if row.polarity_type == "Neg" and row.polarity_prop != 1 else 0
                                        , row.quotation if row.polarity_type == "Pos" and row.polarity_prop != 1 else ''
                                        , '' # Qp_2
                                        , row.quotation if row.polarity_type == "Neg" and row.polarity_prop != 1 else ''
                                        , '' # Qb_2
                                        , row.quotation if row.polarity_type == "Neutral" else ''
                                        , '' # Qneut_2
                                          ]
                else: #업뎃
                    if row.polarity_type == "Pos" and row.polarity_prop != 1:
                        self.ps[row.group][0] +=1
                        if row.polarity_prop > self.ps[row.group][1]:
                            self.ps[row.group][4] = self.ps[row.group][3] # move Qp to Qp_2
                            self.ps[row.group][1] = row.polarity_prop
                            self.ps[row.group][3] = row.quotation #가장 큰 prop의 인용문
                    elif row.polarity_type == "Neg" and row.polarity_prop != 1:
                        self.ps[row.group][0] -= 1
                        if row.polarity_prop > self.ps[row.group][2]: #'Neg'여도 0.5보다 크다
                            self.ps[row.group][6] = self.ps[row.group][5] # move Qn to Qn_2
                            self.ps[row.group][2] = row.polarity_prop
                            self.ps[row.group][5] = row.quotation #Neg 일때도 가장 큰 prop의 인용문
                    elif row.polarity_type == "Neutral":
                        if len(row.quotation) < len(self.ps[row.group][5]): #작은길이로 업뎃
                            self.ps[row.group][8] = self.ps[row.group][7]
                            self.ps[row.group][7] = row.quotation


                #node and edge construct
                if before_art_num != row.article_num:
                    art_subset = df.iloc[before_index:row.Index]
                    before_index = row.Index
                    before_art_num = row.article_num
                    self._add_nodes_and_edges(art_subset)
                if row.Index != 0 and row.Index == df.shape[0]-1:
                    art_subset = df.iloc[before_index:row.Index+1]
                    self._add_nodes_and_edges(art_subset)

            # end of df.itertuples()

            ###
            # add polarity attributes
            for node in self.G.nodes():
                polarity_result = self.ps[node]
                if polarity_result[0] > 0:
                    self.G.nodes()[node]['Polarity'] = 'Positive'
                    self.G.nodes()[node]['ES1'] = polarity_result[3]
                    self.G.nodes()[node]['ES2'] = polarity_result[4]
                    self.G.nodes()[node]['Score_ES1'] = polarity_result[1]
                elif polarity_result[0] < 0:
                    self.G.nodes()[node]['Polarity'] = 'Negative'
                    self.G.nodes()[node]['ES1'] = polarity_result[5]
                    self.G.nodes()[node]['ES2'] = polarity_result[6]
                    self.G.nodes()[node]['Score_ES1'] = polarity_result[2]
                else:
                    self.G.nodes()[node]['Polarity'] = 'Neutral'
                    self.G.nodes()[node]['ES1'] = polarity_result[7]
                    self.G.nodes()[node]['ES2'] = polarity_result[8]
                    self.G.nodes()[node]['Score_ES1'] = 0.5

        except TypeError:
            print('Input must be Pandas.Dataframe type')

        except KeyError:
            print('Input must be a News dataframe')
            print("Columns of News dataframe: ['article_num', 'sentence_num', 'person', 'group', 'noun', 'quotation', 'sentence', 'polarity_type', 'polarity_prop']")

        except AttributeError:
            print('Input must be a News dataframe')
            print("Columns of News dataframe: ['article_num', 'sentence_num', 'person', 'group', 'noun', 'quotation', 'sentence', 'polarity_type', 'polarity_prop']")

    def _combi(self,li):
        """list li 원소들 n개에 대해 nC2개 nested list로 반환"""
        i=0
        combi_list=[]
        while i<len(li):
            j=i+1
            while j<len(li):
                combi_list.append([li[i],li[j]])
                j+=1
            i+=1
        return combi_list

    def _add_nodes_and_edges(self, art_subset):
        unique_person = list(set(art_subset.person))
        unique_group = list(set(art_subset.group))
        # unique_total = unique_person + unique_group

        for person in unique_person:
            if person not in self.G.nodes():
                self.G.add_node(person, Type = 'person', Freq = 1)
            else:
                self.G.nodes()[person]['Freq'] += 1

        for group in unique_group:
            if group not in self.G.nodes():
                self.G.add_node(group, Type = 'group', Freq = 1)
            else:
                self.G.nodes()[group]['Freq'] += 1

        for edge in self._combi(unique_person + unique_group):
            if not self.G.has_edge(*edge):
                self.G.add_edge(*edge, count = 1)
            else:
                self.G.edges()[edge]['count'] += 1


    def _edge_pruning(self, graph, edge_threshold):
        g1 = graph.copy()
        for edge in graph.edges(data=True):
            if edge[2]['count'] <= edge_threshold:
                g1.remove_edge(edge[0],edge[1])
        return g1

    def _node_pruning(self, graph, deg_threshold):
        node_and_degree = graph.degree()
        # node_degree가 1인 것은 삭제
        g2 = graph.copy()
        for node in graph.nodes():
            if node_and_degree[node] <= deg_threshold:
                g2.remove_node(node)
        return g2

    def pre_processing(self, edge_threshold, deg_threshold):
        try:
            if '' not in self.G:
                raise Exception
            # remove empty node
            self.new_G = self.G.copy()
            self.new_G.remove_node("")
            # edge_pruning
            self.new_G = self._edge_pruning(self.new_G, edge_threshold)
            # node_pruning
            self.new_G = self._node_pruning(self.new_G, deg_threshold)
        except:
            print('Need to be load News dataframe, use "read_file" method before pre-processing')

    def set_node_link_attrs(self, subgraph = None, edge_bold = 10):
        """
        Calculate nodes and edges attributes and set them into NNW obj.
        Many edges become bold if Edge_bold increases (default=10)
        """

        if not subgraph:
            graph = self.new_G
        else:
            graph = self.new_G = subgraph

        betweenness_cen = nx.betweenness_centrality(graph)
        degree_cen = nx.degree_centrality(graph)
        closeness_cen = nx.closeness_centrality(graph)

        nx.set_node_attributes(graph, betweenness_cen, 'Betweenness_centrality')
        nx.set_node_attributes(graph, degree_cen, 'Degree_centrality')
        nx.set_node_attributes(graph, closeness_cen, 'Closeness_centrality')

        for node in graph.nodes():
            edge_to_list = list(graph.edges(nbunch=node, data = True))
            count_list = [edge_to[2]['count'] for edge_to in edge_to_list]
            if len(count_list) == 0:
                graph.nodes()[node]['Most_friendly'] = ''

            else:
                most_friendly_index = count_list.index(max(count_list))
                graph.nodes()[node]['Most_friendly'] = edge_to_list[most_friendly_index][1]

        _, max_count = sorted(nx.get_edge_attributes(graph, 'count').items(), key=itemgetter(1))[-1]

        for edge in graph.edges(data = True):
            if edge[2]['count'] >= max_count/edge_bold or edge[2]['count'] >= 30:
                edge[2]['type'] = 'Strong'
            else:
                edge[2]['type'] = 'Weak'


    def get_network_attrs(self, net_name, subgraph = None):
        """
        Return dict(n_nodes, n_edges, n_connected, max_deg_node, max_deg, max_bet_node, max_bet)
        """
        if not subgraph:
            graph = self.new_G
        else:
            graph = self.new_G = subgraph

        network_info = {}
        network_info['label'] = net_name
        network_info['Number_of_nodes'] = nx.number_of_nodes(graph)
        network_info['Number_of_edges'] = nx.number_of_edges(graph)
        network_info['Number_of_subgraph'] = nx.number_connected_components(graph)

        max_deg_node, max_deg = sorted(nx.degree_centrality(graph).items(), key=itemgetter(1))[-1]
        max_bet_node, max_bet = sorted(nx.betweenness_centrality(graph).items(), key=itemgetter(1))[-1]

        network_info['Maximum_degree_centrality_node'] = max_deg_node
        network_info['Maximum_degree_centrality'] = round(max_deg, 2)
        network_info['Maximum_betweenness_centrality_node'] = max_bet_node
        network_info['Maximum_betweenness_centrality'] = round(max_bet, 2)
        network_info['Number_of_positive'] = list(nx.get_node_attributes(graph, 'Polarity').values()).count('Positive')
        network_info['Number_of_negative'] = list(nx.get_node_attributes(graph, 'Polarity').values()).count('Negative')
        network_info['Number_of_neutral'] = network_info['Number_of_nodes'] - network_info['Number_of_positive'] - network_info['Number_of_negative']

        return network_info

    def subgraph(self, n = 0, max_subgraph = False):
        """
        Generate n-th connected components as subgraph (If there are several connected components)
        If max_subgraph is True, generate maximum subgraph (Regardless of n)
        """
        if not self.new_G:
            self.pre_processing(0, 0)
        graph = self.new_G

        if not max_subgraph:
            subgraph = list(nx.connected_component_subgraphs(graph))[n]
        else:
            subgraph = max(nx.connected_component_subgraphs(graph), key=len)
        return subgraph

    def get_node_link_data(self):
        if not self.new_G:
            self.pre_processing(0, 0)
        graph = self.new_G

        graph_dic = nx.node_link_data(graph)
        node_keys = list(graph_dic['nodes'][0].keys())
        node_keys.remove('id')
        for inner_dic in graph_dic['nodes']:
            inner_dic['properties'] = {}
            for key in node_keys:
                inner_dic['properties'].update({key:inner_dic[key]})
                del inner_dic[key]

        edge_keys = list(graph_dic['links'][0].keys())
        edge_keys.remove('target')
        edge_keys.remove('source')
        edge_keys.remove('count')
        for inner_dic in graph_dic['links']:
            inner_dic['properties'] = {}
            for key in edge_keys:
                inner_dic['properties'].update({key:inner_dic[key]})
                del inner_dic[key]

        del graph_dic['directed']
        del graph_dic['graph']
        del graph_dic['multigraph']

        return graph_dic
