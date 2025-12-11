from collections import defaultdict, Counter
from datetime import datetime
from turtle import down
from dateutil import parser
from email.policy import default
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from queue import Queue
from statistics import mean
from statsmodels.stats.weightstats import ztest as ztest
from string import ascii_letters
from sys import set_asyncgen_hooks
from time import time
from typing import Dict
from tqdm import tqdm
# from torch import Set
import email
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import seaborn as sns


def get_key(df):
    dic = {}
    for index, row in df.iterrows():
        dic[row['POST_ID']] = row['PROPERTIES']
    return dic

path = '../data/'
dataset = ['soc-sign-bitcoinotc.csv',\
    'soc-redditHyperlinks-body.tsv',\
        'soc-redditHyperlinks-title.tsv',\
            'email-Eu-core-temporal.txt',\
                'CollegeMsg.txt',\
                'enron_cmu.csv',\
                'sorted_enron.csv',\
                    'emails.csv',\
                        'sorted_emails1.csv']
# dataset list
# 0 Bitcoin OTC trust weighted signed network
# 1 Social Network: Reddit Hyperlink Network
# 2 Social Network: Reddit Hyperlink Network
# 3 email-Eu-core temporal network
# 4 college message dataset
# 5 enron email dataset
from_set = defaultdict(list)
to_set = defaultdict(list)
ft_set = defaultdict(list)

def draw_four_plots(data, title, suptitle):
    fig, axs = plt.subplots(2,2)
    fig.suptitle(suptitle)
    indices = [[0,0],[0,1],[1,0],[1,1]]
    for index, data_i, title_i in zip(indices,data,title):
        axs[index[0], index[1]].hist(data_i)
        axs[index[0], index[1]].set_title(title_i)     
        # axs[index[0], index[1]].set_ylim(0,50000) 
        # axs[index[0], index[1]].set_xlim(0,6) 
    plt.savefig(path + 'figure/'+suptitle+".png")

def draw_before_after_distribution(lines, time = 23708115):
    #before time point
    all_users = set()
    send_valid_users = set()
    send_user_before = []
    send_user_after = []

    to_valid_users = set()
    to_user_before = []
    to_user_after = []
    for line in lines:
        string = line.replace("\n","").split(" ")
        all_users.add(string[0])
        if int(string[2])< time-1:
            send_valid_users.add(string[0])
            send_user_before.append(int(string[2]))
            to_valid_users.add(string[0])
            to_user_before.append(int(string[2]))
    for line in lines:
        string = line.replace("\n","").split(" ")
        if int(string[2]) > time and string[0] in send_valid_users:
            send_user_after.append((int(string[2])-time))
        if int(string[2]) > time and string[1] in to_valid_users:
            to_user_after.append((int(string[2])-time))
            
    suptitle = 'Based on before time point'
    title = ['send_user_before','send_user_after','to_user_before','to_user_after']
    data = [send_user_before,send_user_after,to_user_before,to_user_after]
    draw_four_plots(data, title, suptitle)
    # after time point
    all_users = set()
    send_valid_users = set()
    send_user_before = []
    send_user_after = []

    to_valid_users = set()
    to_user_before = []
    to_user_after = []
    for line in lines:
        string = line.replace("\n","").split(" ")
        all_users.add(string[0])
        if int(string[2]) > time:
            send_valid_users.add(string[0])
            send_user_before.append((int(string[2])-time))
            to_valid_users.add(string[0])
            to_user_before.append((int(string[2])-time))
    for line in lines:
        string = line.replace("\n","").split(" ")
        if int(string[2]) < time+1 and string[0] in send_valid_users:
            send_user_after.append(int(string[2]))
        if int(string[2]) > time+1 and string[1] in to_valid_users:
            to_user_after.append(int(string[2]))
            
    suptitle = 'Based on after time point'
    title = ['send_user_before','send_user_after','to_user_before','to_user_after']
    data = [send_user_before,send_user_after,to_user_before,to_user_after]
    draw_four_plots(data, title, suptitle)

# generate a default form
def def_value():
    return [[],[]]

# get the difference for two lists in a list l  [[1,3,5],[2,8]] ->[2,2],[6],2,6
def get_diff(l):
    l0 = [l[0][i] - l[0][i-1] for i in range(1,len(l[0]))]
    l1 = [l[1][i] - l[1][i-1] for i in range(1,len(l[1]))]
    mean0 = mean(l0) if len(l0) != 0 else -1
    mean1 = mean(l1) if len(l1) != 0 else -1
    return l0,l1,mean0,mean1

def difference(lines,time = 23708115):
    from_users = defaultdict(def_value)
    to_users = defaultdict(def_value)
    print(time)
    for line in lines:
        string = line.replace("\n","").split(" ")
        cur_time = int(string[2])
        if cur_time < time + 1:
            from_users[string[0]][0].append(cur_time)
            to_users[string[1]][0].append(cur_time)
        else:
            from_users[string[0]][1].append(cur_time)
            to_users[string[1]][1].append(cur_time)
    from_user_all = []
    to_user_all = []
    for key in from_users.keys():
        val = get_diff(from_users[key])
        try:
            from_user_all.append(ztest(val[0],val[1])[1])
        except:
            from_user_all.append(0)
    for key in to_users.keys():
        val = get_diff(to_users[key])
        try:
            to_user_all.append(ztest(val[0],val[1])[1])
        except:
            to_user_all.append(0)
    titles = ['from_distribution','to_distribution']
    fig, axs = plt.subplots(2)
    fig.suptitle('Z test for difference distribution')
    data = [from_user_all,to_user_all]
    for i in range(2):
        axs[i].hist(data[i],alpha=0.5,label = 'z test between before and after, time is '+str(time))
        axs[i].legend(loc='upper right')
        axs[i].set_title(titles[i])
    plt.savefig(path + 'figure/Z test for difference '+str(time)+'.png')
    plt.close()

# conduct the ztest
def difference_ztest(lines,time = 23708115):
    from_users = defaultdict(def_value)
    to_users = defaultdict(def_value)
    for line in lines:
        string = line.replace("\n","").split(" ")
        cur_time = int(string[2])
        if cur_time < time + 1:
            from_users[string[0]][0].append(cur_time)
            to_users[string[1]][0].append(cur_time)
        else:
            from_users[string[0]][1].append(cur_time)
            to_users[string[1]][1].append(cur_time)
    from_user_all = []
    to_user_all = []
    for key in from_users.keys():
        val = get_diff(from_users[key])
        from_user_all[0].append(val[2])
        from_user_all[1].append(val[3])
    for key in to_users.keys():
        val = get_diff(to_users[key])
        to_user_all[0].append(val[2])
        to_user_all[1].append(val[3])
        
def between_distribution(lines, time = 23708115):
    def def_set():
        return [set(),set()]
    from_users = defaultdict(def_set)
    to_users = defaultdict(def_set)
    for line in lines:
        string = line.replace("\n","").split(" ")
        cur_time = int(string[2])
        if cur_time < time + 1:
            from_users[string[0]][0].add(string[1])
            to_users[string[1]][0].add(string[0])
        else:
            if string[1] not in from_users[string[0]][0]:
                from_users[string[0]][1].add(string[1])
            if string[0] not in to_users[string[1]][0]:
                to_users[string[1]][1].add(string[0])
    from_num, to_num = 0,0
    for key in from_users.keys():
        if len(from_users[key][1]) !=0:
            from_num +=1
    for key in to_users.keys():
        if len(to_users[key][1]) !=0:
            to_num +=1       
    return from_num, to_num
def between_distribution_change_with_time(lines, start_time = 0, final_time = 63708155, step = 500000):
    from_num_all, to_num_all = [],[]
    # final_time = 63708155
    # final_time = 1500000
    for i in range(start_time,final_time, 500000):
        from_num, to_num = between_distribution(lines,time = i)
        from_num_all.append(from_num)
        to_num_all.append(to_num)
    x = list(range(start_time,final_time, 500000))
    plt.plot(x,from_num_all,label = 'from')
    plt.plot(x,to_num_all,label = 'to')
    plt.xlabel('time')
    plt.ylabel('undiscovered relationships')
    plt.title('relationship growth')
    plt.legend()
    plt.savefig(path + 'figure/relationship growth.png')

def check_email_name(possible_emails,email_address):
    try:
        address = email_address.split(",")
    except:
        return
    # for pm in possible_emails:
    #     if pm not in address:
    for i in range(len(address)):
        try:
            address[i] = address[i].split("@")[1][:-2]
        except:
            address[i] = ''
    return address

def from_to_company(csv_file): # only for dataset 5
    possible_emails = ['hotmail','gmail','yahoo','enron']
    dic_from = defaultdict(int)
    dic_to = defaultdict(int)
    for i in range(len(csv_file)):
        from_email = check_email_name(possible_emails,csv_file.iloc[i]['From'])
        to_email = check_email_name(possible_emails,csv_file.iloc[i]['To'])
        if from_email:
            for email in from_email:
                dic_from[email] +=1
        if to_email:
            for email in to_email:
                dic_to[email] += 1
            
    json_from = json.dumps(dic_from)
    json_to = json.dumps(dic_to)

    f = open("from.json","w")
    f.write(json_from)
    f.close()
    f = open("to.json","w")
    f.write(json_to)
    f.close()

def get_name(email):
    try:
        all_list = email.split(", ")
        # for i in range(len(all_list)):
        #     all_list[i] = all_list[i][1:-1]
        return all_list
    except:
        return []
def get_enron_name(email_list):
    new_list = []
    for e in email_list:
        if 'enron' in e:
            new_list.append(e)
    return new_list
def from_to_name(csv_file): # only for dataset 5, 6
    possible_emails = ['hotmail','gmail','yahoo','enron']
    dic_from = defaultdict(int)
    dic_to = defaultdict(int)
    l = len(csv_file)
    l = 500
    for i in range(250000,300000):
        name_from = get_name(csv_file.iloc[i]['From'])
        name_to = get_name(csv_file.iloc[i]['To'])
        
        for name in name_from:
            dic_from[name] += 1
        for name in name_to:
            dic_to[name] += 1
        
        # except:
        #     print(1)
        #     name_from = get_name(csv_file.iloc[i]['From'])
        #     name_to = get_name(csv_file.iloc[i]['To'])
    
    json_from = json.dumps(dic_from)
    json_to = json.dumps(dic_to)

    f = open("from_name.json","w")
    f.write(json_from)
    f.close()
    f = open("to_name.json","w")
    f.write(json_to)
    f.close()
def emailformat():
    return defaultdict(list)
def get_stage(name):
    if name in dic_stage:
        return dic_stage[name]
    else:
        return 6
def intn1():
    return -1
    
def sort2(left,right):
    # keep left < right
    if left > right:
        return right, left
    return left, right

def online_two_path(graph,dic_two_path,new_edges):
    # update two_path
    for edges in new_edges:
        p1,p2 = edges.split(",")
        for p3 in graph[p1]: # p2-p1-p3
            if p3 not in graph[p2]:
                dic_two_path[p3][p2].add(p1)
                dic_two_path[p2][p3].add(p1)
        for p3 in graph[p2]: # p1-p2-p3
            if p3 not in graph[p1]:
                dic_two_path[p3][p1].add(p2)
                dic_two_path[p1][p3].add(p2)
        # graph[p1].add(p2)
        # graph[p2].add(p1)
    # update graph.  not update with two_path? count 1 day by 1 day? or use a ordered set?
    for edges in new_edges:
        p1,p2 = edges.split(",")
        graph[p1].add(p2)
        graph[p2].add(p1)
    return graph,dic_two_path

def dic_set():
    return defaultdict(set)

def shortest_path(graph, start, goal):
    max_path_len = 6
    explored = set()
     
    # Queue for traversing the
    # graph in the BFS
    queue = Queue()
    queue.put((start, [start]))
     
    # If the desired node is
    # reached
    if start == goal:
        return 0
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue.qsize()>0:
        (node, path) = queue.get()
        if len(path) > max_path_len:
            return max_path_len + 1
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                queue.put((neighbour, path + [neighbour]))
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    return len(path) + 1
            explored.add(node)
    # Condition when the nodes
    # are not connected
    return max_path_len + 1

def generate_noise_dataset(csv_file, year = "1999", noise = 0):
    if year != 'NA' and year in [1999, 2000, 2001, 2002]:
        # 1999 start from 1138
        # 2000 start from 12282
        # 2001 start from 208383
        # 2002 start from 481415
        year_dic = {1999: [1138, 12281], 2000:[12282, 208382], 2001:[208383, 481414], 2002:[481415, 517319]}
        start_i, end_i = year_dic[year]
    for i in range(start_i, end_i):
        p = random.uniform(0, 1)
        if p < noise: # drop in noise
            generate_noise_dataset()
        else:
            try:
                math.isnan(csv_file.iloc[i]['X-cc']) == False
                cc_receiver = []
            except:
                # print(csv_file.iloc[i]['X-cc'])
                a = re.findall(r"CN=(.*?)>", csv_file.iloc[i]['X-cc'])
                cc_receiver = [aa.split("CN=")[-1].lower()+"@enron.com" for aa in a]
                if a == []:
                    a = re.findall(r"<(.*?)>", csv_file.iloc[i]['X-cc'])
                    cc_receiver = [aa.lower() for aa in a]
                    if a == []:
                        print(csv_file.iloc[i]['X-cc'])
                        a = re.findall(r"'(.*?)'", csv_file.iloc[i]['X-cc'])
                        cc_receiver = [aa.lower() for aa in a]
                        if a == []:
                            a = csv_file.iloc[i]['X-cc'].split(",")
                            cc_receiver = [aa.lower() for aa in a]
                            if a == []:
                                print(1)
                    else:
                        if "list not shown" in a[0]:
                            cc_receiver = []
                
            if i%10000 == 0:
                print(i)
            name_from = set(get_name(csv_file.iloc[i]['From']))
            enron_name_from = get_enron_name(name_from)
            name_to = set(get_name(csv_file.iloc[i]['To']))
            enron_name_to = get_enron_name(name_to)
            cur_day = csv_file.iloc[i]['Date'].date()
            if cur_day != prev_day:
                new_edges = edges_today - exsiting_edges
                graph, dic_two_path = online_two_path(graph,dic_two_path,new_edges)
                exsiting_edges.update(new_edges)
                edges_today = set()
                prev_day = cur_day
                distance = {}
            if len(enron_name_from) > 0 or len(enron_name_to) > 0:
                try:
                    content_length, stop_length = remove_stop_words(csv_file.iloc[i]['Content'],words)
                except:
                    content_length, stop_length = 1,0
            for nf in enron_name_from:
                if cur_day == name_prev_day[nf]:
                    name_rank[nf] += 1
                else:
                    name_rank[nf] = 1
                name_prev_day[nf] = cur_day
                for nt in name_to:
                    nt_stage = get_stage(nt)+1
                    nf_stage = get_stage(nf)+1
                    if nt in graph[nf]:
                        two_path_value = -1
                    else:
                        two_path_value = len(dic_two_path[nf][nt])
                    small, large = sort2(nf,nt)
                    key = small + ',' + large
                    if key in distance: # jiayi distance is wrong needs to be updated?
                        dist = distance[key]
                    else:
                        dist = shortest_path(graph, small,large)
                        distance[key] = dist
                    # if nt in graph[nf] or nf in graph[nt]:
                    #     dist = 1
                    # else:
                    #     dist = -1
                    dic_people[nf].append((nt,stop_length,1,csv_file.iloc[i]['Date'],stop_length/content_length,nf_stage/nt_stage,name_rank[nf],two_path_value, dist, cc_receiver))
                    
            for nf in name_from:
                if cur_day == name_prev_day[nf]:
                    name_rank[nf] += 1
                else:
                    name_rank[nf] = 1
                name_prev_day[nf] = cur_day
                for nt in enron_name_to:
                    nt_stage = get_stage(nt)+1
                    nf_stage = get_stage(nf)+1
                    if nt in graph[nf]:
                        two_path_value = -1
                    else:
                        two_path_value = len(dic_two_path[nf][nt])
                    small, large = sort2(nf,nt)
                    key = small + ',' + large
                    # if key in distance:
                    #     dist = distance[key]
                    # else:
                    #     dist = shortest_path(graph, small,large)
                    #     distance[key] = dist
                    if nt in graph[nf] or nf in graph[nt]:
                        dist = 1
                    else:
                        dist = -1
                    dic_people[nt].append((nf,stop_length,-1,csv_file.iloc[i]['Date'],stop_length/content_length,nt_stage/nf_stage,name_rank[nf],two_path_value, dist, cc_receiver))
            for nf in name_from:
                for nt in name_to:
                    a,b = sorted([nf,nt])
                    if a in name_from_to[b]:
                        s = ','.join(sorted([nf,nt]))
                        edges_today.add(s)
                    else:
                        name_from_to[b].add(a)
        
# def json_serial(obj):
#     if(isinstance(obj, datetime)):
#         return obj.isoformat
#     raise TypeError("Type not serializable")
def high_freqeuncy_all(csv_file, down_sampling = 0, start_i = 0, year = 'NA'):
    random.seed(1)
    dic_people = defaultdict(list)
    # start_i = 0
    # start_i = 400000
    prev_day = csv_file.iloc[start_i]['Date'].date()
    name_prev_day = defaultdict(lambda: datetime.min)
    name_rank = defaultdict(int)
    words = stopwords.words()
    exsiting_edges = set()
    edges_today = set()
    name_from_to = defaultdict(set)
    graph = defaultdict(set)
    dic_two_path = defaultdict(dic_set)
    distance = {}
    end_i = len(csv_file)
    if year != 'NA' and year in [1999, 2000, 2001, 2002]:
        # 1999 start from 1138
        # 2000 start from 12282
        # 2001 start from 208383
        # 2002 start from 481415
        year_dic = {1999: [1138, 12281], 2000:[12282, 208382], 2001:[208383, 481414], 2002:[481415, 517319]}
        start_i, end_i = year_dic[year]
    print(start_i, end_i)
    for i in tqdm(range(start_i, end_i)):
        if down_sampling != 0:
            p = random.uniform(0, 1)
            if p < down_sampling:
                continue
        # deal with X-cc
        try:
            math.isnan(csv_file.iloc[i]['X-cc']) == False
            cc_receiver = []
        except:
            # print(csv_file.iloc[i]['X-cc'])
            a = re.findall(r"CN=(.*?)>", csv_file.iloc[i]['X-cc'])
            cc_receiver = [aa.split("CN=")[-1].lower()+"@enron.com" for aa in a]
            if a == []:
                a = re.findall(r"<(.*?)>", csv_file.iloc[i]['X-cc'])
                cc_receiver = [aa.lower() for aa in a]
                if a == []:
                    # print(csv_file.iloc[i]['X-cc'])
                    a = re.findall(r"'(.*?)'", csv_file.iloc[i]['X-cc'])
                    cc_receiver = [aa.lower() for aa in a]
                    if a == []:
                        a = csv_file.iloc[i]['X-cc'].split(",")
                        cc_receiver = [aa.lower() for aa in a]
                        if a == []:
                            print(1)
                else:
                    if "list not shown" in a[0]:
                        cc_receiver = []
            
        if i%10000 == 0:
            print(i)
        name_from = set(get_name(csv_file.iloc[i]['From']))
        enron_name_from = get_enron_name(name_from)
        name_to = set(get_name(csv_file.iloc[i]['To']))
        enron_name_to = get_enron_name(name_to)
        cur_day = csv_file.iloc[i]['Date'].date()
        if cur_day != prev_day:
            new_edges = edges_today - exsiting_edges
            graph, dic_two_path = online_two_path(graph,dic_two_path,new_edges)
            exsiting_edges.update(new_edges)
            edges_today = set()
            prev_day = cur_day
            distance = {}
        if len(enron_name_from) > 0 or len(enron_name_to) > 0:
            try:
                content_length, stop_length = remove_stop_words(csv_file.iloc[i]['Content'],words)
            except:
                content_length, stop_length = 1,0
        for nf in enron_name_from:
            if cur_day == name_prev_day[nf]:
                name_rank[nf] += 1
            else:
                name_rank[nf] = 1
            name_prev_day[nf] = cur_day
            for nt in name_to:
                nt_stage = get_stage(nt)+1
                nf_stage = get_stage(nf)+1
                if nt in graph[nf]:
                    two_path_value = -1
                else:
                    two_path_value = len(dic_two_path[nf][nt])
                small, large = sort2(nf,nt)
                key = small + ',' + large
                if key in distance: # jiayi distance is wrong needs to be updated?
                    dist = distance[key]
                else:
                    dist = shortest_path(graph, small,large)
                    distance[key] = dist
                # if nt in graph[nf] or nf in graph[nt]:
                #     dist = 1
                # else:
                #     dist = -1
                dic_people[nf].append((nt,stop_length,1,csv_file.iloc[i]['Date'],stop_length/content_length,nf_stage/nt_stage,name_rank[nf],two_path_value, dist, cc_receiver))
                
        for nf in name_from:
            if cur_day == name_prev_day[nf]:
                name_rank[nf] += 1
            else:
                name_rank[nf] = 1
            name_prev_day[nf] = cur_day
            for nt in enron_name_to:
                nt_stage = get_stage(nt)+1
                nf_stage = get_stage(nf)+1
                if nt in graph[nf]:
                    two_path_value = -1
                else:
                    two_path_value = len(dic_two_path[nf][nt])
                small, large = sort2(nf,nt)
                key = small + ',' + large
                # if key in distance:
                #     dist = distance[key]
                # else:
                #     dist = shortest_path(graph, small,large)
                #     distance[key] = dist
                if nt in graph[nf] or nf in graph[nt]:
                    dist = 1
                else:
                    dist = -1
                dic_people[nt].append((nf,stop_length,-1,csv_file.iloc[i]['Date'],stop_length/content_length,nt_stage/nf_stage,name_rank[nf],two_path_value, dist, cc_receiver))
        for nf in name_from:
            for nt in name_to:
                a,b = sorted([nf,nt])
                if a in name_from_to[b]:
                    s = ','.join(sorted([nf,nt]))
                    edges_today.add(s)
                else:
                    name_from_to[b].add(a)
                    # smaller before
                    
    print(len(dic_people))
    for nm in name_from_to:
        print("nm = ", nm, end = ',')
        print(dic_people[nm])
    json_people = json.dumps(dic_people, default=str)
    if end_i == len(csv_file):
        if start_i != 0:
            name = "chat_distance2_part"
        else:
            name = "chat_distance2_all"
    else:
        name = "chat_distance_" + str(year)
    if down_sampling != 0:
        name += "_down_"+ str(down_sampling)
    name += ".json"
    f = open(path + name,"w")
    f.write(json_people)
    f.close()

def get_pattern1(): # get pattern on percentage
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    l = []
    for key in dic.keys():
        count_hgy = 0
        count_enron = 0
        for s in dic[key]:
            for p in ['hotmail','gmail','yahoo']:
                if p in s[0]:
                    count_hgy += 1
            if 'enron' in s[0]:
                count_enron += 1
        length = len(dic[key])
        l.append(( count_enron/length, count_hgy/length, 1-count_enron/length- count_hgy/length))


def draw_seaborn(d, name):
    sns.set_theme(style="white")
    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(path + "figure/enron/new/" + name+".png")

def draw_hist_seaborn(d, name, label = [0,0.5,1,2]):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    for i in d.keys():
        ax = sns.kdeplot(d[i],label= 'ratio >= ' + str(i),shade = True)
    ax.set_xlim(0,1)
    ax.set(title = name)
    plt.legend()
    plt.savefig(path + "figure/enron/new/hist_" + name+".png")
    plt.close()
def remove_stop_words(text, words):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in words]
    return len(text_tokens),len(tokens_without_sw)

def get_pattern2(): # get pattern for each one
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    for key in dic.keys():
        l = [[],[],[],[]] # inside, outside, length, last reach time(by minute)    
        dic_tmp = {}
        for s in dic[key]:
            l[0].append(int('enron' in s[0]))
            l[1].append(int('enron' not in s[0]))
            l[2].append(s[1])
            time = parser.parse(s[3])
            if s[0] in dic_tmp:
                delta = time - dic_tmp[s[0]]
                delta = delta.days * 24 + delta.seconds/3600
            else:
                delta = 0
            dic_tmp[s[0]] = time
            l[3].append(delta)
        l = pd.DataFrame(data = l).T
        l.columns = ['inside','outside','length','time']
        name = key.split("@")[0]
        draw_seaborn(l, name)
        print(1)

def get_pattern3(): # get pattern for each one
    f = open(path + "chat_distance2_all.json",'r')
    post = 'all'
    dic = json.load(f)
    #for key in dic.keys():
    for key in ['kenneth.lay@enron.com','mike.grigsby@enron.com','kam.keiser@enron.com','marie.heard@enron.com','stephanie.panus@enron.com','richard.shapiro@enron.com','john.arnold@enron.com','kim.ward@enron.com']:
        if key == 'kim.ward@enron.com':
            print(1)
        l = [[],[],[],[],[],[],[]] # inside, outside, length, length_ratio,last reach time(by minute), length of two path nums
        dic_tmp = {}
        for s in dic[key]:
            l[0].append(s[5]) # inside
            l[1].append(int('enron' not in s[0])) # outside
            l[2].append(s[1]) # length_remove_stop_words
            time = parser.parse(s[3])
            if s[0] in dic_tmp:
                delta = time - dic_tmp[s[0]]
                delta = delta.days * 24 + delta.seconds/3600
            else:
                delta = 0
            dic_tmp[s[0]] = time
            l[3].append(s[4]) # length_ratio
            l[4].append(delta) # reply time
            l[5].append(s[7]) # path num
            # l[7].append(s[8]) # shortest path
            l[6].append(s[6]) # rank in a day

        l = pd.DataFrame(data = l).T
        l.columns = ['inside','outside','length','length ratio','time','number of 2-paths', 'rank']
        name = key.split("@")[0] + post
        draw_seaborn(l, name)

def pattern4_bin_helper(stage, keys):
    res = []
    for i,key in enumerate(keys):
        if stage >= key:
            res.append(key)
    return res
        

def get_pattern4(): # get pattern for stage vs length, stage vs time
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    for key in ['kenneth.lay@enron.com','mike.grigsby@enron.com','kam.keiser@enron.com','marie.heard@enron.com','stephanie.panus@enron.com','richard.shapiro@enron.com','john.arnold@enron.com','kim.ward@enron.com']:
        dic_tmp = {}
        histgram_length = {0:[],1:[],2:[],3:[]} # 0,0.5,1,2
        histgram_time =  {0:[],1:[],2:[],3:[]}
        for s in dic[key]:
            time = parser.parse(s[3])
            if s[2] == 1:
                res_stage = pattern4_bin_helper(1/s[5])
                if s[0] in dic_tmp:
                    delta = time - dic_tmp[s[0]]
                    delta = delta.days * 24 + delta.seconds/3600
                else:
                    delta = 0
                for r in res_stage:
                    histgram_length[r].append(s[4])
                    histgram_time[r].append(delta)  
            else:
                dic_tmp[s[0]] = time
        draw_hist_seaborn(histgram_length, 'length_'+key)  
        draw_hist_seaborn(histgram_time, 'time_'+key)   


def get_pattern5(): # get pattern for stage vs length, stage vs time all together
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    keys = [0, 0.5, 1, 2, 5]
    histgram_length, histgram_time = {},{}
    for key in keys:
        histgram_length[key] = [] # 0,0.5,1,2
        histgram_time[key] = []
    for key in dic.keys():
        dic_tmp = {}  
        for i,s in enumerate(dic[key]):
            time = parser.parse(s[3])
            if s[2] == 1:
                res_stage = pattern4_bin_helper(1/s[5], keys)
                if s[0] in dic_tmp:
                    delta = time - dic_tmp[s[0]]
                    delta = delta.days * 24 + delta.seconds/3600
                else:
                    delta = 0
                for r in res_stage:
                    histgram_length[r].append(s[4])
                    if delta != 0:
                        histgram_time[r].append(delta)  
            else:
                dic_tmp[s[0]] = time
    draw_hist_seaborn(histgram_length, 'length_all', keys)  
    draw_hist_seaborn(histgram_time, 'time_all', keys)   



# start dataset 8

csv_file = pd.read_csv(path + dataset[7])
org = pd.read_csv(path + 'organazition2.csv')
dic_stage = {}
for i in range(len(org)):
    dic_stage[org.iloc[i]['Email']] = org.iloc[i]['Stage'] 

# resort by date
csv_file['Date'] = pd.to_datetime(csv_file['Date'])
csv_file.sort_values(["Date"],axis=0, ascending=True,inplace=True,na_position='first')
csv_file.to_csv(path+'sorted_emails1.csv',index=False)


# high_freqeuncy_all(csv_file, down_sampling = 0)
# high_freqeuncy_all(csv_file, down_sampling = 0, start_i = 400000)
# high_freqeuncy_all(csv_file, down_sampling = 0.3)
# high_freqeuncy_all(csv_file, down_sampling = 0.3, start_i = 400000)
# high_freqeuncy_all(csv_file, year = 1999)
# high_freqeuncy_all(csv_file, year = 2000)
high_freqeuncy_all(csv_file, year = 2001)
# high_freqeuncy_all(csv_file, year = 2002)

# Date From to x-from ~ x-cc 
