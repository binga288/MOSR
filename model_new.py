# use model_new not model this is our final version
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
# from lib2to3.refactor import get_all_fix_names
from xml.etree.ElementInclude import default_loader
from dateutil import parser
from heapq import heappush,heappop
from statistics import mean

import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import os

def init_weights():
    return [1,1]

class Model:
    def __init__(self, weights, alphas, time_window_closeness, time_window_timeliness, owa_parameter, distance_measure="two_path_num", learning_rate = 0.99):
        """initilization

        Args:
            weights (_type_): _description_
            alphas (_type_): _description_
            time_window_closeness (_type_): _description_
            distance_measure (_type_): _description_
        """
        self.weights = weights
        self.alphas = alphas
        self.time_window_closeness = time_window_closeness
        self.time_window_timeliness = time_window_timeliness
        self.owa_parameter = owa_parameter
        self.distance_measure = distance_measure
        # initilize exisiting functions
        self.learning_functions = [self.OWA1, self.OWA2, self.OWA3, self.OWA4, self.timeline, self.WOWA]
        self.k = len(self.learning_functions) # how many functions we have
        # initilize OWA weights
        self.Q_function()
        # initilize our model weights 
        self.user_weights = {}
        self.learning_rate = learning_rate
        # time window expire
        self.expire_window = 3
        # self.max_d indicate those who didn't appear in true reply but in candidates
        self.max_d = 0
        self.stat_pred = []
        self.stat_true = []
    # def insider(self, s, dic_followup):
    #     time = parser.parse(s[3])
    #     if s[0] in dic_followup:
    #         delta = time - dic_followup[s[0]]
    #         delta = (delta.days * 24 + delta.seconds/3600)/24
    #     else:
    #         delta = 0
    #     dic_followup[s[0]] = time

    # def ousider(self, chat_history,i,time_window):
    #     time = parser.parse(chat_history[i][3])
    #     self.frequency(chat_history[:i], time-time_window)

    def distance(self,chat_history_i):
        if self.distance_measure == "two_path_num":
            return math.exp(chat_history_i[7])
        elif self.distance_measure == "shortest_path":
            return math.exp(chat_history_i[8])
        else:
            return 1

    def frequency(self, chat_history, time_stamp):
        """get the frequency 

        Args:
            chat_history (list): the partial chat history between e_i and e_j
            time_stamp (datetime): t_i - w.
        Returns:
            the to/from frequency
        """
        # chat_history[i][2] indicate the direction. -1 is 'from' (e_j->e_i). 1 is 'to' (e_i->e_j).
        f = {-1:0,1:0}
        for i in range(len(chat_history)-1,-1,-1):
            cur_time = parser.parse(chat_history[i][3])
            if cur_time >= time_stamp:
                f[chat_history[i][2]] += 1
            else:
                break
        f = list(f.values())
        gamma_frequency = math.exp(sum([f_i*weight_i for f_i, weight_i in zip(f, self.weights)]))
        return gamma_frequency
    
    def closeness(self, chat_history, flag):
        """get the closeness

        Args:
            chat_history (list): the chat history between e_i and e_j
            i: the index of current chat history
            
            flag (int): indicates insider closeness or outsider closeness
            weights (list): w_1 and w_2. weights for to-frequency and from-frequency. 
        """
        cur_time = parser.parse(chat_history[-1][3])
        # time_window (datetime): w. default 1 day. 
        frequency = self.frequency(chat_history[:-1],cur_time - self.time_window_closeness)
        # insider
        if flag == True:
            stage_ratio = math.exp((0.5/chat_history[-1][5]))
            return frequency * stage_ratio
        else:
            distance = self.distance(chat_history[-1])
            return frequency * distance
    
    def time(self, content, flag):
        if flag == 1:
            dic = self.send_time
            time_window_timeliness = self.time_window_timeliness
        else:
            dic = self.reply_time
            time_window_timeliness = 0
        time = parser.parse(content[3])
        if content[0] in dic:
            delta = time - dic[content[0]]
            delta = (delta.days * 24 + delta.seconds/3600)/24
        else:
            delta = 0
        delta = delta if delta > time_window_timeliness else 0
        return math.exp(delta)

    def timeliness(self,content):
        send_time = self.time(content,1)
        reply_time = self.time(content,-1)
        xi_timeliness = self.alphas[0] * send_time + self.alphas[1] * reply_time
        if send_time != 0:
            self.time(content,1)
        if reply_time != 0:
            self.time(content,-1)
        return xi_timeliness


    def exp(self,p,alpha):
        return math.pow(p,alpha)
    
    def pow(self,p,alpha):
        return math.pow(alpha,p)

    def constant(self,p,alpha):
        return p

    def Q_function(self, type = "exp"):
        if type == "exp": # p ^ \alpha
            func = self.exp
        elif type == 'pow': # \alpha ^ p
            func = self.pow
        else:
            func = self.constant
        self.Q_funcs = []
        for i in range(self.k):
            self.Q_funcs.append(func(i+1, self.owa_parameter)-func(i, self.owa_parameter))

    def check_expire_time(self,cur_time_date):
        for key in self.send_time:
            for i in range(len(self.send_time[key])):
                if (cur_time_date - self.send_time[key][i][-1].date()).days < self.expire_window:
                    break
            i = i-1
            self.send_time[key] = self.send_time[key][i:]
            if len(self.send_time[key]) == 0:
                del(self.send_time[key])
        key_to_del = []
        for key in self.reply_time.keys():
            for i in range(len(self.reply_time[key])):
                if (cur_time_date - self.reply_time[key][i][-1].date()).days < self.expire_window:
                    break
            i = i-1
            self.reply_time[key] = self.reply_time[key][i:]
            if len(self.reply_time[key]) == 0:
                key_to_del.append(key)
        for key in key_to_del:
            del(self.reply_time[key])

    def OWA1(self, send_time, reply_time, cur_time):
        scores = []
        candidates = set(send_time.keys()).union(set(reply_time.keys()))
        candidates = candidates.union(self.cc_set.keys())
        dic_all = {}
        for key in candidates:
            if key in send_time:
                act_send = deepcopy(send_time[key])
            else:
                act_send = []
            if key in reply_time:
                act_reply = deepcopy(reply_time[key])
            else:
                act_reply = []
            if key in self.cc_set:
                act_cc = deepcopy(self.cc_set[key])
            else:
                act_cc = []
            # time for act_send, without 1-
            for i, act in enumerate(act_send):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_send[i] = act
            # time for act_reply, without 1-
            for i, act in enumerate(act_reply):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_reply[i] = act
            # time for act_cc, without 1-
            for i, act in enumerate(act_cc):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_cc[i] = act
            acts = act_send + act_reply + act_cc
            dic_all[key] = np.mean(np.array(acts),axis = 0)
        for act in dic_all.values():
            try:
                sorted_act = sorted(act[:self.k])
            except:
                print(1)
            scores.append(sum([Q * a for Q, a in zip(self.Q_funcs, sorted_act)]))
        rankings = np.argsort(np.array(scores))
        # return result, key is email, value is rank
        dic_res = {}
        for rank, key in zip( rankings, list(dic_all.keys())):
            dic_res[key] = rank
        return dic_res
    
    def OWA2(self, send_time, reply_time, cur_time):
        scores = []
        candidates = set(send_time.keys()).union(set(reply_time.keys()))
        candidates = candidates.union(self.cc_set)
        dic_all = {}
        for key in candidates:
            if key in send_time:
                act_send = deepcopy(send_time[key])
            else:
                act_send = []
            if key in reply_time:
                act_reply = deepcopy(reply_time[key])
            else:
                act_reply = []
            if key in self.cc_set:
                act_cc = deepcopy(self.cc_set[key])
            else:
                act_cc = []
            # time for act_send, with 1-
            for i, act in enumerate(act_send):
                delta = cur_time - act[-1]
                act[-1] = 1 - (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_send[i] = act
            # time for act_reply, without 1-
            for i, act in enumerate(act_reply):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_reply[i] = act
            # time for act_cc, without 1-
            for i, act in enumerate(act_cc):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_cc[i] = act
            acts = act_send + act_reply + act_cc
            dic_all[key] = np.mean(np.array(acts),axis = 0)
        for act in dic_all.values():
            sorted_act = sorted(act[:self.k])
            scores.append(sum([Q * a for Q, a in zip(self.Q_funcs, sorted_act)]))
        rankings = np.argsort(np.array(scores))
        # return result, key is email, value is rank
        dic_res = {}
        for rank, key in zip( rankings, list(dic_all.keys())):
            dic_res[key] = rank
        return dic_res

    def OWA3(self, send_time, reply_time, cur_time):
        scores = []
        candidates = set(send_time.keys()).union(set(reply_time.keys()))
        candidates = candidates.union(self.cc_set.keys())
        dic_all = {}
        for key in candidates:
            if key in send_time:
                act_send = deepcopy(send_time[key])
            else:
                act_send = []
            if key in reply_time:
                act_reply = deepcopy(reply_time[key])
            else:
                act_reply = []
            if key in self.cc_set:
                act_cc = deepcopy(self.cc_set[key])
            else:
                act_cc = []
            # time for act_send, without 1-
            for i, act in enumerate(act_send):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window # /self.expire_window to make it between 0 and 1
                act_send[i] = act
            # time for act_reply, with 1-
            for i, act in enumerate(act_reply):
                delta = cur_time - act[-1]
                act[-1] = 1 - (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_reply[i] = act
            # time for act_cc, without 1-
            for i, act in enumerate(act_cc):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window 
                act_cc[i] = act
            acts = act_send + act_reply + act_cc
            dic_all[key] = np.mean(np.array(acts),axis = 0)
        for act in dic_all.values():
            try:
                sorted_act = sorted(act[:self.k])
            except:
                print(1)
            scores.append(sum([Q * a for Q, a in zip(self.Q_funcs, sorted_act)]))
        rankings = np.argsort(np.array(scores))
        # return result, key is email, value is rank
        dic_res = {}
        for rank, key in zip( rankings, list(dic_all.keys())):
            dic_res[key] = rank
        return dic_res

    def OWA4(self, send_time, reply_time, cur_time):
        scores = []
        candidates = set(send_time.keys()).union(set(reply_time.keys()))
        candidates = candidates.union(self.cc_set.keys())
        dic_all = {}
        for key in candidates:
            if key in send_time:
                act_send = deepcopy(send_time[key])
            else:
                act_send = []
            if key in reply_time:
                act_reply = deepcopy(reply_time[key])
            else:
                act_reply = []
            if key in self.cc_set:
                act_cc = deepcopy(self.cc_set[key])
            else:
                act_cc = []
            # time for act_send, with 1-
            for i, act in enumerate(act_send):
                delta = cur_time - act[-1]
                act[-1] = 1 - (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_send[i] = act
            # time for act_reply, with 1-
            for i, act in enumerate(act_reply):
                delta = cur_time - act[-1]
                act[-1] = 1 - (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_reply[i] = act
            # time for act_cc, without 1-
            for i, act in enumerate(act_cc):
                delta = cur_time - act[-1]
                act[-1] = (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_cc[i] = act
            acts = act_send + act_reply + act_cc
            dic_all[key] = np.mean(np.array(acts),axis = 0)
        for act in dic_all.values():
            try:
                sorted_act = sorted(act[:self.k])
            except:
                print(1)
            scores.append(sum([Q * a for Q, a in zip(self.Q_funcs, sorted_act)]))
        rankings = np.argsort(np.array(scores))
        # return result, key is email, value is rank
        dic_res = {}
        for rank, key in zip( rankings, list(dic_all.keys())):
            dic_res[key] = rank
        return dic_res
    def WOWA(self, send_time, reply_time, cur_time):
        scores = []
        candidates = set(send_time.keys()).union(set(reply_time.keys()))
        candidates = candidates.union(self.cc_set.keys())
        dic_all = {}
        for key in candidates:
            if key in send_time:
                act_send = deepcopy(send_time[key])
            else:
                act_send = []
            if key in reply_time:
                act_reply = deepcopy(reply_time[key])
            else:
                act_reply = []
            if key in self.cc_set:
                act_cc = deepcopy(self.cc_set[key])
            else:
                act_cc = []
            # time for act_send, with increasing weighted value
            for i, act in enumerate(act_send):
                delta = cur_time - act[-1]
                act[-1] = w1 * (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_send[i] = act
            # time for act_reply, without 1-
            for i, act in enumerate(act_reply):
                delta = cur_time - act[-1]
                act[-1] = w2 * (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_reply[i] = act
            # time for act_cc, with shrinking weighted value
            for i, act in enumerate(act_cc):
                delta = cur_time - act[-1]
                act[-1] = w3 * (delta.days * 24 + delta.seconds/3600)/24/self.expire_window
                act_cc[i] = act
            acts = act_send + act_reply + act_cc
            dic_all[key] = np.mean(np.array(acts),axis = 0)
        for act in dic_all.values():
            try:
                sorted_act = sorted(act[:self.k])
            except:
                print(1)
            scores.append(sum([Q * a for Q, a in zip(self.Q_funcs, sorted_act)]))
        rankings = np.argsort(np.array(scores))
        # return result, key is email, value is rank
        dic_res = {}
        for rank, key in zip( rankings, list(dic_all.keys())):
            dic_res[key] = rank
        return dic_res
    def timeline(self,send_time, reply_time, cur_time): # newest rank first
        scores = []
        candidates = set(send_time.keys()).union(set(reply_time.keys()))
        candidates = candidates.union(self.cc_set)
        dic_all = {}
        for key in candidates:
            if key in send_time:
                act_send = deepcopy(send_time[key])
            else:
                act_send = []
            if key in reply_time:
                act_reply = deepcopy(reply_time[key])
            else:
                act_reply = []
            if key in self.cc_set:
                act_cc = deepcopy(self.cc_set[key])
            else:
                act_cc = []
            # time for act_send, with 1-
            act = act_send + act_reply + act_cc
            act.sort(key=lambda act: act[-1])
            dic_all[key] = act[0][-1]

        # return result, key is email, value is rank
        dic_res = {}
        rankings = sorted(dic_all.items(), key=lambda x: x[1], reverse= False)
        for i, name in enumerate(rankings):
            dic_res[name[0]] = i
        return dic_res
            
    def updates(self,errors):

        delta_weights = np.zeros(len(self.learning_functions))
        index = errors.index(min(errors))
        delta_weights[index] = 1
        return delta_weights

    def our_model(self,send_time, reply_time, cur_time, cur_user):
        rankings = []
        errors = []
        for func in self.learning_functions:
            func_res = func(send_time, reply_time, cur_time)
            func_err, _, _, _ = self.error_cal(func_res,self.tmp_send_time)
            rankings.append(func_res)
            errors.append(func_err)   
        # dic to numpy array
        np_rankings = []
        keys = rankings[0].keys()
        for ranking in rankings:
            s = [ranking[key] for key in keys]
            np_rankings.append(s)
        np_rankings = np.array(np_rankings)

        if cur_user not in self.user_weights:
            weights = self.updates(errors)
        else:
            weights = self.user_weights[cur_user]
        res = np.dot(weights,np_rankings)
        delta_weights = self.updates(errors)
        self.user_weights[cur_user] = (self.learning_rate * weights + delta_weights)/(1+self.learning_rate)
        res_dic = {}
        for key, r in zip(keys, res):
            res_dic[key] = r
        return res_dic
    
    def hierarchical_OWA():
        return

    def get_rank_of_res_true(self, res_true):
        dic_all = {}
        for key in res_true:
            act = deepcopy(res_true[key])
            act.sort(key=lambda act: act[-1])
            dic_all[key] = act[0][-3:]

        # return result, key is email, value is rank
        dic_res = {}
        rankings = sorted(dic_all.items(), key=lambda x: x[-1], reverse= True)
        for i, name in enumerate(rankings):
            dic_res[name[0]] = (i,name[1][:2]) # key two-path distance and minimum distance
        return dic_res

    def del_get(self, n):
        name  = n[0]
        time = n[-1]
        time = str(time)
        for j, content in enumerate(self.chat[name]):
            if content[3] == time:
                return j
        return -1

    def ndcg_cal(self, rank_res_true, res_pred):
        keys = set(rank_res_true.keys()).intersection(res_pred.keys())
        if len(keys) != 0:
            tmp_res_pred = []
            for key in keys:
                tmp_res_pred.append([res_pred[key],key])
            sorted_res_pred = sorted(tmp_res_pred)
            new_res_pred = {}
            for i in range(len(sorted_res_pred)):
                new_res_pred[sorted_res_pred[i][1]] = i
            keys = list(keys)
            dcg, idcg = 0, 0
            for key in keys:
                rel_i = len(rank_res_true) - new_res_pred[key]
                pos_i = rank_res_true[key][0] + 1
                # rel_i = 1/(1+res_pred[key])
                # i = 1/(1+rank_res_true[key][0])
                if pos_i == 1:
                    dcg += rel_i 
                else:
                    dcg += (rel_i)/math.log(pos_i,2)
            for i in range(len(rank_res_true)):
                rel_i, pos_i = len(rank_res_true) - i, i + 1
                # rel_i, i = 1/(1+i), 1/(1+i)
                if pos_i == 1:
                    idcg += rel_i 
                else:
                    idcg += (rel_i)/math.log(pos_i,2)
            ndcg = dcg/idcg
            return ndcg
        else:
            return 0

    def error_cal(self, res_pred, res_true):
        dist = 0
        l = 0
        rank_res_true = self.get_rank_of_res_true(res_true)
        # rank_res_true[key] = (idx, [2-path num, dist])
        stat_pred = 0
        stat_true = [0, 0, 0, 0, 0] # 0-index not communicated before, 1-index communicated before, 2-index 2-path num, key not in res_pred, total key
        ndcg = self.ndcg_cal(rank_res_true, res_pred)
        # # test the undiscovered candidates
        if len(rank_res_true) and self.flag == 0:
            t1 = set(rank_res_true.keys()) # t1 is the true
            t2 = set(res_pred.keys())
            a = list(t1)
            b = []
            for aa in a:
                if aa not in t2:
                    b.append(aa)
        #     a = b
        #     for i in range(len(a)):
        #         j = self.del_get([a[i],self.tmp_send_time[a[i]][0][4]])             
        #         print(self.chat[a[i]][max(0,j-5):j+1])
        #         print(1)
            
            

        for key in rank_res_true:
            l += 1
            if key in res_pred:
                dist += (res_pred[key] - rank_res_true[key][0]) ** 2
                stat_pred += 1
            else:
                dist += self.max_d ** 2
                if rank_res_true[key][1][1] == 1: # dist = 1, communicated before
                    stat_true[1] += 1
                else: 
                    stat_true[0] += 1
                if rank_res_true[key][1][0] != 0: # have 2-path 
                    stat_true[2] += 1
                stat_true[3] += 1
        if l == 0:
            error = 0
        else:
            error = dist/l
        # deal with divide by zero
        if len(res_pred) == 0:
            a = 0
        else:
            a = stat_pred/len(res_pred)
        # if len(rank_res_true) == 0:
        #     b = np.array(stat_true)
        # else:
        #     b = np.array(stat_true)/len(rank_res_true)
        stat_true[4] = len(rank_res_true)
        b = np.array(stat_true)
        return error, a, b, ndcg

    def model_calculate(self, prev_time, cur_time, e_i):
        # calculate the error
        tmp_err = []
        ndcgs = []
        for func in self.learning_functions:
            self.flag = 1 # delete later
            func_res = func(self.send_time, self.reply_time, cur_time)
            func_err, _, _, ndcg = self.error_cal(func_res,self.tmp_send_time)
            tmp_err.append(func_err)
            ndcgs.append(ndcg)
        # MRAC model
        func_res = self.our_model(self.send_time, self.reply_time, cur_time, e_i)
        self.flag = 0 # delete later
        func_err, stat_pred_acc_per, stat_true_per, ndcg =  self.error_cal(func_res,self.tmp_send_time)
        tmp_err.append(func_err)
        self.stat_pred.append(stat_pred_acc_per)
        self.stat_true.append(stat_true_per)
        ndcgs.append(ndcg)
        # NN model

        # all errors
        for k in range(len(tmp_err)):
            self.err[k].append(tmp_err[k])
            self.err_by_date[k][prev_time.date()].append(tmp_err[k])
            self.ndcgs[k].append(ndcgs[k])
            self.ndcg_by_date[k][prev_time.date()].append(ndcgs[k])
        # update send time
        for e_j, acts in self.tmp_send_time.items():
            if e_j not in self.send_time:
                self.send_time[e_j] = []
            self.send_time[e_j] += acts
            if e_j in self.reply_time:
                try:
                    self.reply_time[e_j] = self.reply_time[e_j][len(acts):]
                except:
                    print(1)
        self.tmp_send_time = {}
        self.cc_set = {}
        # check expire time
        self.check_expire_time(cur_time.date())

    def check_cc(self):
        # check change after add cc emails
        previous = set(self.send_time.keys()).union(set(self.reply_time.keys())) # predicted candidates
        new = previous.union(self.cc_set.keys()) # with cc 
        previous_sect = previous.intersection(self.tmp_send_time.keys()) # real vs send
        new_sect = new.intersection(self.tmp_send_time.keys()) # cc vs send
        return [len(previous_sect), len(new_sect), len(self.tmp_send_time)]

    def run(self, chat):
        self.chat = chat # delete later
        self.err = []
        self.err_by_date = []
        self.ndcg_by_date = []
        self.candidates_stat = []
        self.ndcgs = []
        for i in range(len(self.learning_functions) + 1):
            self.err.append([])
            self.err_by_date.append(defaultdict(list))
            self.ndcgs.append([])
            self.ndcg_by_date.append(defaultdict(list))
        loop = 0
        # initilization
        self.send_time = {}
        self.tmp_send_time = {}
        self.reply_time = {}
        
        for e_i in chat:
        # #     e_i = 'michelle.lokay@enron.com'
        # e_i = 'michelle.lokay@enron.com'
        # if e_i ==  'michelle.lokay@enron.com':
            # check whether length is 0 
            if self.send_time != {} or self.reply_time != {}: # self.tmp_send_time
                # print(e_i, end = ",")
                self.model_calculate(prev_time, cur_time, prev_e_i)
            prev_e_i = e_i
            chat_e_i = defaultdict(list)
            # send time: e_i send to e_j reply_time: e_i receive from e_j
            self.send_time = {}
            self.tmp_send_time = {}
            self.reply_time = {}
            self.cc_set = {}
            if not (chat[e_i] != [] and len(chat[e_i][0]) > 3):
                # print(e_i)
                continue
            prev_time = parser.parse(chat[e_i][0][3])
            # use cur_day_receive and prev_day_send
            replied_today = []
            sent_today = []
            for i, content in enumerate(chat[e_i]):
                self.content = (i,e_i) # deletecravenchrya
                # pre calculated
                if loop % 1000 == 0:
                    print(loop)
                loop += 1
                e_j = content[0]
                cur_time = parser.parse(content[3])
                chat_e_i[e_j].append(content)
                chat_history = chat_e_i[e_j]
                
                if cur_time.date() != prev_time.date():    
                    # these two lines aim to check cc stat  
                    tmp = self.check_cc()
                    self.candidates_stat.append(tmp)
                    # start model calculate
                    self.model_calculate(prev_time, cur_time, e_i)
                    prev_time = cur_time
                # begin model
                closeness = self.closeness(chat_history, 'enron' in e_j)
                verbosity = content[4]
                act = [closeness, verbosity, content[-3], content[-2], cur_time]
                for e_cc in content[-1]:
                    if e_cc not in self.cc_set:
                        self.cc_set[e_cc] = []
                    self.cc_set[e_cc].append([closeness+1, verbosity, content[-3], content[-2], cur_time]) # add
                # -1 is ei receive, 1 is ei send
                if content[2] == 1: # we send emails to others
                    # deal with send_time
                    if e_j not in self.tmp_send_time:
                        self.tmp_send_time[e_j] = []
                    self.tmp_send_time[e_j].append(act)
                    # # since we reply, update self.reply
                    # if e_j in self.reply_time:
                    #     self.reply[e_j] = self.reply[e_j][1:]
                    #     if len(self.reply_time[e_j]) == 0:
                    #         del(self.reply_time[e_j]) 
                
                # whether we need to reply
                if content[2] == -1: # we receive emails from others
                    # deal with reply time
                    if e_j not in self.reply_time:
                        self.reply_time[e_j] = []
                    self.reply_time[e_j].append(act)
                    # since we receive the reply from others, update self.send_time
                    if e_j in self.send_time:
                        self.reply_time[e_j] = self.reply_time[e_j][1:]
                        if len(self.reply_time[e_j]) == 0:
                            del(self.reply_time[e_j])
        return self.err, self.err_by_date

def draw_hist_seaborn(d, s = ['OWA1', 'OWA2', 'OWA3', 'OWA4' 'baseline','our']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    for i in range(len(d)):
        ax = sns.kdeplot(d[i],label= s[i],shade = True)
    ax.set_xlim(0,1)

    ax.set(title = 'Loss')
    plt.legend()
    filename = pre_s + "accuracy_all" + idx_figure + ".png"
    plt.savefig(path + "figure/enron/new/" + filename)
    plt.close()

def draw_curve_seaborn(d, s = ['OWA1', 'OWA2', 'OWA3', 'OWA4' 'baseline','our']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    for i in range(len(d)):
        ax = plt.plot(d[i],label= s[i],alpha = 0.4)
    # ax.set_xlim(0,60)
    # ax.title(title = 'Errors curve')
    plt.legend()
    filename = pre_s + "accuracy_all_curve" + idx_figure + ".png"
    plt.savefig(path + "figure/enron/new/" + filename)
    plt.close()

def main(pre_s, idx_figure):
    f = open(path + file_name,'r')
    chat = json.load(f)
    time_window_closeness = timedelta(hours = 24)
    time_window_timeliness = 0.5
    owa_parameter = 2.5
    # weights, alphas, time_window_closeness, time_window_timeliness, owa_parameter, distance_measure="two_path_num", k = 3):
    args = parse()
    model = Model([1,1],[1,1], time_window_closeness, time_window_timeliness, owa_parameter)
    model.max_d = args.max_d
    model.distance_measure = args.distance_measure
    model.learning_rate = args.learning_rate
    idx_figure += "max_d" + str(args.max_d) + "_" + args.distance_measure + "_" + "learning_rate" + str(args.learning_rate)
    print(idx_figure)
    # previous pre_s is distance2 all
    pre_s = "window" + str(model.expire_window) + "/" + pre_s + "_num" + idx_figure
    print(pre_s)
    if not os.path.exists(path + "model/" + "window" + str(model.expire_window)):
        os.makedirs(path + "model/" + "window" + str(model.expire_window))
    if args.model_load == True:
        try:
            with open(path + "model/" + pre_s + ".model","rb") as ef: # existing model
                model = pickle.load(ef)
            err = model.err
            err_by_date = model.err_by_date
        except:
            err, err_by_date = model.run(chat)
            with open(path + "model/" + pre_s + ".model","wb") as cf: # created mpdel
                pickle.dump(model, cf)
    else:
        err, err_by_date = model.run(chat)
        with open(path + "model/" + pre_s + ".model","wb") as cf: # created mpdel
            pickle.dump(model, cf)

    # Put run stats
    if not os.path.exists(path + "window" + str(model.expire_window)):
        os.makedirs(path + "window" + str(model.expire_window))
    with open( path + pre_s + "test_error", "wb") as fp:   #Pickling
        pickle.dump(err, fp)
    with open( path + pre_s + "test_error_by_date", "wb") as fp:   #Pickling
        pickle.dump(err_by_date, fp)
    with open( path + pre_s + "stat_true", "wb") as fp:   #Pickling
        pickle.dump(model.stat_true, fp)
    with open( path + pre_s + "stat_pred", "wb") as fp:   #Pickling
        pickle.dump(model.stat_pred, fp)
    with open( path + pre_s + "candidates", "wb") as fp:   #Pickling
        pickle.dump(model.candidates_stat, fp)
    with open( path + pre_s + "ndcgs", "wb") as fp:
        pickle.dump(model.ndcgs, fp)
    with open( path + pre_s + "ndcg_by_date", "wb") as fp:
        pickle.dump(model.ndcg_by_date, fp)
    error_by_date_sum = []
    keys = sorted(err_by_date[0].keys())
    for i in range(len(err_by_date)):
        tmp_error_sum = []
        for key in keys:
            tmp_error_sum.append(sum(err_by_date[i][key])/len(err_by_date[i][key]))
        error_by_date_sum.append(tmp_error_sum)
    with open( path + pre_s + "test_error_by_date_sum", "wb") as fp:   #Pickling
        pickle.dump(error_by_date_sum, fp)


def parse():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument("--max_d", "-md", type=int, default= 10, help="Int parameter")
    parser.add_argument("-distance_measure", "-dm", type = str, default = "two_path_num")
    parser.add_argument("-version","-v", type = int, default = 0)
    parser.add_argument("-down_sampling", "-ds", type = float, default= 0) # The part to delete
    parser.add_argument("-learning_rate","-lr", type = float, default = 0.99)
    parser.add_argument("-year", "-y", type = int, default = 0)
    parser.add_argument("-model_load", "-ml", type = bool, default = False)
    args = parser.parse_args()
    return args

# version 0 is all
# version 1 is part
if __name__ == '__main__':
    args = parse()
    path = '../data/'
    file_names = ['all', 'part', 'all_down', 'part_down']
    file_name = "chat_distance2_" + file_names[args.version]
    if args.version in [2,3]:
        file_name += '_' + str(args.down_sampling) 
    if args.year != 0 and args.year in [1999,2000,2001,2002]:
        file_name = "chat_distance_" + str(args.year)
    file_name += '.json'
    # file_name = "chat_distance2_part_down_0.3.json"
    idx_figure = "_version" + str(args.version) + "_"
    pre_s = file_name[5:-5]
    print(idx_figure)
    print(pre_s)
    main(pre_s, idx_figure)
    
    # distance2_all version0
    # distance2_part version1
    # distance2_all_down_0.3 version2
    # distance2_part_down_0.3 version3