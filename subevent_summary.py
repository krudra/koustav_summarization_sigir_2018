#!/usr/bin/python2

import sys
from collections import Counter
import re
from textblob import *
from gurobipy import *
import gzip
import os
import time
import codecs
import math
import networkx as nx
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic, genesis
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import aspell
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import pylab as pl
from itertools import cycle
from operator import itemgetter

LOWLIMIT = 0
UPPERLIMIT = 1

WT1 = 0.2
WT2 = 0.5
WT3 = 0.3

LSIM = 0.7
lmtzr = WordNetLemmatizer()
ASPELL = aspell.Speller('lang', 'en')
WORD = re.compile(r'\w+')

cachedstopwords = stopwords.words("english")
AUX = ['be','can','cannot','could','am','has','had','is','are','may','might','dare','do','did','have','must','need','ought','shall','should','will','would','shud','cud','don\'t','didn\'t','shouldn\'t','couldn\'t','wouldn\'t']
NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]


def compute_summary(ifname,parsefile,eventfile,placefile,keyterm,date,Ts):

	###################################### Read Place Information ############################################################
	PLACE = {}
        fp = codecs.open(placefile,'r','utf-8')
        for l in fp:
                if PLACE.__contains__(l.strip(' \t\n\r').lower())==False:
                	PLACE[l.strip(' \t\n\r').lower()] = 1
        fp.close()

	######################################## Processing Parse File ###########################################################
	
	RPL = ['+','-',',','+91']
	index = 0
	count = 0
	dic = {}
	L = 0
	TAGREJECT = ['#','@','~','U','E','G',',']

	fp = codecs.open(parsefile,'r','utf-8')
	fs = codecs.open(eventfile,'r','utf-8')
	ft = codecs.open(ifname,'r','utf-8')

        t0 = time.time()
	T = {}
	CT = {}
	SCT = {}
	content_count = {}
	notag_content_count = {}
	topic_count = {}
	TOPIC_SET = set([])

	for l in fp:
		wl = l.split('\t')
		if len(wl)==8:
                        seq = int(wl[0])
                        main_word = wl[1].strip(' #\t\n\r').lower()
                        word = wl[1].strip(' #\t\n\r').lower()
                        tag = wl[4].strip(' \t\n\r')
                        dep = wl[6].strip(' \t\n\r')
                        if dep=='_':
                                dep = int(wl[7].strip(' \t\n\r'))
                        else:
                                dep = int(wl[6])


                        if tag=='$':
                                s = word.strip(' \t\n\r')
                                Q = s
                                for x in RPL:
                                        Q = s.replace(x,'')
                                        s = Q
                                Q = s.lstrip('0')
                                s = Q
                                try:
                                	w = str(numToWord(int(s)))
                               		if len(w.split())>1: # like 67
                                        	w = s
                            	except Exception as e:
                                	w = str(s)
                                word = w.lower()
                        elif tag=='N':
				try:
                                	w = lmtzr.lemmatize(word)
                                	word = w.lower()
				except Exception as e:
					pass
                        elif tag=='^':
				try:
                                	w = lmtzr.lemmatize(word)
                                	word = w.lower()
				except Exception as e:
					pass
				tag = 'N'
                        elif tag=='V':
                                try:
                                        w = Word(word.lower())
                                        x = w.lemmatize("v")
                                except Exception as e:
                                        x = word.lower()
                                word = x.lower()
                        else:
				pass

                        temp = [word,tag,dep,main_word]
                        dic[seq] = temp
		else:
			
			#################### Subevent Extraction ############################################################
				
			EVENT = extract_events(fs.readline().strip(' \t\n\r'))
			ev = []
			for k,v in dic.iteritems():
				if v[1]=='V' and v[0] not in AUX and v[0] not in cachedstopwords and ASPELL.check(v[0])==1 and len(v[0])>1 and v[0] not in NEGATE:
					ev.append(k)
				elif EVENT.__contains__(v[0])==True and v[0] not in AUX and v[0] not in cachedstopwords and ASPELL.check(v[0])==1 and len(v[0])>1 and v[0] not in NEGATE:
					ev.append(k)
				else:
					pass
			topic = set([])
			for k,v in dic.iteritems():
				try:
					if v[2] in ev and v[1]=='N' and ASPELL.check(v[0])==1 and len(v[0])>1:
						topic.add((v[0],dic[v[2]][0]))
						TOPIC_SET.add((v[0],dic[v[2]][0]))
						#T[(v[0],dic[v[2]][0])] = 1
				except:
					print(v)

			content = set([])
			TL = ft.readline().split('\t')
			temp = TL[3].split()
			for x in temp:
				x_0 = x.split('_')[0].strip(' \t\n\r')
				x_1 = x.split('_')[1].strip(' \t\n\r')
				if x_1=='PN':
					s = x_0 + '_CN'
					content.add(s)
				else:
					content.add(x)
			All = set([])
			temp = TL[4].split()
			for x in temp:
				x_0 = x.split('_')[0].strip(' \t\n\r')
				x_1 = x.split('_')[1].strip(' \t\n\r')
				if x_1=='PN':
					s = x_0 + '_CN'
					All.add(s)
				else:
					All.add(x)
				
			L = int(TL[5])
			
			######################### SET COUNT #################################################################

			for x in content:
				x10 = x.split('_')[0].strip(' \t\n\r')
				if notag_content_count.__contains__(x10)==True:
					v = notag_content_count[x10]
					v+=1
					notag_content_count[x10] = v
				else:
					notag_content_count[x10] = 1
			
			for x in content:
				if content_count.__contains__(x)==True:
					v = content_count[x]
					v+=1
					content_count[x] = v
				else:
					content_count[x] = 1

			for x in topic:
				if topic_count.__contains__(x)==True:
					v = topic_count[x]
					v+=1
					topic_count[x] = v
				else:
					topic_count[x] = 1

			
			k = should_select(SCT,All)
			if k==1:
				CT[index] = content
				SCT[index] = All
				T[index] = [TL[2].strip(' \t\n\r'),content,topic,L]
				index+=1

			dic = {}
			count+=1
       
	fp.close()
	fs.close()
	ft.close()
	print(count,index)


	CONTENT_WEIGHT = compute_tfidf_NEW(content_count,count,PLACE)
	TOPIC_WEIGHT = compute_pmi_topic(topic_count,notag_content_count)
	NORM_CONTENT_WEIGHT = set_weight(CONTENT_WEIGHT,LOWLIMIT,UPPERLIMIT)
	NORM_TOPIC_WEIGHT = TOPIC_WEIGHT

	########################################### Update Tweet Set (topic to cluster) ########################################
	
	TW = {}
	for i in range(0,index,1):
		v = T[i]
		temp = v[2]
		mod_sub = set([])
		for x in temp:
			mod_sub.add(x)

		TW[i] = [v[0],v[1],mod_sub,v[3],1]
	

	########################################## Summarize Tweets #############################################################

	L = len(TW.keys())
        tweet_cur_window = {}
        for i in range(0,L,1):
                temp = TW[i]
                tweet_cur_window[i] = [temp[0].strip(' \t\n\r'),int(temp[3]),temp[1],temp[2],float(temp[4])] # tweet, length, content, topic

        ofname = keyterm + '_test_' + date + '.txt'
        optimize(tweet_cur_window,NORM_CONTENT_WEIGHT,NORM_TOPIC_WEIGHT,ofname,Ts,0.4,0.6)
        t1 = time.time()
        print('Summarization done: ',ofname,' ',t1-t0)

def compute_similarity(S1,S2):
	common = set(S1).intersection(set(S2))
	X = len(common) + 4.0 - 4.0
	Y = min(len(S1),len(S2)) + 4.0 - 4.0
	if Y==0:
		return 0
	Z = round(X/Y,4)
	return Z

def should_select(T,new):
        if len(new)==0:
                return 0
        for i in range(0,len(T),1):
                temp = T[i]
                common = set(temp).intersection(set(new))
                if len(common)==len(new):
                       return 0
        return 1

def extract_events(line):
	EVENT = {}
	wl = line.split()
	for w in wl:
		X = w.split('/')
		if X[len(X)-1]=='B-EVENT' and X[len(X)-2].startswith('V')==True and len(X[0])>2:
			word = X[0].strip(' \t\n\r').lower()
			try:
				y = Word(word)
				z = y.lemmatize("v")
				word = z
			except Exception as e:
				print(e)
				pass
			if EVENT.__contains__(word)==False:
				EVENT[word] = 1
        return EVENT
		
def set_weight(P,L,U):
        min_p = min(P.values())
        max_p = max(P.values())

        x = U - L + 4.0 - 4.0
        y = max_p - min_p + 4.0 - 4.0
        factor = round(x/y,4)

        mod_P = {}
	for k,v in P.iteritems():
		val = L + factor * (v - min_p)
		mod_P[k] = round(val,4)

        count = 0
        return mod_P

def optimize(tweet,con_weight,sub_weight,ofname,L,A1,A2):


        ################################ Extract Tweets and Content Words ##############################
        con_word = {}
	sub_word = {}
        tweet_word = {}
        tweet_index = 1
        for  k,v in tweet.iteritems():
                set_of_words = v[2]
                for x in set_of_words:
                	if con_word.__contains__(x)==False:
                                if con_weight.__contains__(x)==True:
                                        p1 = round(con_weight[x],4)
                                else:
                                        p1 = 0.0
                                con_word[x] = p1 * WT2
                
		set_of_subs = v[3]
                for x in set_of_subs:
                	if sub_word.__contains__(x)==False:
                                if sub_weight.__contains__(x)==True:
                                        p1 = round(sub_weight[x],4)
                                else:
                                        p1 = 0.0
                                sub_word[x] = p1 * WT3

                tweet_word[tweet_index] = [v[1],set_of_words,set_of_subs,v[0],v[4]]  #Length of tweet, set of content words present in the tweet, set of subevents present in the tweet, tweet itself
                tweet_index+=1

        ############################### Make a List of Tweets ###########################################
        sen = tweet_word.keys()
        sen.sort()
        entities = con_word.keys()
	subevents = sub_word.keys()
        print(len(sen),len(entities),len(subevents))

        ################### Define the Model #############################################################

        m = Model("sol1")

        ############ First Add tweet variables ############################################################

        sen_var = []
        for i in range(0,len(sen),1):
                sen_var.append(m.addVar(vtype=GRB.BINARY, name="x%d" % (i+1)))

        ############ Add entities variables ################################################################

        con_var = []
        for i in range(0,len(entities),1):
                con_var.append(m.addVar(vtype=GRB.BINARY, name="y%d" % (i+1)))
        
	############ Add subevents variables ################################################################

        sub_var = []
        for i in range(0,len(subevents),1):
                sub_var.append(m.addVar(vtype=GRB.BINARY, name="z%d" % (i+1)))

        ########### Integrate Variables ####################################################################
        m.update()

        P = LinExpr() # Contains objective function
        C1 = LinExpr()  # Summary Length constraint
        C4 = LinExpr()  # Summary Length constraint
        C2 = [] # If a tweet is selected then the content words are also selected
        counter = -1
        for i in range(0,len(sen),1):
                P += sen_var[i] * WT1
                C1 += tweet_word[i+1][0] * sen_var[i]
                v = tweet_word[i+1][1] # Entities present in tweet i+1
                C = LinExpr()
                flag = 0
                for j in range(0,len(entities),1):
                        if entities[j] in v:
                                flag+=1
                                C += con_var[j]
                if flag>0:
                        counter+=1
                        m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))
                
                v = tweet_word[i+1][2] # Subevents present in tweet i+1
		C = LinExpr()
                flag = 0
                for j in range(0,len(subevents),1):
                        if subevents[j] in v:
                                flag+=1
                                C += sub_var[j]
                if flag>0:
                        counter+=1
                        m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))


        for i in range(0,len(entities),1):
                P += con_word[entities[i]] * con_var[i]
                C = LinExpr()
                flag = 0
                for j in range(0,len(sen),1):
                        v = tweet_word[j+1][1]
                        if entities[i] in v:
                                flag = 1
                                C += sen_var[j]
                if flag==1:
                        counter+=1
                        m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))

	for i in range(0,len(subevents),1):
                P += sub_word[subevents[i]] * sub_var[i]
                C = LinExpr()
                flag = 0
                for j in range(0,len(sen),1):
                        v = tweet_word[j+1][2]
                        if subevents[i] in v:
                                flag = 1
                                C += sen_var[j]
                if flag==1:
                        counter+=1
                        m.addConstr(C,GRB.GREATER_EQUAL,sub_var[i], "c%d" % (counter))

        counter+=1
        m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))


        ################ Set Objective Function #################################
        m.setObjective(P, GRB.MAXIMIZE)

        ############### Set Constraints ##########################################

        fo = codecs.open(ofname,'w','utf-8')
        try:
                m.optimize()
                for v in m.getVars():
                        if v.x==1:
                                temp = v.varName.split('x')
                                if len(temp)==2:
					X = ''
					EV = tweet_word[int(temp[1])][2]
					if len(EV)!=0:
						for x in EV:
							X = X + x[0] + '$#@' + x[1] + ' '
						X = X.strip(' ')
					else:
						X = 'NIL'
                                        fo.write(tweet_word[int(temp[1])][3])
                                        fo.write('\n')
        except GurobiError as e:
                print(e)
                sys.exit(0)

        fo.close()

def compute_tfidf_NEW(word,tweet_count,PLACE):
        score = {}
        discard = []
        #THR = int(round(math.log10(tweet_count),0))
        THR = 5
        N = tweet_count + 4.0 - 4.0
        for k,v in word.iteritems():
                D = k.split('_')
                D_w = D[0].strip(' \t\n\r')
                D_t = D[1].strip(' \t\n\r')
                if D_w not in discard:
                        tf = v
                        w = 1 + math.log(tf,2)
                        #w = tf
                        df = v + 4.0 - 4.0
                        #N = tweet_count + 4.0 - 4.0
                        try:
                                y = round(N/df,4)
                                idf = math.log10(y)
                        except Exception as e:
                                idf = 0
                        val = round(w * idf, 4)
                        if D_t=='P' and tf>=THR:
                                score[k] = val
                        elif tf>=THR and D_t=='S':
                                score[k] = val
                        elif tf>=THR and len(D_w)>2:
                                score[k] = val
                        else:
                                score[k] = 0
                else:
                        score[k] = 0
        return score

def compute_pmi_topic(topic_count,content_count):
	TOPIC = []
        for k,v in topic_count.iteritems():
                try:
                        x1 = k[0]
                        x2 = k[1]
                        w1 = content_count[x1]
                        w2 = content_count[x2]
                        y1 = v + 4.0 - 4.0
                        #y2 = (w1 * w2) + 4.0 - 4.0
                        y2 = min(w1,w2) + 4.0 - 4.0
                        #y2 = w1 + w2  + 4.0 - 4.0
                        #z = math.log(y1*count/y2,2)
                        z = round(y1/y2,4)

                        p1 = v + 4.0 - 3.0
                        p2 = y1/p1
                        q1 = min(w1,w2) + 4.0 - 4.0
                        q2 = min(w1,w2) + 4.0 - 3.0
                        q3 = q1/q2

                        PMI = round(z*p2*q3,4)
                        #PMI = z
                        #if w1 >=10 and w2 >= 10 and v>=10:
                        TOPIC.append((k,v,w1,w2,PMI))
                except Exception as e:
                        pass
	score = {}
	for T in TOPIC:
		if T[1]>=5 and T[2]>=10 and T[3]>=10:
			score[T[0]] = T[4]
		else:
			score[T[0]] = 0
	print('Score count: ',len(score.keys()))
	return score

def numToWord(number):
        word = []
        if number < 0 or number > 999999:
                return number
                # raise ValueError("You must type a number between 0 and 999999")
        ones = ["","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
        if number == 0: return "zero"
        if number > 9 and number < 20:
                return ones[number]
        tens = ["","ten","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
        word.append(ones[int(str(number)[-1])])
        if number >= 10:
                word.append(tens[int(str(number)[-2])])
        if number >= 100:
                word.append("hundred")
                word.append(ones[int(str(number)[-3])])
        if number >= 1000 and number < 1000000:
                word.append("thousand")
                word.append(numToWord(int(str(number)[:-3])))
        for i,value in enumerate(word):
                if value == '':
                        word.pop(i)
        return ' '.join(word[::-1])


def main():
	try:
		_, ifname, parsefile, eventfile, placefile, keyterm, date, Ts = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	compute_summary(ifname,parsefile,eventfile,placefile,keyterm,date,int(Ts))
	print('Koustav Done')

if __name__=='__main__':
	main()
