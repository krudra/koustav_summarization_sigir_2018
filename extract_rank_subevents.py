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

def compute_summary(parsefile,eventfile,ofname):


	######################################## Processing Parse File ###########################################################
	
	RPL = ['+','-',',','+91']
	index = 0
	count = 0
	dic = {}
	L = 0
	TAGREJECT = ['#','@','~','U','E','G',',']

	fp = open(parsefile,'r')
	fs = open(eventfile,'r')

        t0 = time.time()
	T = {}
	CT = {}
	SCT = {}
	content_count = {}
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
                                	#count+=1
                                	word = w.lower()
				except Exception as e:
					pass
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
			if tag not in TAGREJECT:
				L+=1
		else:
			
			#################### Content Word Extraction ######################################################
			content = set([])
			All = set([])
			for k,v in dic.iteritems():
				if v[1] not in TAGREJECT:
					All.add(v[0])
				if v[1]=='N' or v[1]=='V':
					if ASPELL.check(v[0])==1 and len(v[0])>1:
						if v[0] not in AUX and v[0] not in cachedstopwords and v[0] not in NEGATE:
							content.add(v[0])
				elif v[1]=='$':
					content.add(v[0])
				else:
					pass

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
				if v[2] in ev and v[1]=='N' and ASPELL.check(v[0])==1 and len(v[0])>1:
					topic.add((v[0],dic[v[2]][0]))
					TOPIC_SET.add((v[0],dic[v[2]][0]))

			######################### SET COUNT #################################################################

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

			dic = {}
			L = 0
			count+=1
       
	fp.close()
	fs.close()

	TOPIC = []
	for k,v in topic_count.iteritems():
		try:
			x1 = k[0]
			x2 = k[1]
			w1 = content_count[x1]
			w2 = content_count[x2]
			y1 = v + 4.0 - 4.0
			y2 = min(w1,w2) + 4.0 - 4.0
			z = round(y1/y2,4)

			p1 = v + 4.0 - 3.0
			p2 = y1/p1
			q1 = min(w1,w2) + 4.0 - 4.0
			q2 = min(w1,w2) + 4.0 - 3.0
			q3 = q1/q2

			PMI = round(z*p2*q3,4)
			TOPIC.append((k,v,w1,w2,PMI))
		except Exception as e:
			pass
	
	fo = codecs.open(ofname,'w','utf-8')
	TOPIC.sort(key=itemgetter(4),reverse=True)
	for x in TOPIC:
		s = x[0][0].strip(' \t\n\r') + '\t' + x[0][1].strip(' \t\n\r') + '\t' + str(x[1]) + '\t' + str(x[2]) + '\t' + str(x[3]) + '\t' + str(x[4])
		fo.write(s + '\n')
	fo.close()

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
		_, parsefile, eventfile, ofname = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	compute_summary(parsefile,eventfile,ofname)
	print('Done')

if __name__=='__main__':
	main()
