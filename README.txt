# koustav_summarization_sigir_2018
Contains codes and datasets related to sub-event identification and summarizing information during disaster
Our SIGIR paper "Identifying Sub-events and Summarizing Information during Disasters" consists of two parts --- 1. Sub-event detection and 2. Summarization of information using different small scale information
Long ranging natural disasters span over quite a few days and information is scattered across different humanitarian classes like infrastructure, missing, shelter, volunteer etc. We have considered three different disasters Nepal earthquake, Typhoon Hagupit, and Pakistan flood in this paper
Dataset directory contains Tweet-IDs and ground truth summaries used in this project.
Codes directory contains codes and a README containing information about how to run those codes.
For details --- please go through the paper!!


############################ Sub-event Detection ###################################################################
How to run: python extract_rank_subevent.py <parsed input file> <entity tagged file> <output file>

Input =>
        1. input parsed file contains dependency relations of tweets [done using Twitter parser (Kong Emnlp 2014)]
        2. file contains Entity and event tagged tweets [done using Twitter NER tagger (Ritter ACL 2011)]
        3. output file : Noun \t Verb \t Co-occurrence count \t Frequency of noun \t Frequency of verb \t Simpson score

Sample run: python extract_run_subevent.py infrastructure_parsed_20150426.txt infrastructure_ner_20150426.txt infrastructure_subevent_20150426.txt

Note:   1. entity file can be skipped from the code if we want to get output fast
        2. This code extracts all the noun-verb pairs. However, we can discard random pairs by putting a threshold on the co-occurrence frequency (10), noun frequency (5), verb frequency (5)

Sample files are in the Codes directory.



############################ Summarization #########################################################################

#############################Step1: Concept extraction #############################################################
How to run: python extract_terms.py <input file> <place file> <output file>
Inputs: 1. input file format => Tweet ID \t Tweet \t Confidence score
        2. Place file contains information about locations
        3. output file contains concept details

Sample run: python extract_terms.py infrastructure_20150426.txt nepal_place.txt infrastructure_concept_20150426.txt

############################# Class Level Summarization ############################################################
How to run: python subevent_summary.py <input concept file> <input parse file> <input event tagged file> <place file> <keyword> <date> <word length>

Sample run: python subevent_summary.py infrastructure_concept_20150426.txt infrastructure_parsed_20150426.txt infrastructure_ner_20150426.txt nepal_place.txt infrastructure 20150426 200

############################# High Level Summary ###################################################################
How to run: python general_subevent_summary.py <keyterm> <place> <date> <word length>

Sample run: python general_subevent_summary.py Nepal nepal_place.txt 20150426 200

Create two separate directories 1. concept_extraction and 2. parsed_event_files. Put all the relevant class specific files from all the datasets into these two directories.

Ground Truth Summaries
-----------------------
We have prepared the ground truth summaries following the instructions of Document Understanding Conferences(DUC). 5 graduate students participated in this task. If you need ground truth summaries kindly sent a mail to rudra@l3s.de and pawang@cse.iitkgp.ac.in.

If you are using the dataset of this paper, kindly cite the following article:

Koustav Rudra, Pawan Goyal, Niloy Ganguly, Prasenjit Mitra, and Muhammad Imran. 2018. Identifying Sub-events and Summarizing Disaster-Related Information from Microblogs. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR '18). ACM, New York, NY, USA, 265-274. DOI: https://doi.org/10.1145/3209978.3210030

@inproceedings{Rudra:2018:ISS:3209978.3210030,
 author = {Rudra, Koustav and Goyal, Pawan and Ganguly, Niloy and Mitra, Prasenjit and Imran, Muhammad},
 title = {Identifying Sub-events and Summarizing Disaster-Related Information from Microblogs},
 booktitle = {The 41st International ACM SIGIR Conference on Research \&\#38; Development in Information Retrieval},
 series = {SIGIR '18},
 year = {2018},
 isbn = {978-1-4503-5657-2},
 location = {Ann Arbor, MI, USA},
 pages = {265--274},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3209978.3210030},
 doi = {10.1145/3209978.3210030},
 acmid = {3210030},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {class-based summarization, high-level summarization, humanitarian classes, situational information, sub-event detection},
}

