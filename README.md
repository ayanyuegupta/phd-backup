# Analysis of the Vocabulary of the British State 2000-2020

Code used for my PhD thesis *BERT for Discourse Analysis: A Pragmatist Approach to Governmentality*. My thesis explores the utility of deep learning language models as a tool for understanding governmental discourse. I argue that BERT can be used as a tool for understanding the linguistic diffusion through which neoliberal governmental discourses surrounding 'resilience', 'sustainability' and 'wellbeing' spread throughout the organisations of the British state during 2000-2020. 

Purpose of code is to (a) examine vocabulary variation across the British state's division of labour and (b) analyse the word senses that result from the diffusion of 'resilience', 'resilient', 'sustainable', 'sustainability' and 'wellbeing' into the organisations of the British state. Word sense induction via BERT was used for (b).

The method of word sense induction via BERT used here was taken from **Lucy, L., Bamman, D. (2021). ‘Characterizing English Variation across Social Media Communities with BERT’. Transactions of the Association for Computational Linguistics: 538–56.** Much of the code used to perform word sense induction here is taken from Lucy and Bamman's code, with some alterations made to adapt the code to my purposes. Many thanks to Lucy Li and David Bamman for making the code available: https://github.com/lucy3/ingroup_lang 

## Data

Dataset was constructed from government publications scraped from https://www.gov.uk/search, https://webarchive.nationalarchives.gov.uk/search and https://www.legislation.gov.uk. Approximately 170000 documents were scraped, from which a stratified random sample of 92 million tokens was constructed. 

## Some Example Results

<img src="/assets/6.png" width="50%">
The plots on the left of this figure show the relative frequencies over time of 'resilience', 'resilient', 'sustainable', 'sustainability' and 'wellbeing'. The plots on the right show the relative frequencies over time of the senses of 'resilience', 'resilient', 'sustainable', 'sustainability' and 'wellbeing' induced through BERT. 

The general increase in the relative frequencies of these words in the 2000s corroborates research that notes the increased prevalence of discussion about resilience/sustainability/wellbeing over the past 2-3 decades. Authors have argued this prevalence is the result of the diffusion of these notions from academia into state organisations (e.g. the way 'resilience' is used in state organisations has it's roots in C.S. Holling's use of 'resilience' in his analysis of ecosystems). 

Word sense induction via BERT allows one to see what particular senses are responsible for the increased prevalence of the target words. For example, in the case of 'resilience' it is plain that it's increase in relative frequency during 2000 - 2010 is largely the result of the increase in relative frequency of the sense *resilience 3*. This sense concerns uses of 'resilience' concerned with the implementation of a risk/crisis management framework, codified in legislation through the *Civil Contingencies Act 2004*. Thus the most distinctive terms of *resilience 3* (measured using TF-IDF) are bigrams such as 'resilience forums', 'national resilience' and 'category responders', which are all terms which relate to the organisations (e.g. Local Resilience Forums) and statutory requirements (e.g. Category 1 and 2 responders are required by law to assess the risk of emergencies) through which resilience as a conceptual framework for risk management is implemented.    

## Description of files

Get word counts and specificity/volatility scores:
* measures.py

Perform word sense induction on total vocabulary:
* cluster_train.py
* cluster_match.py

Induce additional senses for 'resilience', 'sustainable', 'sustainability' & 'wellbeing' and get/visualise sense scores:
* add_senses_run.py This runs the following:
    * add_centroids.py
    * add_senses.py
    * add_senses_measures.py 
    

Analyse type specificity and type volatility -- run regressions, hypothesis tests, get visualisations etc.:
* analyse_type.py

Analyse sense specificity and sense volatility -- run regressions, hypothesis tests, get visualisations etc.:
* analyse_sense.py
* analyse_sense2.py
* analyse_sense3.py

Get sense specificities and volatilities for induced senses of 'resilience', 'sustainable', 'sustainability' & 'wellbeing':
* measures2.py

Visualise change in relative frequency of 'resilience', 'sustainable', 'sustainability' & 'wellbeing':
* analyse_diffusion.py

Analyse use of 'resilience', 'sustainable', 'sustainability' and 'wellbeing' -- get frequency distributions, compare means, get effect sizes etc.:
* analyse_targets.py
* analyse_targets2.py


Retrieve contents of induced sense clusters, get cluster key terms:
* analyse_sense_clusters.py


