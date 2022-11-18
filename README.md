# Analysis of the Vocabulary of the British State 2000-2020

Code used for my PhD thesis *BERT for Discourse Analysis: A Pragmatist Approach to Governmentality*. Purpose of code is to (a) examine vocabulary variation across the British state's division of labour and (b) analyse the word senses that result from the diffusion of 'resilience', 'sustainable', 'sustainability' and 'wellbeing' into the organisations of the British state.

The method of word sense induction via BERT used here was taken from **Lucy, L., Bamman, D. (2021). ‘Characterizing English Variation across Social Media Communities with BERT’. Transactions of the Association for Computational Linguistics: 538–56.** Many thanks to Lucy Li and David Bamman for making the code available: https://github.com/lucy3/ingroup_lang 


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

## Data
Dataset was constructed from government publications scraped from https://www.gov.uk/search, https://webarchive.nationalarchives.gov.uk/search and https://www.legislation.gov.uk. Approximately 170000 documents were scraped, from which a stratified random sample of 92 million tokens was constructed. 
