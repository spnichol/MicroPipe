
# coding: utf-8

# In[1]:

from pyrattle.eq.settings import S3_ACCESS_KEY, S3_SECRET_KEY, AZ_ACCOUNT_NAME, AZ_ACCOUNT_KEY
from PrattleNER.db_utils.models import * 
from PrattleNER.db_utils import * 
from pyrattle.eq.postgres.util import PrattleConnection, FactsetConnection
from pyrattle.eq.postgres.pr_utils.pr_utils import * 
from pyrattle.eq.postgres.pr_utils.dedupe import * 
import concurrent.futures 
from PrattleNER.matching import StringMatcher, create_vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from fuzzywuzzy import fuzz
import ast 
import nltk.data
import pandas as pd 
from PrattleNER.db_utils.models import * 
from PrattleNER.db_utils import * 
from PrattleNER.StrUtils import * 
from PrattleNER.NERModel import * 
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta


# In[2]:

NER_USER = 'bill'
NER_PASS = 'xBj3T6ugLQbJ'
NER_HOST = 'db.dev.tv.prattle.co'
NER_DB = 'bill'
ner_session = create_session(NER_USER, NER_PASS, NER_HOST, NER_DB)
ner_session = ner_session()
ner_conn = PrattleConnection(username=NER_USER, password=NER_PASS, host=NER_HOST, db=NER_DB)


# In[3]:

engine = create_engine('postgresql://{0}:{1}@{2}/{3}'.format(PG_USER, PG_PASS, PG_HOST, PG_DB), pool_size=60)

def create_session(PG_USER, PG_PASS, PG_HOST, PG_DB, engine):
    Session = sessionmaker(bind=engine)
    return Session


PG_session = create_session(PG_USER, PG_PASS, PG_HOST, PG_DB, engine)
PG_session = PG_session()
PG_conn = PrattleConnection(username=PG_USER, password=PG_PASS, host=PG_HOST, db=PG_DB)


# In[4]:

timespan = datetime.utcnow()-timedelta(hours=2)


# In[5]:

# for_proc = pconn.raw_sql("""select pr.id, pr.title, prd.text, pr.harvest_datetime_utc, pr.source
# from press_release pr
# left join press_release_data prd on prd.pr_id = pr.id 
# left join pr_scoreable scr on scr.pr_id = pr.id 
# left join press_release_nn_preds nnp on nnp.pr_id = pr.id 
# where  prd.text notnull and pr.title notnull
# and pr.retrieval_time >= '{}' and pr.id not in (select doc_id from ner_processed)""".format(timespan))


for_proc = pconn.raw_sql("""select pr.id, pr.title, prd.text, pr.harvest_datetime_utc, pr.source
from press_release pr
left join press_release_data prd on prd.pr_id = pr.id 
left join pr_scoreable scr on scr.pr_id = pr.id 
left join press_release_nn_preds nnp on nnp.pr_id = pr.id 
where  prd.text notnull and pr.title notnull
and pr.retrieval_time >= '{}'""".format(timespan))


# In[6]:

len(for_proc)


# In[7]:

mod_name = 'ner_v1_0.0'
ner_1 = NERModel(model_name=mod_name, gpu=False, num_cores=60)
ner_1.load_ner_model()

mod_name = 'ner_alt_v0.0.0'
ner_2 = NERModel(model_name=mod_name, gpu=False, num_cores=10)
ner_2.load_ner_model()


# In[8]:

preds_1 = ner_1.fast_data_prep(for_proc)
preds_2 = ner_2.fast_data_prep(for_proc)

preds_1 = ner_to_df(preds_1)
preds_2 = ner_to_df(preds_2)

all_preds = pd.concat([preds_1, preds_2]).reset_index(drop=True)


# In[9]:

CLN = CleanNEROutput(verbose=True, load_files=True)
weights, black_list, all_ents = CLN.weights, CLN.black_list, CLN.all_ents


# In[10]:

def mark_done(ids):
    ids = list(set([int(x.split('_')[0]) for x in ids]))
    for this_id in ids:
        fi = NerProcessed(doc_id=this_id, doc_type='pr', processed=datetime.utcnow())
        PG_session.add(fi)
    PG_session.commit()

def process_entity(items):
    try:
        df = pd.DataFrame(list(items), columns=['doc_id', 'extract', 'inf_res_id'])
        CLN = CleanNEROutput(verbose=False, load_files=False, weights=weights, black_list=black_list, all_ents=all_ents)
        CLN.process(ner_res=df[['doc_id', 'extract', 'inf_res_id']])
        CLN.cleanify()
        CLN.create_char_tokenizer()
        matched = CLN.matchify()
        matched['toke_match'] = matched.apply(match_terms, axis=1)
        matched.loc[matched.ent_type.str.lower().str.contains('org'), 'ent_type'] = 'org'
        matched.loc[matched.ent_type.str.lower().str.contains('pers'), 'ent_type'] = 'pers'
        matched = matched[['match_ratio', 'clean_extract', 'ent_name', 'rat', 'doc_id', 'match_id', 'ent_type', 'inf_res_id', 'ent_id', 'toke_match', 'extract']]
        matched['final_match'] = False
        matched['match'] = False 
        matched.loc[(matched.ent_type.str.contains('pers')) & (matched.rat >= 95), 'match'] = True 
        matched.loc[(matched.ent_type.str.contains('org')) & (matched.rat >= 91), 'match'] = True
        matched.loc[(matched.match == True) & (matched.toke_match > 0.5), 'final_match'] = True 
        matched = matched.sort_values('rat', ascending=False).drop_duplicates(subset=['match_id'])
        matched['rat'] = matched['rat'].astype(float)
    except Exception as e:
        return None 

    for i, r in matched.iterrows():
        try:

            fi = InfRes(doc_id=int(r.doc_id.split('_')[0]), 
                        doc_type='pr', 
                        ent_id = r.ent_id, 
                        extract=r.extract, 
                        clean_extract=r.clean_extract, 
                        added=datetime.utcnow(), 
                        mapped=r.final_match,                                                                  
                        token_ratio = r.toke_match, 
                        char_ratio = r.match_ratio, 
                        string_ratio = r.rat)

            ner_session.add(fi)
            ner_session.commit()

        except Exception as e:
            ner_session.rollback()

    mark_done(matched.doc_id.tolist())
    return True 


# In[11]:

prd = list(zip(all_preds.doc_id, all_preds.extract, all_preds.ent_type))


# In[ ]:

#create chunks of 100 ids for each core to process 
chunk_size = 100
these_chunks = chunks(prd, chunk_size)
all_docs = list() 
for chunk in these_chunks:
    all_docs.append(chunk)


# In[ ]:

import warnings
warnings.filterwarnings("ignore")
print("Firing up workers.")
errs = 0 
num_workers = 60 #adjust workers here 
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    print("Pulling res.\n")
    ft = FunTracker(timer_type='long')
    futures = [executor.submit(process_entity, x) for x in all_docs]
    chunk_stat = 0 
    errs = 0 
    for future in concurrent.futures.as_completed(futures):
        this_res = future.result()
        if not this_res:
            errs +=1 
        chunk_stat += chunk_size 
        ft.status(chunk_stat, len(all_docs)*chunk_size)
print("A total number of {} errors.".format(errs))


# In[ ]:

len(prd)


# In[ ]:



