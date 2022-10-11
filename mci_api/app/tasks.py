import time
import json
import pandas as pd
import numpy as np
import redis
from celery import Celery
# from mci_model.predict import make_prediction
from db_management.db_setup_and_utils import *
from config import settings

redis_db = redis.StrictRedis(os.getenv('REDIS_HOST'), os.getenv('REDIS_PORT'), decode_responses=True)

app = Celery(
    'tasks', 
    backend=f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}",
    broker=f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
    # backend=settings.db_settings.REDIS_URI, 
    # broker=settings.db_settings.REDIS_URI
    )

# @app.task(name='tasks.infer')
# def infer(input_data):
#     return make_prediction(input_data=pd.DataFrame(input_data).replace({np.nan: None}))



@app.task(name='tasks.db_write')
def db_write(input_data, results, model_version):
    save_to_db(db_session=db_session, # imported from db_setup_and_utils
            data=input_data,
            version=model_version, # provided by invoker 
            predictions=results['predictions'])

    # Logic to prevent simulatneous writes to cache from different workers: 
    # last_id is the control key and, if set to 0, disallows workers from
    # proceeding with writing, otherwise a worker process picks up the last_id
    # sets it to 0 to prevent other worker from writing, and proceeds to write
    # then releases the 'lock' by setting the last_id to the last_id written

    counter = int(redis_db.get('last_id'))
    while counter == 0:
        time.sleep(1)
        counter = int(redis_db.get('last_id'))

    redis_db.set('last_id', 0)

    for rec, pred in zip(input_data, results['predictions']):
        counter += 1
        rec['Lat'], rec['Long'] = pred[0], pred[1]
        redis_db.hmset(counter, rec)
        
    redis_db.set('last_id', counter)
    
    return input_data