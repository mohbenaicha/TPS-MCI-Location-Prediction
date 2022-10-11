import json
import time
from typing import Any
from threading import Thread
import os
import sys
import inspect
import multiprocessing as mp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from fastapi.encoders import jsonable_encoder
from loguru import logger
import asyncio

from mci_model import __version__ as MODEL_VERSION
from mci_model.config.base import config # ml package config
from mci_model.predict import make_prediction

from app import __version__
from app import schemas
from app.config import settings # application config
from app.db_management.db_setup_and_utils import *
# from app.tasks import infer
from app.tasks import db_write

from app.monitoring.metrics_utils import setup_metrics_table, calc_metrics
from app.monitoring.prometheus_monitoring import (LAT_TRACKER, 
    LONG_TRACKER, 
    LAT_GAUGE, 
    LONG_GAUGE, 
    PRED_COUNTER, 
    REQUEST_LATENCY,
    APPLICATION_DETAILS,
    send_metrics)


# Setup router for endpoints
api_router = (
    APIRouter()
)


# Setup endpoints
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Health end point return model and api version
    """
    health = schemas.Health(
        name=settings.APP_NAME, api_version=__version__, model_version=MODEL_VERSION
    )

    return health.dict()


@api_router.post("/predict", response_model= schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleMCIDataInputs) -> Any:
    """
    Post request as test data and receive repsonse as predictions
    """
    
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    write_invoke = db_write.delay(jsonable_encoder(input_data.inputs), results=results, model_version=MODEL_VERSION) # check if results needs json encoding

    logger.info(f"Write to DB status: {write_invoke.status}")

    

    # # concatenate processed predictions
    # for result in results:
    #     schemas.result_dict['errors'].append(result['errors'] if result['errors'] else None)
    #     schemas.result_dict['predictions'].extend(result['predictions'])
    #     schemas.result_dict['version'] = result['version']

    # if any([error is not None for error in schemas.result_dict["errors"]]):
    #     raise HTTPException(status_code=400, detail=schemas.result_dict["errors"])
    
    # return schemas.result_dict
        
    # Sending Prometheus metrics {host}:8002 # TODO: incorporate into worker tasks
    # prom_metrics_thread = Thread(
    #     target=send_metrics,
    #     args=(APP_NAME, MODEL_NAME, MODEL_VERSION, results, start_t)
    # )
    # prom_metrics_thread.start()
    
    return results



import redis
r = redis.StrictRedis(os.getenv('REDIS_HOST'), os.getenv('REDIS_PORT'), decode_responses=True)

@api_router.get("/metrics", status_code=200)
def get_metrics():
    
    window_size = 10000
    # window_size = r.get('window_size')
    last_rec = int(r.get('last_id'))
    
    _, train_data = connect_and_load(db_uri=settings.db_settings.SQLALCH_DB_URI)
    live_data = pd.DataFrame([r.hget(rec) for rec in range(1, window_size)])

    metrics_table = setup_metrics_table(train_data)
    args = [(metrics_table[[col]], train_data[[col]], live_data[[col]], col) for col in live_data.columns[1:]]
    
    with mp.Pool(mp.cpu_count()) as pool:
        metrics_table = pd.concat(pool.starmap(calc_metrics, args), 1)

    metrics_table.reset_index(level=0, inplace=True)
    metrics_table.rename(columns={'index': 'Metric Name'}, inplace=True)
    
    return jsonable_encoder(metrics_table.to_dict())