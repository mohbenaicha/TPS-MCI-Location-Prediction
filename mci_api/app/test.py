import json
import time
from typing import Any
from threading import Thread
import os
import sys
import inspect
import numpy as np
import pandas as pd
import redis
from loguru import logger
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from mci_model import __version__ as MODEL_VERSION
from mci_model.config.base import config # ml package config
from mci_model.predict import make_prediction


from app import __version__
from app import schemas
from app.config import settings # application config
from app.db_management.db_setup_and_utils import *
from app.tasks import db_write
# from app.tasks import infer
from app.monitoring.metrics_utils import setup_metrics_table, calc_metrics
from app.monitoring.prometheus_monitoring import (LAT_TRACKER, 
    LONG_TRACKER, 
    LAT_GAUGE, 
    LONG_GAUGE, 
    PRED_COUNTER, 
    REQUEST_LATENCY,
    APPLICATION_DETAILS,
    send_metrics)



def main():
    r = redis.StrictRedis(os.getenv('REDIS_HOST'), os.getenv('REDIS_PORT'), decode_responses=True)
    window_size = 10
    # window_size = r.get('window_size')
    
    last_rec = int(r.get('last_id'))
    
    _, train_data = connect_and_load(db_uri=settings.db_settings.SQLALCH_DB_URI)
    
    live_data = pd.DataFrame([r.hgetall(rec) for rec in range(1, window_size)])
    
    metrics_table = calc_metrics(
        setup_metrics_table(train_data),
        train_data,
        live_data)

    # return jsonable_encoder(metrics_table.to_dict())
    logger.info(metrics_table)
    return metrics_table


if __name__ == '__main__':
    main()
