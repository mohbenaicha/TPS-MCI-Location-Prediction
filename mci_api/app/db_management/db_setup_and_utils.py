# Given the simplicty of database setup, this module handles
# sessinos, models and writing to the database

import logging
from typing import List, Dict, Union, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
import numpy as np
import os
import pandas as pd
from config import settings
from mci_model.config.base import config, DATASET_DIR

_logger = logging.getLogger(__name__)

engine = create_engine(settings.db_settings.SQLALCH_DB_URI)
# engine = create_engine(settings.db_settings.SQLALCH_DB_URI,)
# allows disparate sections of the application to call upon a global 
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()

# Define model for table creation
class LR_Predictions_Model(Base):
    __tablename__ = "linear_regression_io" # TODO: update to env. var.
    
    id = Column(Integer, primary_key=True) # record id
    user_id = Column(String(36), nullable=False) # for registering client id
    
    # stores a DateTime object; indexed to speed up retrieval when calculating
    # metrics during monitoring since it's don't on the basis of time
    
    datetime_captured = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    ) 
    
    model_version = Column(String(36), nullable=False)
    
    # sqlalchemy defines JSONB as 
    # given we're writing arrays, binary json encodes/decodes faster
    inputs = Column(JSONB) 
    outputs = Column(JSONB)


class WriteToDB():
    
    def __init__(self, db_session: Session, user_id: Optional[str] = None):
        self.user_id = user_id if user_id else int(np.random.randint(0, 1e6, 1)[0])
        self.db_session = db_session
    

    def close(self) -> None:
        if self.db_session:
            self.db_session.remove()
            _logger.debug('Session closed.')
        else:
            _logger.warning('Session was not closed or no session to close.')


    def add_and_commit(self, 
        inputs: List[Dict[str, Union[str, float, int]]],
        model_version: str,
        predictions: List[List[float]],
        db_model: Base,
        close_session:bool=True):

        input_output = LR_Predictions_Model(
            user_id = self.user_id,
            model_version=model_version,
            inputs=inputs,
            outputs=predictions)
        try:
            self.db_session.add(input_output)
            self.db_session.commit()
            _logger.debug(f'Data saved to {db_model}')
        except Exception as e:
            _logger.error(e)
            
        if close_session:
            self.close()


def save_to_db(db_session, data, version, predictions):
    writer = WriteToDB(db_session=db_session)
    writer.add_and_commit(
        inputs=data,
        model_version=version,
        predictions=predictions,
        db_model='linear_regression_io', 
    )


def connect_and_load(db_uri):
    engine = create_engine(db_uri)
    sql_df = pd.read_sql_table("linear_regression_io", con=engine)
    
    # munge json array of inputs from postgres jsonb field.
    inputs_df = sql_df.inputs.apply(
        lambda row: pd.DataFrame(row)).tolist()
    outputs_df = sql_df.outputs.apply(lambda row: pd.DataFrame(row)).to_list()

    inputs_df = pd.concat(inputs_df, sort=False)
    outputs_df = pd.concat(outputs_df)

    outputs_df.rename(columns={0: "Lat", 1: "Long"}, inplace=True)
    train_data_path = DATASET_DIR / f'{config.app_config.test_data_file}'
    live_data = pd.concat((inputs_df, outputs_df), axis=1).reset_index().drop(columns=['index'])
    train_data = pd.read_csv(train_data_path, usecols=live_data.columns).reset_index().drop(columns=['index'])
    
    return live_data[-10000:], train_data
