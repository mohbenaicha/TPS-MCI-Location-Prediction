
import math
import numpy
from sklearn.model_selection import train_test_split
from mci_model.pipeline import mci_pipeline
from mci_model.predict import make_prediction
from mci_model.config.base import config
from mci_model.utilities.validation import validate_inputs



def test_pipeline_transformation_validation(test_pipeline_data, test_input_data):
	'''A long unit test that tests whether the pipeline transforms and validates 
	features properly'''

	m_config = config.model_config

	# Some data received my be missing such columns and 
	# does contain such columns, to maintain consistency in testing
	# they're added with a value of 0 if they don't exist to ensure
	# they are properly handled by the pipeline

	for feat in m_config.features_to_drop:
		if not feat in test_pipeline_data.columns:
			test_pipeline_data[feat] = 0

	
	# Step 1: Pipeline transformations and feature addition
	
	# Given
	transformed_X = mci_pipeline[:-3].fit_transform(
		test_pipeline_data[m_config.train_features + m_config.inference_features_to_add],
		test_pipeline_data[m_config.targets]) 

	# Then
	assert all([feat not in list(transformed_X.columns) for feat in m_config.features_to_drop])
	assert all([feat in transformed_X.columns for feat in list(m_config.engineered_features.values())[:4]])
	
	# Given
	transformed_columns = list(m_config.datetime_features.values())[1:] + list(m_config.engineered_features.values())[:4]
	
	# Then
	assert transformed_X[transformed_columns].iloc[0, :].to_list() == [61, 3, 6, 2, 8, 'weekend', 'med', 'Winter', 'non-holiday']	
	assert all([feat in transformed_X.columns.to_list() for feat in list(m_config.engineered_features.values())[4:]])
	assert transformed_X[list(m_config.engineered_features.values())[4:]].iloc[0, :].to_list() == [252, 83, 19]

	


	# Step 2: test pipeline validates input features
	mci_pipeline.fit(
	    test_pipeline_data[m_config.train_features + m_config.inference_features_to_add],
	    test_pipeline_data[m_config.targets],	
		)

	# Given
	validated_inputs, errors = validate_inputs(input_data=test_input_data)
	   
	# # remove feature adding step as in inference, that's a given input
	mci_pipeline.steps.pop(5)
	preds = mci_pipeline.predict(test_input_data[m_config.train_features + m_config.inference_features_to_add])

	# Then

	assert errors is None
	assert len(preds) == 30000
    
    