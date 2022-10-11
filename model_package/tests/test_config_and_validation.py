import os
import json
from pathlib import Path
import logging
import pytest
from pydantic import ValidationError

from mci_model.config.base import (config, 
	find_config_file, 
	get_config, 
	create_config)
from mci_model.pipeline import mci_pipeline
from mci_model.utilities.validation import validate_inputs


def test_config_validation(config_dir):
	parsed_config=get_config(cfg_path=find_config_file())
	cfg_obj = create_config(
		parsed_config=parsed_config
	)

	assert cfg_obj.model_config
	assert cfg_obj.app_config

	invalid_cfg_path = 'tests/invalid_config.yml' 
	parsed_invalid_config = get_config(invalid_cfg_path)
	
	with pytest.raises(ValidationError) as e:
		create_config(parsed_config=parsed_invalid_config)
	
	assert all([string in str(e) for string in 
		['en', 'test_size', 'not allowed', 'field required']])

def test_validator(test_input_data):
	   
    # Given
    assert len(test_input_data) == 30000
    validated_inputs, errors = validate_inputs(input_data=test_input_data)
    
    # Then
    assert errors is None
    assert len(validated_inputs) == 30000