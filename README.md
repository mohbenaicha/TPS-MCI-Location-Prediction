# MCI-Crime-Location-Prediction

This is a Python package and API I built for predicting locations of major crimes using the Toronto Police MCI dataset. The package and API are kept in this single repo to facilitate access to them; otherwise, the package is downloadable through ` pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ MCI-Crime-Location-Prediction` and the API is accessible at: https://mci-app.herokuapp.com/.


## Test out the model
- The API's accessible through its graphical UI: https://mci-app.herokuapp.com/ (the app runs on a single minimal cloud instance and may be sleeping so it may take some time to load)

#### Using the API's graphical UI:
- Open the 'Proceed' link in a new tab and keep the existing tab to follow the rest of the instructions
- Expand the `POST` request section
- Find the `Try it out` button.
- Edit the 'Request body' input box to your liking then hit `Execute`.
- Scroll down further (ignore the cURL request template following it)
- Under the reponse body, under status code 200, check the predictions and a Google Maps link that you can paste onto Google Maps to get a visual on the location (disclaimers at Appendix). 


#### Using the API through cURL:

```
curl -X 'POST' 'https://mci-app.herokuapp.com/api/v1/predict' -H 'accept: application/json' -H 'Content-Type: application/json' \
-d '{"inputs": [{
      "occurrencehour": 11, "Pub_Id": 136, "Park_Id": 43, "PS_Id": 22, "premises_type": "House", "occurrencemonth": "July", "occurrencedayofweek": "Sunday", "MCI": "Assault", "Neighbourhood": "Eglinton East (138)", "occurrenceday": 14, "occurrencedayofyear": 195
    }
  ]
}'
```

## Alternatively, make a prediction manually:
- install the package `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ MCI-Crime-Location-Prediction` (ideally, in a new virtual environment)
- launch python in the environment used to install MCI-Crime-Location-Prediction
- import the package and make a prediction

```
import mci_model

sample = [{"occurrencehour": 11, "Pub_Id": 136, "Park_Id": 43, "PS_Id": 22, "premises_type": "House", "occurrencemonth": "July", "occurrencedayofweek": "Sunday", "MCI": "Assault", "Neighbourhood": "Eglinton East (138)", "occurrenceday": 14, "occurrencedayofyear": 195}]

print(mci_model.predict(input_data=sample))
```

## Or, run the API locally: 
  - create a new folder locally, cd into the folder and `git init` it, then clone the current GitHub repo 
  
      `git clone https://github.com/mohbenaicha/MCI-Crime-Locations`
      
  - cd into the /mci-api directory, make sure you have `tox` installed (`pip install tox`), 
  - run `tox -e run` to host the API
  - copy+paste the url provided by uvicorn into a browser.
  - follow the steps on using the API above 'Using the API's graphical UI'


## Appendix: Notes and Disclaimers:

 - Note: A status code 200 means the post request was valid and should yield a prediction and Google Maps link within the response body.
 - Note: Date-time variables are correlated, naturally, so querying January and occurencedayofyear = 360 would throw the model off not due to poor predictive capability, but merely due to lack of sense. The same can be said for pub/park/PS ids and neighbourhood names since these spatial variables are naturally correlated. In terms of predictions, the model pipeline's takes care of correlated variables to mitigate collinearity.
 - Disclaimer: Any element of the model's input data that may give rise to privacy concerns has been anonymized, but the predictions are the user's responsiblity.
 - Disclaimer: Location predictions do not entail, in any way, actual occurence of crimes therein. Furthermore, **the API USER MUST RESPECT THE PRIVACY CONCERNS of the parties concerned** should predictions fall on or near private property.
