from app import __version__, schemas
from prometheus_client import Histogram, Summary, Gauge, Counter, Info
import time


LAT_TRACKER = Histogram(
    name='lat_prediction',
    documentation='mci model latitude predictions',
    labelnames=['app_name', 'model_name', 'model_version']
)

LONG_TRACKER = Histogram(
    name='long_prediction',
    documentation='mci model longitude predictions',
    labelnames=['app_name', 'model_name', 'model_version']
)


LAT_GAUGE = Gauge(
    name='lat_prediction_gauge',
    documentation='mci model latitude predictions gauge',
    labelnames=['app_name', 'model_name', 'model_version']
)

LONG_GAUGE = Gauge(
    name='long_prediction_gauge',
    documentation='mci model longitude predictions gauge',
    labelnames=['app_name', 'model_name', 'model_version']
)

PRED_COUNTER = Counter(
    name='prediction_counter',
    documentation='App average latency per request in seconds',
    labelnames=['app_name', 'model_name', 'model_version']
)

REQUEST_LATENCY = Summary(
    name='per_prediction_latency',
    documentation='App average latency per request in seconds',
    labelnames=['app_name', 'model_name', 'model_version']
)


APPLICATION_DETAILS = Info(
    'application_details',
    'Capture model version information',
)


# define functions for multi-threading
def send_metrics(APP_NAME, MODEL_NAME, MODEL_VERSION, results, start):
    for prediction in results["predictions"]:
        LAT_TRACKER.labels(APP_NAME,MODEL_NAME, MODEL_VERSION).observe(abs(prediction[0]))
        LONG_TRACKER.labels(APP_NAME,MODEL_NAME, MODEL_VERSION).observe(abs(prediction[1]))

        LAT_GAUGE.labels(APP_NAME,MODEL_NAME, MODEL_VERSION).set(abs(prediction[0]))
        LONG_GAUGE.labels(APP_NAME,MODEL_NAME, MODEL_VERSION).set(abs(prediction[1]))
        
        PRED_COUNTER.labels(APP_NAME,MODEL_NAME, MODEL_VERSION).inc()

    avg_pred_latency = (time.time()-start)/len(results)
    REQUEST_LATENCY.labels(APP_NAME,MODEL_NAME, MODEL_VERSION).observe(avg_pred_latency)