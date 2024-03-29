from typing import Any, Callable, Coroutine
from fastapi import APIRouter, FastAPI, Request, Response
from typing import Any, Coroutine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
import sys

from api import api_router
from config import settings, setup_app_logging

setup_app_logging(config=settings)


 

app = FastAPI(
    title=settings.APP_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Home end point"""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API (Staging)</h1>"
        "<div>"
        # api/v1/doc is the default fast api document url
        "You've reached the MCI Prediction API <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)  # return a webpage response


# route to health and predict end points
app.include_router(api_router, prefix=settings.API_V1_STR)
# route to home end point
app.include_router(root_router)


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        # admits various origins (protocols/ports/etc.) if front end and
        # backend attempt to communicate
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        # allows any request method (post/get)
        allow_methods=["*"],
        # allows request with any header like cookies/User-Agent; can disable
        # not browser user-agents for example
        allow_headers=["*"],
    )


if __name__ == "__main__":
    
    # Use this for debugging purposes only
    # logger.warning("Running in development mode.")
    
    import uvicorn
    from prometheus_client import start_http_server
    
    # log level does not follow production api log level since .run is
    # for debugging in development; production api captures INFO level
    # uvicorn logs using Loguru
    
    # start_http_server(settings.PROM_METRICS_PORT)
    uvicorn.run("main:app", host="0.0.0.0", port=settings.APP_PORT, log_level="info", reload=True)
