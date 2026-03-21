import logging
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.routes import reset_orphaned_jobs
    reset_orphaned_jobs()
    yield


app = FastAPI(title="MathMotion", lifespan=lifespan)
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text()


if __name__ == "__main__":
    from mathmotion.utils.config import get_config
    config = get_config()
    host = config.server.host
    port = config.server.port
    url = f"http://{host}:{port}"
    print(f"\n  MathMotion → {url}\n")
    webbrowser.open(url)
    uvicorn.run("app:app", host=host, port=port, reload=False, log_config=None)
