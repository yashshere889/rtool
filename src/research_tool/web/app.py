"""The ASGI app: ``/api/*`` plus the built single-page app (with history fallback to index.html)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import __version__
from ..core.errors import RToolError
from .api import router
from .runner import ClusterRunner, LocalRunner, Runner
from .security import SecurityMiddleware
from .settings import UISettings

# The app shell may load only its own scripts and styles; papers are shown in same-origin frames.
APP_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "font-src 'self' data:; connect-src 'self'; frame-src 'self'; object-src 'self'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'self'"
)

NOT_BUILT = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Research Tool</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 4rem auto; line-height: 1.5">
<h1>Research Tool</h1><p>The API is running, but the web app has not been built into this copy of rtool.</p>
<p>On a development machine: <code>scripts/build_frontend.sh</code>, or run <code>npm run dev</code> in
<code>frontend/</code> and open the address it prints.</p></body></html>"""


def create_app(settings: UISettings, runner: Runner | None = None) -> FastAPI:
    app = FastAPI(
        title="Research Tool",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.runner = runner or (LocalRunner() if settings.runs_locally else ClusterRunner())
    app.include_router(router, prefix="/api")

    @app.exception_handler(RToolError)
    async def rtool_error(_request: Request, exc: RToolError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    static = Path(settings.static_dir)

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    async def app_shell(path: str) -> FileResponse | HTMLResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "no such API endpoint")
        asset = (static / path).resolve() if path else None
        if asset is not None and asset.is_file() and static.resolve() in asset.parents:
            return FileResponse(
                asset,
                headers={"cache-control": "public, max-age=31536000, immutable"}
                if path.startswith("assets/")
                else {"cache-control": "no-cache"},
            )
        index = static / "index.html"
        if not index.is_file():
            return HTMLResponse(NOT_BUILT, headers={"content-security-policy": APP_CSP})
        return FileResponse(index, headers={"cache-control": "no-cache", "content-security-policy": APP_CSP})

    app.add_middleware(SecurityMiddleware, token=settings.token)
    return app
