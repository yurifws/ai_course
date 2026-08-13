# FastAPI entry point: create the app and mount the process router.
# Run with: uvicorn main:app --reload  (from the app/ directory)

from fastapi import FastAPI
from router import router as process_router

# Empty FastAPI instance — routes come from included routers, not here.
app = FastAPI()
# Attach the top-level router (which itself nests /events).
app.include_router(process_router)
