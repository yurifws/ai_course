# Event ingest endpoint: accept a JSON payload, acknowledge with 202 Accepted.
# Mounted at POST /events/ via router.py. Body is validated by EventSchema.

import json
from http import HTTPStatus

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import Response

router = APIRouter()


class EventSchema(BaseModel):
    # Pydantic model = request body schema + automatic 422 on invalid JSON.
    event_id: str
    event_type: str
    event_data: dict


@router.post("/", dependencies=[])
def handle_event(data: EventSchema) -> Response:
    # FastAPI parses JSON into EventSchema before this runs.
    print(f"Received event: {data}")

    # 202 Accepted: request was valid and queued; processing is not done yet.
    # Return raw JSON via Response so we control status_code independently of the body.
    return Response(
        status_code=HTTPStatus.ACCEPTED,
        content=json.dumps({"message": "Data received successfully"}),
    )
