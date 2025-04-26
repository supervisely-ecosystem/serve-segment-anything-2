import asyncio
import json
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from supervisely.api.api import ApiField

class M:
    def __init__(self):
        self.app = None

app = FastAPI()
m = M()
m.app = app

@app.post("/track_stream")
async def track_stream(request: Request):
    return await _setup_stream(request)


@app.post("/pop_tracking_results")
def pop_tracking_results(request: Request):
    return {"pending_results": []}


async def _setup_stream(request: Request):
    inference_request_uuid = uuid.uuid5(
        namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
    ).hex

    context = None
    connection_is_active = True
    async def read_stream():
        nonlocal connection_is_active
        nonlocal context
        print("reading stream")
        while connection_is_active:
            data = None
            try:
                message = await request.receive()
                print("Received message:", message)
            except Exception as e:
                print("Error receiving message:", e)
            else:
                if message["type"] == "http.disconnect":
                    print("Client disconnected")
                    connection_is_active = False
                    break
                data = message.get("body", b"")
            if data:
                print("Received data:", data)
            else:
                print("No data received")
            try:
                data = json.loads(data)
                context = data["context"]
            except Exception:
                pass
            await asyncio.sleep(1)
    asyncio.create_task(read_stream())

    while context is None:
        await asyncio.sleep(1)
    print("Context received:", context)

    track_id = context["trackId"]
    video_id = context["videoId"]
    session_id = context["sessionId"]
    frame_index = context["frameIndex"]
    frames_count = context["frames"]
    frame_range = (frame_index, frame_index+frames_count)
    total = 100

    async def event_generator():
        nonlocal connection_is_active
        i = 0
        while connection_is_active:
            await asyncio.sleep(1)
            if i == 0:
                data = {"trackId": track_id, "action": "inference-started", "payload": {"inference_request_uuid": inference_request_uuid}}
            else:
                payload = {
                    ApiField.TRACK_ID: track_id,
                    ApiField.VIDEO_ID: video_id,
                    ApiField.FRAME_RANGE: frame_range,
                    ApiField.PROGRESS: {
                        ApiField.CURRENT: i,
                        ApiField.TOTAL: total,
                    },
                }
                data = {
                    ApiField.SESSION_ID: session_id,
                    ApiField.ACTION: "progress",
                    ApiField.PAYLOAD: payload,
                }
            print("Sending data:", i)
            yield f"data: {json.dumps(data)}\n\n"
            i += 1
            if i > 100:
                print("Stopping stream, i > 100")
                connection_is_active = False
                break


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
