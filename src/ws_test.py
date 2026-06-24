"""Local end-to-end test of the offload server: GET the page + JS, then stream a
JPEG over the WebSocket and print the returned sector distances. Retries while the
model loads. Run AFTER starting `python src/server.py` (in another terminal)."""
import asyncio, aiohttp, cv2

URL = "https://localhost:8443"


async def main():
    img = cv2.imread("/Users/levilingsch/Code/AISprint/work/rgb/IMG_0281.png")
    data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])[1].tobytes()
    for _ in range(60):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.get(URL + "/", ssl=False) as r:
                    html = await r.text()
                async with s.get(URL + "/nav-remote.js", ssl=False) as r:
                    js_ct = r.headers.get("content-type")
                async with s.ws_connect(URL + "/ws", ssl=False) as ws:
                    await ws.send_bytes(data)
                    msg = await ws.receive()
                    print("GET / ->", "remote.html OK" if "Echolocation" in html else "?? unexpected")
                    print("GET /nav-remote.js -> content-type:", js_ct)
                    print("WS round-trip reply:", msg.data)
                async with s.post(URL + "/api/claude", data=b"{}", ssl=False) as r:
                    print("POST /api/claude -> status", r.status, "(503 if no key set yet — expected)")
            return
        except Exception:
            await asyncio.sleep(0.5)
    print("FAILED to connect to server")


asyncio.run(main())
