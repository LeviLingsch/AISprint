#!/usr/bin/env python3
"""Laptop offload server for the live demo (Route B).

The phone runs the web UI (camera + audio + Claude taps) and streams JPEG frames
here over a secure WebSocket; we run the metric depth model (fast on MPS), reduce
it to 7 sector distances, and send those back. The phone synthesizes the audio
locally. Only depth is offloaded — that was the slow part on the phone.

Reuses the validated desktop engine: DepthEngine (live.py) + sector_distances (synth.py).

    python src/gencert.py     # once (re-run after tethering)
    python src/server.py       # serves https://<laptop-ip>:8443/
    python src/server.py --check   # smoke test: model + cert + one inference, no serving
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys, ssl, json, re, subprocess, asyncio
import numpy as np
import cv2
import aiohttp
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from live import DepthEngine
from synth import sector_distances

WEB = os.path.join(ROOT, "web")
CERT = os.path.join(ROOT, "certs")
N_SECTORS, ROI_TOP, ROI_BOTTOM, SIZE, PORT, HTTP_PORT = 7, 0.10, 0.80, 322, 8443, 8000
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
engine = None


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    print("[ws] phone connected")
    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            d = sector_distances(engine.infer(rgb), n_sectors=N_SECTORS,
                                 roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM)
            await ws.send_str(json.dumps({"dist": [round(float(x), 3) for x in d]}))
        elif msg.type == web.WSMsgType.ERROR:
            break
    print("[ws] phone disconnected")
    return ws


def file_route(name):
    async def h(request):
        return web.FileResponse(os.path.join(WEB, name))
    return h


async def index(request):
    return web.FileResponse(os.path.join(WEB, "remote.html"))


def load_key():
    """API key for the Claude proxy: env var first, then secret/anthropic_key.txt."""
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if k:
        return k
    p = os.path.join(ROOT, "secret", "anthropic_key.txt")
    return open(p).read().strip() if os.path.exists(p) else None


async def claude_proxy(request):
    """Forward the phone's Claude request with the server-held key; stream SSE back."""
    key = load_key()
    if not key:
        return web.json_response({"error": "no API key on server"}, status=503)
    body = await request.read()
    headers = {"content-type": "application/json", "x-api-key": key,
               "anthropic-version": "2023-06-01"}
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180))
    try:
        up = await session.post(ANTHROPIC_URL, data=body, headers=headers)
        if up.status != 200:
            return web.Response(status=up.status, text=await up.text())
        resp = web.StreamResponse(status=200, headers={
            "content-type": "text/event-stream", "cache-control": "no-cache"})
        await resp.prepare(request)
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
        await resp.write_eof()
        return resp
    finally:
        await session.close()


def local_ipv4s():
    ips = []
    try:
        out = subprocess.check_output(["ifconfig"], text=True)
        for m in re.finditer(r"inet (\d+\.\d+\.\d+\.\d+)", out):
            ip = m.group(1)
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def make_ssl():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(os.path.join(CERT, "cert.pem"), os.path.join(CERT, "key.pem"))
    return ctx


def build_app():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/api/claude", claude_proxy)
    for f in ("nav-remote.js", "synth-worklet.js", "claude-discuss.js"):
        app.router.add_get("/" + f, file_route(f))
    return app


async def serve():
    runner = web.AppRunner(build_app())
    await runner.setup()
    print("\n=== open the app ===")
    await web.TCPSite(runner, "127.0.0.1", HTTP_PORT).start()
    print(f"   http://localhost:{HTTP_PORT}/      (on THIS laptop, uses its webcam — no cert needed)")
    if os.path.exists(os.path.join(CERT, "cert.pem")):
        await web.TCPSite(runner, "0.0.0.0", PORT, ssl_context=make_ssl()).start()
        for ip in local_ipv4s():
            print(f"   https://{ip}:{PORT}/   (on the PHONE — trust certs/cert.pem first)")
    else:
        print("   (no cert yet — run `python src/gencert.py` to enable the phone URL)")
    print("\nClaude key:", "loaded" if load_key() else "NOT SET (Claude taps off — see SETUP_DEMO.md)")
    print("Ctrl-C to stop.\n")
    await asyncio.Event().wait()


def main():
    global engine
    check = "--check" in sys.argv
    have_cert = os.path.exists(os.path.join(CERT, "cert.pem"))
    engine = DepthEngine(size=SIZE)

    if check:
        p = os.path.join(ROOT, "work/rgb/IMG_0279.png")
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        d = sector_distances(engine.infer(rgb), n_sectors=N_SECTORS,
                             roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM)
        print("inference dist:", np.round(d, 2).tolist())
        print("cert:", "OK" if have_cert and make_ssl() else "MISSING (run gencert.py)")
        print("claude key:", "loaded" if load_key() else "not set")
        print("check OK")
        return

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
