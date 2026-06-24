# Live demo — phone UI + laptop compute (Route B, over USB)

The **phone** runs the web UI (camera, spatial audio, Claude taps). The **laptop**
runs the depth model and returns 7 sector distances over a USB link. Only depth is
offloaded — that was the slow part on the phone. Audio + Claude stay on the phone.

## One-time on the laptop
```bash
pip install -r requirements.txt   # aiohttp + cryptography (already present on this Mac)
```

## Quick test on the laptop (no cert, no phone)
`python src/server.py`, then open **http://localhost:8000/** on the laptop. localhost is a
secure context, so the camera works with no certificate — handy for verifying the whole
pipeline (depth + audio + Claude) before dealing with the phone, and as a laptop-webcam
fallback demo. The phone still uses the `https://…:8443/` URL (it can't reach "localhost").

## API key (for the Claude taps) — lives on the laptop, not the phone
Pick one:
- **File:** put your key in `secret/anthropic_key.txt` (one line, no quotes), **or**
- **Env var:** `export ANTHROPIC_API_KEY=sk-ant-...` in the shell you start the server from.

The server prints `Claude key: loaded` at startup when it finds it. The phone never sees
the key — its Claude requests are forwarded by the laptop with the key added.

## Each demo
1. **Connect the phone to the laptop by USB.**
2. **Make a local link** (avoids campus WiFi blocking device-to-device):
   - **iPhone:** Settings → Personal Hotspot → ON. Plug in USB. On the Mac the iPhone
     appears as a network service and the Mac gets a `172.20.10.x` address.
   - **Pixel:** Settings → Network → Hotspot & tethering → **USB tethering** ON.
3. **Match the cert to the new IP:** `python src/gencert.py` — note the printed `https://…` URL.
4. **Trust the cert on the phone** (once per cert):
   - **iPhone:** AirDrop `certs/cert.pem` to the phone → Settings → "Profile Downloaded" →
     Install. Then Settings → General → About → **Certificate Trust Settings** → turn ON
     full trust for `Levis-MacBook-Pro.local`.
   - **Pixel:** copy `certs/cert.pem` to the phone → Settings → Security → Encryption &
     credentials → Install a certificate → **CA certificate**.
5. **Start the server:** `python src/server.py`
6. On the phone open the printed URL (e.g. `https://172.20.10.2:8443/`). Tap **Start**,
   allow **Camera + Mic**, put on **headphones**.

## Using it
- **Continuous / Pulse / Sweep** buttons, **range − / +**.
- **Double-tap** = Claude describes the scene · **single-tap** = ask a follow-up ·
  **long-press or say "stop"** = stop. (Key is on the laptop — see *API key* above.)

## If something breaks
- **Camera blocked / "not secure":** the cert isn't trusted yet (step 4), or you opened
  `http://` instead of `https://`.
- **WebSocket error:** server not running, or the URL's IP isn't in the cert — re-run
  `gencert.py` after tethering, then reconnect.
- **Claude can't reach the API:** set the key on the laptop (`secret/anthropic_key.txt` or
  `ANTHROPIC_API_KEY`); the phone still needs internet (cellular via Personal Hotspot is fine).
- **Still slow:** confirm you're on USB, not WiFi. Depth runs ~30 fps on the Mac.

## Fallback (if iPhone cert trust fights you mid-demo)
Use the phone as a plain webcam into the desktop app instead: **Camo** (iPhone) or
**DroidCam/Iriun** (Pixel) over USB, then `python src/live.py --camera <N>` with
headphones on the laptop. Loses the phone UI but is bulletproof.
