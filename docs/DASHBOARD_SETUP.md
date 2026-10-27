# Running the dashboard for a phone test

The app's built-in backend URL is `https://meshsos.raj-ojha.in`, and that
host is currently down (Cloudflare 522, the origin is not responding).
Until it is back, run the backend on your own laptop.

## Start it

Double-click `run_backend.bat`, or in Command Prompt from the repo folder:

```
python -m pip install fastapi "uvicorn[standard]" pydantic
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**`--host 0.0.0.0` is the part that matters.** Without it the server
binds to localhost only: the laptop can open it and the phones cannot.

## Point the phones at it

1. Find the laptop's address: `ipconfig`, look for **IPv4 Address**.
   It looks like `192.168.1.7`.
2. Both phones must be on the **same WiFi** as the laptop.
3. In the app, press **CONFIG** and enter `http://192.168.1.7:8000`
   (your own address, with `http://` and `:8000`).
4. Windows will ask whether to allow Python through the firewall the
   first time. **Allow it**, and tick Private networks.

Do not enter `localhost` or `10.0.2.2`. The app deliberately ignores
both and falls back to the hosted URL, because on a phone `localhost`
means the phone itself.

## Check it works before testing the mesh

On the laptop, open `http://localhost:8000` — the dashboard should load.

Then from the phone's browser, open `http://192.168.1.7:8000`. If the
dashboard loads there too, the phones can reach the server and the only
thing left to test is the mesh itself. If it does not load, the problem
is the network or the firewall, not your app — fix it here rather than
debugging the mesh.

## The full chain, and where to look at each step

| Step | What happens | Where you see it |
|---|---|---|
| 1 | Phone A compresses the text | status line under the box: `25 B, 1 packet` |
| 2 | Phone A broadcasts over BLE | status line: `Sent 1 ... on air 25 B` |
| 3 | Phone B hears it | notification: `SOS heard from ...` |
| 4 | Phone B uploads it | needs WiFi with internet capability |
| 5 | Dashboard shows it | a marker on the map, and `/api/stats` |

If step 5 is empty, check it directly rather than guessing:

```
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/events
```

`/api/events` lists every message the server received, newest last. If a
message is there but no marker appears, the problem is the map, not the
mesh.

## Known: both phones need internet for step 4

`checkUplinkAvailable()` only uploads when the phone's active network
reports internet capability. On WiFi with working internet that is true.
If you test with both phones in aeroplane mode — which is the right way
to prove the mesh works without towers — **nothing will reach the
dashboard**, by design. Do that test for the mesh, then turn WiFi back
on for the dashboard test.
