# Hosting the AutoX docs on the tailnet (agent instructions)

> This file is **not** part of the rendered Sphinx docs. It's a standalone
> runbook: hand it to an agent (or follow it yourself) to build the docs and serve
> them from this machine so teammates on the same Tailscale tailnet can read them.

## Goal

Build the Sphinx site and serve `docs/build/html/` over HTTP, reachable by other
machines on the same tailnet (private VPN) — no public exposure, no port
forwarding.

## Assumptions

- Run everything from the repo root (`AutoX/`).
- The Sphinx toolchain is in the project's dev dependency group; the project's
  virtualenv (`.venv/`) already has `sphinx-build`. If `make`/`sphinx-build`
  aren't found, install the dev deps (`uv sync` / `pip install -e '.[dev]'`) or
  call the venv binary directly (`.venv/bin/sphinx-build`).
- Tailscale is installed and this machine is logged into the team's tailnet. If
  not: `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`.

## Steps

### 1. Build the HTML

```sh
cd docs
make html
# or, equivalently, from the repo root:
# .venv/bin/sphinx-build -b html docs/source docs/build/html
```

Output lands in `docs/build/html/`. A handful of `autodoc` "failed to import"
warnings for hardware-heavy modules are expected and harmless — the build still
succeeds. `docs/build/` is gitignored; never commit it, just rebuild.

### 2. Find this machine's tailnet identity

```sh
tailscale ip -4     # e.g. 100.101.102.103  (the 100.x.y.z address to share)
tailscale status    # shows the MagicDNS name, e.g. jetson-orin
```

If MagicDNS is enabled on the tailnet, teammates can use the name instead of the
raw IP.

### 3. Serve the static site

Python's built-in static server is enough (Sphinx output is fully static):

```sh
python3 -m http.server 8000 --directory docs/build/html
```

`http.server` binds to all interfaces by default, which already includes the
Tailscale interface. To expose it **only** on the tailnet (not the local LAN /
Wi-Fi), bind to the tailnet IP from step 2:

```sh
python3 -m http.server 8000 --bind 100.101.102.103 --directory docs/build/html
```

### 4. Share the URL

Teammates on the same tailnet open:

```
http://100.101.102.103:8000/     # by Tailscale IP
http://jetson-orin:8000/         # by MagicDNS name, if enabled
```

Done — traffic rides the encrypted WireGuard tunnel, so no firewall changes and
nothing is exposed publicly.

## Optional niceties

- **HTTPS with a real name, no cert wrangling** (requires HTTPS enabled in the
  tailnet admin console):

  ```sh
  tailscale serve 8000     # proxies local :8000 over HTTPS on the tailnet
  # teammates: https://jetson-orin.<your-tailnet>.ts.net/
  tailscale serve status   # inspect
  tailscale serve --https=443 off   # stop
  ```

- **Keep it running after logout:** run the `http.server` line inside `tmux` /
  `screen`, or as a small `systemd` user service, so it survives SSH disconnects.

- **Live-reload while editing docs:** `sphinx-autobuild docs/source
  docs/build/html --host 0.0.0.0 --port 8000` rebuilds on save (install with
  `uv add --dev sphinx-autobuild` / `pip install sphinx-autobuild`).

## Do NOT

- Do **not** use `tailscale funnel` — that publishes to the **public** internet.
  These docs contain internal code structure and notes; keep them tailnet-only.
- Do **not** commit `docs/build/`. Rebuild wherever you serve it.
- Remember the served HTML is a **static snapshot** from the last `make html`.
  After editing any `.md`/`.rst` or a docstring, rebuild for teammates to see it.
