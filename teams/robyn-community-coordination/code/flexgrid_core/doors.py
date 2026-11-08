"""The control doors, and what a person said that must outlive everything else.

  token    every POST on the fleet, the coordinator, the buyers, the data centers and the Jetson daemon asks for secrets/control_token in the X-FlexGrid-Token header, once that file
           is there. With no file nothing is asked: offline runs, the tests and Town's harness have none, and the deploy turns it on. /health says which ("doors": "token" or "open").
           The token says "one of this system's own tools". It isn't a person: it frees no alarm and raises no limit. Those take a person's signed word, POST /person/<node>.
  book     one small file per node, state/person/<node>.json: the alarm holding it, the limits a person signed, the numbers their messages have used. It's written before the door answers, and it
           isn't tied to a run or an age, so a file set aside by keep.py, a new Town run or a restart can't take an alarm with it. It sits outside state/live for that reason.
  floor    beside it, <node>.floor: when that file was last lost. Its numbers went with it, so from then on a word's number must be later than this. A word's number is when it was signed
           Live only, like keep.py: fleet.py is the only caller, and no offline run has a fleet.

Only a person's signed word ends an alarm, so a deployed node needs a person's public key in its agent.json, or an alarm there stays for good. What a deploy runs, each in its own config
folder ($FLEXGRID_CONFIG_DIR):

  python -m flexgrid_core.doors token                              makes secrets/control_token, if there's none. The same file goes to every machine
  python -m flexgrid_core.doors key owner                          makes secrets/person_owner.key, if there's none, and prints the public half. The private half stays where it was made
  python -m flexgrid_core.doors trust owner <public key>           on a node's machine: person_keys in its agent.json. It reads this at its next start
  python -m flexgrid_core.doors word hh-001 clear_alarm --url URL  a person's signed word, sent to the fleet or the buildings. On a building the role is facility_manager
  python -m flexgrid_core.doors word hh-001 set_limit --group hvac_zones --name max_rise_c --value 1.5 --url URL

None of these prints a token or a private key.
"""
import argparse
import hmac
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("flexgrid-doors")
HEADER, TOKEN_FILE = "X-FlexGrid-Token", "control_token"
READS = ("GET", "HEAD", "OPTIONS")       # what changes nothing: nothing is asked of these
_write = threading.Lock()                # one read-and-write at a time. Each is a person's word or an alarm, so nothing waits long


def token() -> str | None:
    """The control token, or None when this machine has none. A file that's there but empty or unreadable shuts every door: someone meant to turn them on."""
    try:
        return (config.CONFIG_DIR / "secrets" / TOKEN_FILE).read_text().strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        log.error("the control token can't be read, so every door stays shut: %s", exc)
        return ""


def headers() -> dict[str, str]:
    """What a call to a control door sends, for this system's own services and nothing else: Town and the model never see this."""
    t = token()
    return {HEADER: t} if t else {}


def state() -> str:
    return "open" if token() is None else "token"


def door() -> Any:
    """One line per service, FastAPI(dependencies=[doors.door()]), so a route added later is covered too. It's read on each POST, and a POST is rare."""
    from fastapi import Depends, HTTPException, Request

    def asked(request: Request) -> None:
        want = None if request.method in READS else token()
        if want is None:
            return
        if not want or not hmac.compare_digest(request.headers.get(HEADER, "").encode("utf-8", "replace"), want.encode()):
            raise HTTPException(401, f"This door needs the control token, in the {HEADER} header.")
    return Depends(asked)


# What a page this service serves sends with its POSTs (coordinator.page: the Control Room and the phone). It asks a person for the token once, on the first POST a door refuses, keeps it in
# that browser tab alone (sessionStorage: gone when the tab closes), and sends it in the header on a POST to this same server and nowhere else. The token is never in the page, a URL or
# a log: the page has only these lines, and the person types it. A wrong one is forgotten at the next refusal. GETs are left as they are, and so is a request to any other server.
# The page's own fetch sites are in viz/live/src (fg-data.js, fg-monitor.js, fg-agents.js), outside this repository and not carried by a deploy, so the one place every fetch
# passes through, window.fetch, takes the header there.
PAGE_DOOR = """<script>/* the control doors: flexgrid_core/doors.py */(function () {
  var HEADER = 'X-FlexGrid-Token', KEY = 'fg.control-token', real = window.fetch && window.fetch.bind(window), held = '', asking = null;
  if (!real) return;
  function get() { try { return window.sessionStorage.getItem(KEY) || ''; } catch (e) { return held; } }
  function put(v) { held = v; try { if (v) window.sessionStorage.setItem(KEY, v); else window.sessionStorage.removeItem(KEY); } catch (e) { /* a private window: this page's memory */ } }
  function here(u) { try { return new URL(u, window.location.href).origin === window.location.origin; } catch (e) { return false; } }
  function ask() {
    if (asking) return asking;
    asking = new Promise(function (done) {
      var back = document.createElement('div'), box = document.createElement('form'), say = document.createElement('p'), field = document.createElement('input'),
          go = document.createElement('button'), no = document.createElement('button');
      back.setAttribute('style', 'position:fixed;inset:0;z-index:2147483647;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;font:15px system-ui,sans-serif');
      box.setAttribute('style', 'background:#14171c;color:#e8eaed;border:1px solid #3a3f47;border-radius:10px;padding:18px 20px;max-width:420px;width:calc(100% - 32px)');
      say.textContent = 'This button changes the live system, so it needs the control token. Paste it once: it stays in this tab only, and goes only to this server.';
      say.setAttribute('style', 'margin:0 0 12px;line-height:1.4');
      field.type = 'password'; field.autocomplete = 'off'; field.setAttribute('aria-label', 'Control token');
      field.setAttribute('style', 'width:100%;box-sizing:border-box;padding:8px;border-radius:6px;border:1px solid #3a3f47;background:#0c0e11;color:inherit;margin-bottom:12px');
      go.type = 'submit'; go.textContent = 'Send'; no.type = 'button'; no.textContent = 'Cancel';
      go.setAttribute('style', 'padding:6px 14px;margin-right:8px'); no.setAttribute('style', 'padding:6px 14px');
      function end(v) { back.remove(); asking = null; done(v); }
      box.addEventListener('submit', function (e) { e.preventDefault(); end(field.value.trim()); });
      no.addEventListener('click', function () { end(''); });
      back.addEventListener('keydown', function (e) { if (e.key === 'Escape') end(''); });
      box.append(say, field, go, no); back.append(box); document.body.append(back); field.focus();
    });
    return asking;
  }
  window.fetch = function (input, init) {
    var url = typeof input === 'string' || input instanceof URL ? String(input) : null, how = String((init && init.method) || 'GET').toUpperCase();
    if (url === null || how === 'GET' || how === 'HEAD' || how === 'OPTIONS' || !here(url)) return real(input, init);
    function send(tok, again) {
      var h = new Headers((init && init.headers) || {}), more = {headers: h};
      if (tok) h.set(HEADER, tok);
      if (again) more.signal = undefined;        /* asked a person: their time doesn't count against the page's own timeout */
      return real(input, Object.assign({}, init || {}, more));
    }
    return send(get()).then(function (r) {
      if (r.status !== 401) return r;
      put('');
      return ask().then(function (tok) {
        if (!tok) return r;
        put(tok);
        return send(tok, true).then(function (r2) { if (r2.status === 401) put(''); return r2; });
      });
    });
  };
})();</script>"""


def page_door(html: str) -> str:
    """A page this service serves, with PAGE_DOOR first in its head, before any script that could POST. With no token on this machine the doors are open, and the page is as it was."""
    if token() is None:
        return html
    i = html.lower().find("<head>")
    at = i + len("<head>") if i >= 0 else 0
    return html[:at] + PAGE_DOOR + html[at:]


def book_path(node: str) -> Path:
    return config.state_dir() / "person" / f"{node}.json"


def floor_path(node: str) -> Path:
    return book_path(node).with_suffix(".floor")


def floor(node: str) -> int:
    """No number at or below this is taken on this node: the moment its file was lost, in the milliseconds a word's number counts in. 0 if it never was."""
    try:
        return int(floor_path(node).read_text().strip())
    except (OSError, ValueError):
        return 0


def lost(node: str) -> None:
    """This node's file couldn't be read, and the numbers it held went with it. Any old word could have been used already, so none is taken from now on: a person signs again. On a code-only
    node the file was the only copy of those numbers, and an old clear freed a newer alarm (rescore2_safety.md S9)."""
    from .keep import write_atomic
    try:
        write_atomic(floor_path(node), str(max(floor(node), int(time.time() * 1000))).encode())
    except OSError as exc:                                 # a second fault at once: it's in the log, and the node is held all the same
        log.error("%s: the floor for a person's numbers can't be written: %s", node, exc)


def book(node: str) -> dict[str, Any] | None:
    """What this node's file says, or None if it has none. A file that can't be read is moved aside and comes back as {"bad": why}: it may have held an alarm, so the caller holds the node.
    Its numbers went with it, so lost() sets the floor."""
    p = book_path(node)
    try:
        d = json.loads(p.read_text())
        limits = d.get("limits", {}) if isinstance(d, dict) else None
        if not isinstance(d, dict) or not isinstance(d.get("seq", {}), dict) or not all(isinstance(v, int) for v in d.get("seq", {}).values()) \
                or not isinstance(limits, dict) or not all(isinstance(x, dict) and all(isinstance(v, (int, float)) for v in x.values()) for x in limits.values()):
            raise ValueError("not this file's shape")
        return d
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        try:
            p.replace(p.with_suffix(".bad"))
        except OSError:
            pass
        lost(node)
        log.error("%s: a person's file can't be read: %s", node, exc)
        return {"bad": f"{type(exc).__name__}: {str(exc)[:120]}"}


def limit_said(node: str, msg: dict[str, Any]) -> dict[str, Any]:
    """A signed set_limit that went through, as a change to note(): {"limits": ...} with this one on top of what the file holds. Nothing for any other word."""
    if msg.get("verb") != "set_limit":
        return {}
    limits = (book(node) or {}).get("limits") or {}
    return {"limits": {**limits, msg["group"]: {**limits.get(msg["group"], {}), msg["name"]: msg["value"]}}}


def numbers(node: str, keys: Any = ()) -> dict[str, int]:
    """The last number each signer's key has used on this node. keys: the signers this node takes. Each starts above the floor, if the file was ever lost."""
    seq = dict((book(node) or {}).get("seq") or {})
    low = floor(node)
    return {**seq, **{k: max(int(seq.get(k, 0)), low) for k in keys}} if low else seq


def note(node: str, **changes: Any) -> None:
    """Changes this node's file, on disk when it returns: keep.write_atomic, so the file is the old one or the new one, whatever happens. A number only ever goes up."""
    from .keep import write_atomic
    with _write:
        d = book(node) or {}
        d.pop("bad", None)
        seq = {**(d.get("seq") or {})}
        for k, v in (changes.pop("seq", None) or {}).items():
            seq[k] = max(int(seq.get(k, 0)), int(v))
        write_atomic(book_path(node), json.dumps({**d, **changes, "seq": seq}, sort_keys=True).encode())


# ---------------------------------------------------------------------- what a deploy and a person run
def key_path(role: str) -> Path:
    return config.CONFIG_DIR / "secrets" / f"person_{role}.key"


def trust(role: str, public: str) -> None:
    """This machine's nodes take this person's word: agent.json's person_keys, and nothing else in the file changes."""
    from .keep import write_atomic
    if not re.fullmatch(r"[0-9a-f]{64}", public):
        sys.exit("a public key is 64 hex characters: what `key` printed")
    p = config.CONFIG_DIR / "agent.json"
    cfg = json.loads(p.read_text()) if p.exists() else {}
    write_atomic(p, json.dumps({**cfg, "person_keys": {**(cfg.get("person_keys") or {}), role: public}}, indent=2).encode())


def word(a: argparse.Namespace) -> dict[str, Any]:
    """A person's word, signed here. Its number is the time, so this needs no count of its own and each word goes above the last."""
    import httpx

    from . import signing
    from .node import policy as P
    role = a.role or ("facility_manager" if a.node.startswith("bld-") else "owner")
    if not key_path(role).exists():
        sys.exit(f"no key for {role} here: `key {role}` makes one, and `trust` on the node's machine takes it")
    msg = P.signed(signing.load_or_create(key_path(role)), role, a.node, int(time.time() * 1000), a.verb, a.group or P.SITE, a.name, a.value)
    try:
        r = httpx.post(f"{a.url.rstrip('/')}/person/{a.node}", json=msg, headers=headers(), timeout=30)
    except httpx.HTTPError as exc:
        return {"status": "blocked", "why": f"the door didn't answer: {type(exc).__name__}"}
    return r.json() if r.status_code == 200 else {"status": "blocked", "why": f"the door answered {r.status_code}: {r.text[:200]}"}


def main(argv: list[str] | None = None) -> int:
    from .node import policy as P
    ap = argparse.ArgumentParser(prog="python -m flexgrid_core.doors", description="The control token and a person's key. Nothing here prints a secret.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("token", help="make secrets/control_token, if there's none")
    k = sub.add_parser("key", help="make a person's key, if there's none, and print its public half")
    t = sub.add_parser("trust", help="this machine's nodes take this person's word")
    for x in (k, t):
        x.add_argument("role", choices=["owner", "facility_manager"])
    t.add_argument("public_key")
    w = sub.add_parser("word", help="send a person's signed word to a node")
    w.add_argument("node")
    w.add_argument("verb", choices=[v for v in P.PERSON_VERBS if v != "read"])
    w.add_argument("--url", required=True, help="the fleet's or the buildings' status API")
    w.add_argument("--role", choices=["owner", "facility_manager"], help="owner for a home, facility_manager for a building, unless you say")
    w.add_argument("--group", default="")
    w.add_argument("--name", default="")
    w.add_argument("--value", type=float)
    a = ap.parse_args(argv)
    os.umask(0o077)                                      # what's written here is nobody else's, from its first byte
    if a.cmd == "token":
        made = token() is None
        if made:
            config.write_secret(TOKEN_FILE, secrets.token_urlsafe(32))
        print(f"{config.CONFIG_DIR / 'secrets' / TOKEN_FILE}: {'made' if made else 'already there'}")
    elif a.cmd == "key":
        from . import signing
        print(signing.public_hex(signing.load_or_create(key_path(a.role))))
    elif a.cmd == "trust":
        trust(a.role, a.public_key)
        print(f"{a.role} is on file in {config.CONFIG_DIR / 'agent.json'}: nodes here read it at their next start")
    else:
        out = word(a)
        print(json.dumps(out))
        return 0 if out.get("status") == "done" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
