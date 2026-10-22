"""Git da Google Colab senza mai mostrare il token.

Il token GitHub va salvato nei Secrets di Colab (icona a forma di chiave
nella barra a sinistra) con il nome GITHUB_TOKEN. Non scriverlo MAI in una
cella: finirebbe nel notebook e quindi su GitHub.

Uso tipico in un notebook:
    from src.colab_git import commit_and_push
    commit_and_push("Messaggio del commit", user="tuo-username", repo="MirAi")
"""
from __future__ import annotations

import os
import re
import subprocess

from src import config

# Qualunque cosa somigli a un token o a un URL con credenziali viene oscurata.
_SECRET_PATTERNS = [
    re.compile(r"(github_pat_|ghp_|gho_|ghs_)[A-Za-z0-9_]+"),
    re.compile(r"(https?://)[^/@\s]*@"),
]


def _hide(text: str, token: str = "") -> str:
    if token:
        text = text.replace(token, "***")
    for pat in _SECRET_PATTERNS:
        text = pat.sub(lambda m: (m.group(1) if m.group(1).startswith("http") else "") + "***"
                       + ("@" if m.group(0).endswith("@") else ""), text)
    return text


def get_token() -> str:
    from google.colab import userdata  # disponibile solo su Colab
    token = (userdata.get("GITHUB_TOKEN") or "").strip()  # toglie spazi e «a capo» incollati
    if not token:
        raise RuntimeError("Secret GITHUB_TOKEN mancante o senza accesso al notebook.")
    if any(c.isspace() for c in token):
        raise RuntimeError("Il token contiene spazi: ricopialo nel secret GITHUB_TOKEN.")
    return token


def _url(user: str, repo: str, token: str) -> str:
    return f"https://x-access-token:{token}@github.com/{user}/{repo}.git"


def _git(args: list[str], cwd: str, token: str = "") -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    out = _hide((r.stdout + r.stderr).strip(), token)
    if r.returncode != 0:
        safe = " ".join("<url-con-token>" if "x-access-token" in a else a for a in args)
        raise RuntimeError(f"git {safe} fallito:\n{out}")
    return out


def set_identity(name: str, email: str, cwd: str = config.COLAB_REPO_DIR) -> None:
    _git(["config", "user.name", name], cwd)
    _git(["config", "user.email", email], cwd)


def clone_or_pull(user: str, repo: str, dest: str = config.COLAB_REPO_DIR, branch: str = "main") -> None:
    import os
    token = get_token()
    if os.path.isdir(os.path.join(dest, ".git")):
        print(_git(["pull", _url(user, repo, token), branch], dest, token))
    else:
        print(_git(["clone", _url(user, repo, token), dest], "/content", token))
        # il remote salvato NON contiene il token
        _git(["remote", "set-url", "origin", f"https://github.com/{user}/{repo}.git"], dest)


def commit_and_push(message: str, user: str, repo: str, paths: list[str] | None = None,
                    cwd: str = config.COLAB_REPO_DIR, branch: str = "main") -> None:
    token = get_token()
    _git(["add", *(paths or ["-A"])], cwd)
    nothing_staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd).returncode == 0
    if nothing_staged:
        print("Niente da committare.")
        return
    print(_git(["commit", "-m", message], cwd))
    print(_git(["push", _url(user, repo, token), f"HEAD:{branch}"], cwd, token) or "Push completato.")


def init_and_push(user: str, repo: str, name: str, email: str,
                  cwd: str = config.COLAB_REPO_DIR, branch: str = "main") -> None:
    """Solo la prima volta: trasforma la cartella in un repository e la carica su GitHub.

    Si può rieseguire senza problemi: salta i passi già fatti.
    """
    token = get_token()
    if not os.path.isdir(os.path.join(cwd, ".git")):
        _git(["init", "-b", branch], cwd)
    set_identity(name, email, cwd)
    _git(["add", "-A"], cwd)
    has_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True).returncode == 0
    has_staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd).returncode != 0
    if not has_head or has_staged:
        msg = "Struttura iniziale del progetto MirAi" if not has_head else "Aggiornamento struttura iniziale"
        print(_git(["commit", "-m", msg], cwd))
    remotes = _git(["remote"], cwd).split()
    if "origin" not in remotes:
        _git(["remote", "add", "origin", f"https://github.com/{user}/{repo}.git"], cwd)
    try:
        print(_git(["push", _url(user, repo, token), f"HEAD:{branch}"], cwd, token) or "Push completato.")
    except RuntimeError as e:
        if "rejected" in str(e) or "fetch first" in str(e):
            raise RuntimeError("Il repository su GitHub non è vuoto: ricrealo SENZA README, "
                               ".gitignore e licenza, poi riprova.") from e
        raise
