"""Central API configuration for 16score-desktop.
All URLs are read from here — change BACKEND_URL to switch environments.
"""
import os
import json

# ── Base URL ────────────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get(
    "SCORE16_API",
    "https://beta-api-new.16score.com/api"
).rstrip("/")

# ── AI pipeline (Free-Fire gRPC frame ingest) ───────────────────────────────
AI_HOST = os.environ.get("AI_HOST", "127.0.0.1")
AI_INGEST_PORT = int(os.environ.get("AI_INGEST_PORT", "50052"))

# ── Endpoints ────────────────────────────────────────────────────────────────
def url(path: str) -> str:
    return f"{BACKEND_URL}/{path.lstrip('/')}"

LOGIN_URL          = url("User/Login")
USER_FROM_TOKEN    = url("User/GetUserFromToken")
USER_TENANTS       = url("Tenant/CurrentUserTenants")
GET_LEAGUES        = url("League/GetAllLeagues")
TODAY_MATCHES      = url("LeagueMatch/getTodayLeagueMatch")
LEAGUE_GROUPS      = url("LeagueGroup/getAllLeagueGroup")
MATCHES_BY_LEAGUE  = url("LeagueMatch/getMatchesByLeagueId")
START_MATCH        = url("LeagueMatch/startMatchById")
MATCH_INFO         = url("LeagueMatchData/LeagueMatch/info")
TEAM_PLAYERS       = url("LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players")
KILLFEED           = url("LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed")

# ── Saved config (token, last match) ─────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.expanduser("~"), ".16score_desktop.json")

def load_saved_config() -> dict:
    try:
        with open(_CFG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(data: dict):
    try:
        existing = load_saved_config()
        existing.update(data)
        with open(_CFG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass
