# websites domain
import os

# VWA
REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
HOMEPAGE = os.environ.get("HOMEPAGE", "")
CLASSIFIEDS = os.environ.get("CLASSIFIEDS", "")
CLASSIFIEDS_RESET_TOKEN = os.environ.get("CLASSIFIEDS_RESET_TOKEN", "")
REDDIT_RESET_URL = os.environ.get("REDDIT_RESET_URL", "")

# WebArena
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
GITLAB = os.environ.get("GITLAB", "")
MAP = os.environ.get("MAP", "")

assert (
    REDDIT
    and SHOPPING
    and WIKIPEDIA
    and HOMEPAGE
    and CLASSIFIEDS
    and CLASSIFIEDS_RESET_TOKEN
#    and REDDIT_RESET_URL
), (
    f"Please setup the URLs and tokens to each site. Current: "
    + f"Reddit: {REDDIT}"
    # + f"  Reddit reset url: {REDDIT_RESET_URL}"
    + f"Shopping: {SHOPPING}"
    + f"Wikipedia: {WIKIPEDIA}"
    + f"Homepage: {HOMEPAGE}"
    + f"Classifieds: {CLASSIFIEDS}"
    + f"  Classifieds reset token: {CLASSIFIEDS_RESET_TOKEN}"
)


ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "classifieds": {
        "username": "blake.sullivan@gmail.com",
        "password": "Password.123",
    },
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
}

URL_MAPPINGS = {
    # VWA:
    REDDIT: "http://reddit.com",
    SHOPPING: "http://onestopmarket.com",
    WIKIPEDIA: "http://wikipedia.org",
    HOMEPAGE: "http://homepage.com",
    CLASSIFIEDS: "http://classifieds.com",
}

# WebArena:
if SHOPPING_ADMIN:
    URL_MAPPINGS[SHOPPING_ADMIN] = "http://luma.com/admin"
if GITLAB:
    URL_MAPPINGS[GITLAB] = "http://gitlab.com"
if MAP:
    URL_MAPPINGS[MAP] = "http://openstreetmap.org"
