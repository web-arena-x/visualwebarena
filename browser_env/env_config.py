# websites domain
import os

REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
HOMEPAGE = os.environ.get("HOMEPAGE", "")
CLASSIFIEDS = os.environ.get("CLASSIFIEDS", "")
CLASSIFIEDS_RESET_TOKEN = os.environ.get("CLASSIFIEDS_RESET_TOKEN", "")
REDDIT_RESET_URL = os.environ.get("REDDIT_RESET_URL", "")

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
}

URL_MAPPINGS = {
    REDDIT: "http://reddit.com",
    SHOPPING: "http://onestopmarket.com",
    WIKIPEDIA: "http://wikipedia.org",
    HOMEPAGE: "http://homepage.com",
    CLASSIFIEDS: "http://classifieds.com",
}
