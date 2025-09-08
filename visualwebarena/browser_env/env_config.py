# websites domain
import os

DATASET = os.environ["DATASET"]
if DATASET not in ["webarena", "visualwebarena"]:
    raise ValueError("Please set the DATASET environment variable, the possible options are `webarena`, `visualwebarena` and `miniwob++`")

# WebArena
if DATASET == "webarena":
    REDDIT = os.environ.get("REDDIT", "")
    SHOPPING = os.environ.get("SHOPPING", "")
    SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
    GITLAB = os.environ.get("GITLAB", "")
    WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
    MAP = os.environ.get("MAP", "")
    HOMEPAGE = os.environ.get("HOMEPAGE", "")
    assert (
        REDDIT
        and SHOPPING
        and SHOPPING_ADMIN
        and GITLAB
        and WIKIPEDIA
        and MAP
        and HOMEPAGE
    ), (
        f"Please setup the URLs to each site. Current: \n"
        + f"Reddit: {REDDIT}\n"
        + f"Shopping: {SHOPPING}\n"
        + f"Shopping Admin: {SHOPPING_ADMIN}\n"
        + f"Gitlab: {GITLAB}\n"
        + f"Wikipedia: {WIKIPEDIA}\n"
        + f"Map: {MAP}\n"
        + f"Homepage: {HOMEPAGE}\n"
    )
    
    URL_MAPPINGS = {
        REDDIT: "http://reddit.com",
        SHOPPING: "http://onestopmarket.com",
        SHOPPING_ADMIN: "http://luma.com/admin",
        GITLAB: "http://gitlab.com",
        WIKIPEDIA: "http://wikipedia.org",
        MAP: "http://openstreetmap.org",
        HOMEPAGE: "http://homepage.com",
    }
    
elif DATASET == "visualwebarena":
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
    ), (
        f"Please setup the URLs and tokens to each site. Current: "
        + f"Reddit: {REDDIT}"
        + f"Shopping: {SHOPPING}"
        + f"Wikipedia: {WIKIPEDIA}"
        + f"Homepage: {HOMEPAGE}"
        + f"Classifieds: {CLASSIFIEDS}"
        + f"Classifieds reset token: {CLASSIFIEDS_RESET_TOKEN}"
    )
    
    URL_MAPPINGS = {
        REDDIT: "http://reddit.com",
        SHOPPING: "http://onestopmarket.com",
        WIKIPEDIA: "http://wikipedia.org",
        HOMEPAGE: "http://homepage.com",
        CLASSIFIEDS: "http://classifieds.com",
    }
    
else:
    raise ValueError(f"Dataset not implemented: {DATASET}")


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