"""Script to automatically login each website"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from playwright.sync_api import sync_playwright

from browser_env.env_config import (
    ACCOUNTS,
    CLASSIFIEDS,
    REDDIT,
    SHOPPING,
)

HEADLESS = True
SLOW_MO = 0


SITES = ["shopping", "reddit", "classifieds"]
URLS = [
    f"{SHOPPING}/wishlist/",
    f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
    f"{CLASSIFIEDS}/index.php?page=user&action=items",
]
EXACT_MATCH = [True, True, True]
KEYWORDS = ["", "Delete", "My listings"]


def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=True, slow_mo=SLOW_MO)
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    page.goto(url)
    time.sleep(1)
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url


def renew_comb(comb: list[str]) -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()

    if "shopping" in comb:
        username = ACCOUNTS["shopping"]["username"]
        password = ACCOUNTS["shopping"]["password"]
        page.goto(f"{SHOPPING}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click()

    if "reddit" in comb:
        username = ACCOUNTS["reddit"]["username"]
        password = ACCOUNTS["reddit"]["password"]
        page.goto(f"{REDDIT}/login")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "classifieds" in comb:
        username = ACCOUNTS["classifieds"]["username"]
        password = ACCOUNTS["classifieds"]["password"]
        page.goto(f"{CLASSIFIEDS}/index.php?page=login")
        page.locator("#email").fill(username)
        page.locator("#password").fill(password)
        page.get_by_role("button", name="Log in").click()

    context.storage_state(path=f"./.auth/{'.'.join(comb)}_state.json")

    context_manager.__exit__()


def main() -> None:
    for site in SITES:
        renew_comb([site])

    for c_file in glob.glob("./.auth/*.json"):
        comb = c_file.split("/")[-1].rsplit("_", 1)[0].split(".")
        for cur_site in comb:
            url = URLS[SITES.index(cur_site)]
            keyword = KEYWORDS[SITES.index(cur_site)]
            match = EXACT_MATCH[SITES.index(cur_site)]
            print(c_file, url, keyword, match)
            assert not is_expired(Path(c_file), url, keyword, match), url


if __name__ == "__main__":
    main()
