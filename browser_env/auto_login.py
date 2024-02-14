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
    GITLAB,
    SHOPPING_ADMIN,
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

# If you want to test WebArena tasks, uncomment the following lines to add the configs:
# SITES.extend(["shopping_admin", "gitlab"])
# URLS.extend([f"{SHOPPING_ADMIN}/dashboard", f"{GITLAB}/-/profile"])
# EXACT_MATCH.extend([True, True])
# KEYWORDS.extend(["Dashboard", ""])
assert len(SITES) == len(URLS) == len(EXACT_MATCH) == len(KEYWORDS)

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


def renew_comb(comb: list[str], auth_folder: str = "./.auth") -> None:
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

    if "shopping_admin" in comb:
        username = ACCOUNTS["shopping_admin"]["username"]
        password = ACCOUNTS["shopping_admin"]["password"]
        page.goto(f"{SHOPPING_ADMIN}")
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click()

    if "gitlab" in comb:
        username = ACCOUNTS["gitlab"]["username"]
        password = ACCOUNTS["gitlab"]["password"]
        page.goto(f"{GITLAB}/users/sign_in")
        page.get_by_test_id("username-field").click()
        page.get_by_test_id("username-field").fill(username)
        page.get_by_test_id("username-field").press("Tab")
        page.get_by_test_id("password-field").fill(password)
        page.get_by_test_id("sign-in-button").click()

    context.storage_state(path=f"{auth_folder}/{'.'.join(comb)}_state.json")

    context_manager.__exit__()


def main(auth_folder: str = "./.auth") -> None:
    pairs = list(combinations(SITES, 2))

    with ThreadPoolExecutor(max_workers=8) as executor:
        for pair in pairs:
            # Auth doesn't work on this pair as they share the same cookie
            if "reddit" in pair and (
                "shopping" in pair or "shopping_admin" in pair
            ):
                continue
            executor.submit(
                renew_comb, list(sorted(pair)), auth_folder=auth_folder
            )

        for site in SITES:
            executor.submit(renew_comb, [site], auth_folder=auth_folder)

    for c_file in glob.glob(f"{auth_folder}/*.json"):
        comb = c_file.split("/")[-1].rsplit("_", 1)[0].split(".")
        for cur_site in comb:
            url = URLS[SITES.index(cur_site)]
            keyword = KEYWORDS[SITES.index(cur_site)]
            match = EXACT_MATCH[SITES.index(cur_site)]
            print(c_file, url, keyword, match)
            assert not is_expired(Path(c_file), url, keyword, match), url


if __name__ == "__main__":
    main()
