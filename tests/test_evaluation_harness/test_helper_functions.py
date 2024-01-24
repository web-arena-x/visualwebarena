import json
import os

from browser_env import ScriptBrowserEnv
from browser_env.env_config import *
from evaluation_harness.helper_functions import (
    get_query_text,
    get_query_text_lowercase,
    reddit_get_latest_comment_content_by_username,
    reddit_get_parent_comment_username_of_latest_comment_by_username,
    shopping_get_num_reviews,
    shopping_get_order_product_option,
    shopping_get_order_product_quantity,
    shopping_get_product_attributes,
    shopping_get_product_price,
    shopping_get_rating_as_percentage,
    shopping_get_sku_latest_review_rating,
    shopping_get_sku_latest_review_text,
    shopping_get_sku_latest_review_title,
    shopping_get_sku_product_page_url,
)

HEADLESS = True
config_file_folder = "tests/test_evaluation_harness/configs"


def test_shopping_get_attributes(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_shopping_attr.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/shopping_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(f"{SHOPPING}/nec-np4100-6200-lumen-xga-dlp-projector.html")
    manufacturer = shopping_get_product_attributes(
        env.page, "manufacturer |OR| brand name"
    )

    env.reset(options={"config_file": config_file})
    env.page.goto(
        f"{SHOPPING}/lg-50nano80upa-50-nanocell-4k-nano80-series-smart-ultra-hd-tv-with-an-lg-sn6y-3-1-channel-dts-virtual-high-resolution-soundbar-and-subwoofer-2021.html"
    )
    brand_name = shopping_get_product_attributes(
        env.page, "manufacturer |OR| brand name"
    )

    # remove tmp config file
    os.remove(config_file)
    assert "NEC Displays" in manufacturer
    assert "LG" in brand_name


def test_get_query_text(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_shopping_query.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/shopping_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(
        f"{SHOPPING}/la-guapa-virtual-projection-keyboard-laser-projection-bluetooth-wireless-keyboard-for-smart-phone-pc-tablet-laptop-wireless-laser-projection-keyboard-silver.html"
    )
    query_text = get_query_text(
        env.page, "#maincontent > div.page-title-wrapper.product > h1 > span"
    )
    assert "Projection Keyboard" in query_text

    query_text_lower = get_query_text_lowercase(
        env.page, "#maincontent > div.page-title-wrapper.product > h1 > span"
    )
    assert "projection keyboard" in query_text_lower

    # remove tmp config file
    os.remove(config_file)


def test_get_product_price(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_shopping_price.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/shopping_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(
        f"{SHOPPING}/la-guapa-virtual-projection-keyboard-laser-projection-bluetooth-wireless-keyboard-for-smart-phone-pc-tablet-laptop-wireless-laser-projection-keyboard-silver.html"
    )
    product_price = shopping_get_product_price(env.page)
    assert product_price == 26.99

    env.page.goto(f"{SHOPPING}")
    product_price = shopping_get_product_price(env.page)
    assert product_price == 0

    # remove tmp config file
    os.remove(config_file)


def test_get_num_reviews(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_shopping_num_reviews.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/shopping_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(
        f"{SHOPPING}/la-guapa-virtual-projection-keyboard-laser-projection-bluetooth-wireless-keyboard-for-smart-phone-pc-tablet-laptop-wireless-laser-projection-keyboard-silver.html"
    )
    product_reviews = shopping_get_num_reviews(env.page)
    assert product_reviews == 12
    # remove tmp config file
    os.remove(config_file)


def test_get_product_rating(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = (
        f"{config_file_folder}/config_shopping_rating_percentage.json.tmp"
    )

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/shopping_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(
        f"{SHOPPING}/v8-energy-healthy-energy-drink-steady-energy-from-black-and-green-tea-pomegranate-blueberry-8-ounce-can-pack-of-24.html"
    )
    product_rating = shopping_get_rating_as_percentage(env.page)
    assert product_rating == 57

    env.page.goto(f"{SHOPPING}/catalogsearch/advanced/")
    product_rating = shopping_get_rating_as_percentage(env.page)
    assert product_rating == 0
    # remove tmp config file
    os.remove(config_file)


def test_shopping_get_sku_product_page_url(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    true_url = f"{SHOPPING}/xbox-wireless-controller-phantom-white-special-edition.html"
    url = shopping_get_sku_product_page_url("B07P3L5GMW")

    assert url == true_url


# NOTE: These fail if the B07N4Q7P67 reviews are modified and are hence just useful as a sanity check.
def test_shopping_get_sku_latest_review_text(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    true_text = "Good quality"
    text = shopping_get_sku_latest_review_text("B07N4Q7P67")

    assert text == true_text, f"Expected: {true_text}\nGot: {text}"


def test_shopping_get_sku_latest_review_title(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    true_title = "Fits Nintendo switch"
    title = shopping_get_sku_latest_review_title("B07N4Q7P67")

    assert title == true_title, f"Expected: {true_title}\nGot: {title}"


def test_shopping_get_sku_latest_review_text(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    true_text = "Good quality"
    text = shopping_get_sku_latest_review_text("B07N4Q7P67")

    assert text == true_text, f"Expected: {true_text}\nGot: {text}"


def test_shopping_get_sku_latest_review_rating(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    true_rating = "100"
    rating = shopping_get_sku_latest_review_rating("B07N4Q7P67")

    assert rating == true_rating, f"Expected: {true_rating}\nGot: {rating}"


def test_shopping_get_order_product_quantity(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_test_shopping_get_order_product_quantity.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/shopping_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(f"{SHOPPING}/sales/order/view/order_id/170/")

    quantity = shopping_get_order_product_quantity(env.page, "B087QSCXGT")
    assert quantity == 1

    quantity = shopping_get_order_product_quantity(env.page, "B08JLHHCM6")
    assert quantity == 1

    quantity = shopping_get_order_product_quantity(env.page, "B09LQTV3RX")
    assert quantity == 1

    # remove tmp config file
    os.remove(config_file)


def test_shopping_get_order_product_option(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_test_shopping_get_order_product_option.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/shopping_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(f"{SHOPPING}/sales/order/view/order_id/170/")

    option = shopping_get_order_product_option(env.page, "B09LQTV3RX", "Color")
    assert option == "Blue"

    option = shopping_get_order_product_option(env.page, "B09LQTV3RX", "Size")
    assert option == "Large"

    # remove tmp config file
    os.remove(config_file)


def test_reddit_get_latest_comment_content_by_username(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_test_reddit_get_post_comment_tree.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/reddit_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(f"{REDDIT}/f/AskReddit/116809")

    comment_content = reddit_get_latest_comment_content_by_username(
        env.page, "DavosLostFingers"
    )
    assert comment_content == "Constantly on their phone"

    # remove tmp config file
    os.remove(config_file)


def test_reddit_get_parent_comment_username_of_latest_comment_by_username(
    script_browser_env: ScriptBrowserEnv,
) -> None:
    env = script_browser_env
    config_file = f"{config_file_folder}/config_test_reddit_get_parent_comment_tree.json.tmp"

    with open(config_file, "w") as f:
        json.dump({"storage_state": ".auth/reddit_state.json"}, f)
    env.reset(options={"config_file": config_file})
    env.page.goto(f"{REDDIT}/f/memes/127590")

    comment_content = (
        reddit_get_parent_comment_username_of_latest_comment_by_username(
            env.page, "Veryhawtwoman"
        )
    )
    assert comment_content == "Da_Bro_Main"

    # remove tmp config file
    os.remove(config_file)
