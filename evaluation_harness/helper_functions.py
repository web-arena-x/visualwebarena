"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""

import base64
from io import BytesIO
import json
from datetime import datetime, timezone
from typing import Any, Union
from urllib.parse import urlparse

import requests
from beartype import beartype
from beartype.typing import Dict, List
from playwright.sync_api import Page
from evaluation_harness.env_config import ACCOUNTS, SHOPPING
from llms.providers.openai_utils import (
    generate_from_openai_chat_completion,
)

GPT_MODEL = "gpt-4-1106-preview"


class PseudoPage:
    def __init__(self, original_page: Page | None, url: str):
        self.url = url
        self.original_page = original_page

    def from_dict(self, data: dict[str, Any]):
        for key, value in data.items():
            if key == "image_str_cache":
                image_cache = []
                for cur_image_str in value:
                    cur_image_cache = []
                    for image_str in cur_image_str:
                        cur_image_cache.append(
                            Image.open(BytesIO(base64.b64decode(image_str)))
                        )
                    image_cache.append(cur_image_cache)
                setattr(self, "image_cache", image_cache)
            else:
                setattr(self, key, value)

    def __getattr__(self, attr: str) -> Any:
        if self.original_page is None:
            return getattr(self, attr)

        # Delegate attribute access to the original page object
        if attr in ["url"]:
            return getattr(self, attr)
        else:
            return getattr(self.original_page, attr)


@beartype
def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


@beartype
def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(f"{SHOPPING}/rest/V1/orders", params=params, headers=header)
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url


@beartype
def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


@beartype
def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


@beartype
def shopping_get_sku_latest_review_text(sku: str) -> str:
    """Get the latest review text for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    text: str = response_obj[-1]["detail"]
    return text


@beartype
def shopping_get_sku_latest_review_title(sku: str) -> str:
    """Get the latest review title for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    title: str = response_obj[-1]["title"]
    return title


@beartype
def shopping_get_sku_product_page_url(sku: str) -> str:
    """Get product page url from sku"""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{SHOPPING}/rest/V1/products/{sku}", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    for custom_attributes in response_obj["custom_attributes"]:
        if custom_attributes["attribute_code"] == "url_key":
            return f"{SHOPPING}/{custom_attributes['value']}.html"
    return ""


@beartype
def shopping_get_all_product_order(
    page: Page | PseudoPage,
) -> List[Dict[str, str]]:
    """
    Get info of all product in a given order page.

    Example output:
    [
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Strawberry, 19oz Box (12 Bars)\nSize\n12 Count (Pack of 1)",
            "options": {
                "Size": "12 Count (Pack of 1)"
            },
            "sku": "B00MXUFL0E",
            "price": "$24.50",
            "qty": "Ordered2",
            "subtotal": "$49.00"
        },
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Chocolatey Chip Cookie Dough, 19oz Box (12 Bars)",
            "sku": "B07ZD2PB9F",
            "price": "$42.30",
            "qty": "Ordered2",
            "subtotal": "$84.60"
        }
    ]
    """
    try:
        result = page.evaluate(
            f"""
(() => {{
    try {{
        const products = [...document.querySelector("#my-orders-table").getElementsByTagName('tbody')].map(
            (x) => {{
                return [...x.getElementsByTagName('td')].reduce(function(obj, y) {{
                    const key = y.className.split(' ')[1];
                    obj[key] = y.outerText;
                    // check if options exist
                    if (key === 'name' && y.querySelector('dl')) {{
                        var option_dict = {{}}
                        const options = [...y.querySelector('dl').children];
                        for (let i = 0; i < options.length; i += 2) {{
                            option_dict[options[i].outerText] = options[i+1].outerText;
                        }}
                        obj['options'] = option_dict;
                    }}
                    return obj;
                }}, {{}})
            }}
        );
        return products;
    }} catch (e) {{
        // If any errors are caught, return an empty string
        return e;
        return [];
    }}
}})();
            """
        )
        return result
    except Exception as e:
        result = []

    return result


@beartype
def shopping_get_order_product_name_list(page: Page | PseudoPage) -> str:
    try:
        products = shopping_get_all_product_order(page)

        return " |OR| ".join([p["name"] for p in products])
    except Exception:
        return ""


@beartype
def shopping_get_order_product_quantity(page: Page | PseudoPage, sku: str) -> int:
    try:
        if "|OR|" in sku:
            skus = sku.split(" |OR| ")
        else:
            skus = [sku]

        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() in skus:
                # Ordered{qty}
                return int(product["qty"][7:])
        return 0
    except Exception:
        return 0


@beartype
def shopping_get_order_product_option(
    page: Page | PseudoPage, sku: str, option_name: str
) -> str:
    try:
        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() == sku:
                # Ordered{qty}
                return product["options"][option_name]
        return ""
    except Exception as e:
        return ""


@beartype
def shopping_get_product_attributes(page: Page | PseudoPage, attribute: str) -> str:
    # Get the values of all cells in the table for the given attribute
    try:
        result = page.evaluate(
            f"""
                (() => {{
                try {{
                    // Create an array of search terms, splitting the string by ' |OR| '
                    const searchTerms = '{attribute}'.toLowerCase().split(' |or| ');
                    // Convert the children of the tbody inside the element with the given ID into an array
                    return Array.from(
                    document.querySelector('#productDetails_detailBullets_sections1 > tbody').children
                    )
                    // Filter the array to only include elements where the first child's text includes any of the search terms
                    .filter(x =>
                    searchTerms.some(term => x.children[0].outerText.toLowerCase().includes(term))
                    )
                    // Map over the filtered elements to get the outerText of their second child
                    .map(x => x.children[1].outerText)
                    // Join all the resulting strings with a comma and a space
                    .join(', ')
                }} catch (e) {{
                    // If any errors are caught, return an empty string
                    return ''
                }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


@beartype
def shopping_get_product_price(page: Page | PseudoPage) -> Union[float, int]:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseFloat(document.querySelector(\"#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span\")
                    .outerText.substr(1));
                    return res ? res : 0;
                }})();
            """
        )
    except Exception:
        result = 0

    return result


@beartype
def shopping_get_num_reviews(page: Page | PseudoPage) -> int:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseInt(document.querySelector(\"#tab-label-reviews-title\")
                    .outerText.split(' ')[1]);
                    return res ? res : 0; }}
                )();
            """
        )
    except Exception:
        result = 0

    return result


@beartype
def shopping_get_rating_as_percentage(page: Page | PseudoPage) -> int:
    """Get the rating of the product on the shopping website as a percentage out of 100."""
    try:
        rating = page.evaluate(
            """
                (() => {{
                    ratingPercentage = parseFloat(document.querySelector('.rating-result').title.replace('%', ''));
                    return ratingPercentage ? ratingPercentage : 0;
                }})();
            """
        )
    except Exception:
        rating = 0

    return rating


@beartype
def get_query_text(page: Page | PseudoPage, selector: str) -> str:
    """Get the text content of the element matching the given selector.

    Note that this function DOES NOT perform downcasing.
    """
    try:
        result = page.evaluate(
            f"""
                (() => {{
                    try {{
                        return document.querySelector('{selector}').textContent;
                    }} catch (e) {{
                        return '';
                    }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


@beartype
def get_query_text_lowercase(page: Page | PseudoPage, selector: str) -> str:
    """Get the lowercase text content of the element matching the given selector."""
    return get_query_text(page, selector).lower()


@beartype
def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


@beartype
def reddit_get_post_comment_tree(page: Page | PseudoPage) -> Dict[str, Any]:
    try:
        comment_tree = page.evaluate(
            f"""(function buildCommentTree(node, data_level) {{
    let tree = {{
        "username": node.querySelector(".fg-inherit").outerText,
        "net_score": parseInt(node.querySelector(".vote__net-score").outerText),
        "content": node.querySelector(".comment__content").outerText,
        "time": new Date(node.querySelector('.comment__main > header > h1 > span > time').dateTime),
        "children": []
    }};
    node.querySelectorAll(".comment").forEach((child) => {{
        if (parseInt(child.getAttribute('data-level')) === data_level+1) {{
            tree['children'].push(buildCommentTree(child, data_level+1));
        }}
    }})

    return tree;
}})(document.querySelector("#main"), 0)"""
        )
    except Exception:
        comment_tree = {}

    return comment_tree


@beartype
def reddit_get_latest_comment_obj_by_username(
    page: Page | PseudoPage, username: str
) -> Dict[str, Any]:
    try:
        comment_tree = reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            if node["username"] == username:
                if node["time"] > latest_time:
                    comment = {
                        "username": node["username"],
                        "net_score": node["net_score"],
                        "content": node["content"],
                        "time": node["time"],
                    }
                    latest_time = node["time"]

            for child in node["children"]:
                dfs(child)

        dfs(comment_tree)

    except Exception as e:
        comment = {}
    return comment


@beartype
def reddit_get_latest_comment_content_by_username(
    page: Page | PseudoPage, username: str
) -> str:
    try:
        comment = reddit_get_latest_comment_obj_by_username(page, username)
        content = comment["content"]

    except Exception:
        content = ""

    return content


@beartype
def reddit_get_parent_comment_obj_of_latest_comment_by_username(
    page: Page | PseudoPage, username: str
) -> Dict[str, Any]:
    try:
        comment_tree = reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            for child in node["children"]:
                if child["username"] == username:
                    if child["time"] > latest_time:
                        comment = {
                            "username": node["username"],
                            "net_score": node["net_score"],
                            "content": node["content"],
                            "time": node["time"],
                        }
                        latest_time = child["time"]
                else:
                    dfs(child)

        dfs(comment_tree)

    except Exception:
        comment = {}
    return comment


@beartype
def reddit_get_parent_comment_username_of_latest_comment_by_username(
    page: Page | PseudoPage, username: str
) -> str:
    try:
        comment = reddit_get_parent_comment_obj_of_latest_comment_by_username(
            page, username
        )
        username = comment["username"]

    except Exception:
        username = ""

    return username


@beartype
def gitlab_get_project_memeber_role(page: Page | PseudoPage, account_name: str) -> str:
    # get the account index
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role


@beartype
def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=512,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response, response
        return 1.0


def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0


def llm_fuzzy_exact_match(pred: str, reference: str, task: str) -> tuple[str, float]:
    user_message = f"""Determine if the prediction is correct by comparing it with the reference answer.
- The reference answer presents the correct answer in its minimal form.
- When the reference answer is about time duration, distance, or quantity, the prediction can be in a different format, but the information should be equivalent.

Task: {task}
Reference answer: {reference}
Prediction: {pred}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude with the score using the format: "Answer: Correct"/"Answer: Incorrect"
""".strip()

    messages: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message.strip()},
    ]

    response = generate_from_openai_chat_completion(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        context_length=0,
    ).lower()

    if "answer: correct" in response:
        return response, 1.0
    else:
        assert "answer: incorrect" in response, response
        return response, 0.0


def llm_fuzzy_must_include(pred: str, reference: str, task: str) -> tuple[str, float]:
    user_message = f"""Determine if the prediction contains the required information.
- The prediction is considered as containing the required information if it can entail the required information.
- The prediction can contain additional information.
- When the information is about time duration, distance, or quantity, the prediction can be in a different format, but the information should be equivalent.

Task: {task}
Required information: {reference}
Prediction: {pred}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude with the score using the format: "Answer: Contain"/"Answer: Not contain"
""".strip()

    messages: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message.strip()},
    ]

    response = generate_from_openai_chat_completion(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        context_length=0,
    ).lower()

    if "answer: contain" in response:
        return response, 1.0
    else:
        assert "answer: not contain" in response, response
        return response, 0.0


def llm_fuzzy_na_match(pred: str, reference: str, task: str) -> tuple[str, float]:
    user_message = f"""Determine if the predicted reason given for why the task cannot be completed is correct.
- The prediction is only considered as correct if the reference and the prediction can entail each other.
- Make sure you understand the task and the reason presented in the reference. Do not be loose with the interpretation.

Task: {task}
Reference: {reference}
Prediction: {pred}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude using the format: "Answer: Correct"/"Answer: Incorrect".
""".strip()

    messages: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message.strip()},
    ]

    response = generate_from_openai_chat_completion(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        context_length=0,
    ).lower()

    if "answer: correct" in response:
        return response, 1.0
    else:
        assert "answer: incorrect" in response, response
        return response, 0.0


@beartype
def llm_question_answering(
    question: str, answer: str, passage: str
) -> tuple[str, float]:
    user_message = f"""Provide a binary answer to the question given the passage.
- Carefully read the passage,
- Make sure you understand the question and do not be loose with the interpretation.

Passage: {passage}
Question: {question}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude using the format: "Answer: Yes"/"Answer: No".
""".strip()

    messages: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message.strip()},
    ]

    response = generate_from_openai_chat_completion(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        context_length=0,
    ).lower()

    if f"answer: {answer.lower()}" in response:
        return response, 1.0
    else:
        return response, 0.0


@beartype
def llm_context_aware_question_answering(
    question: str, answer: str, passage: str, context: str
) -> tuple[str, float]:
    user_message = f"""Provide a binary answer to the question given the passage.
- `Task` indicates the context in which the passage is presented.
- Carefully read the passage.
- Make sure you understand the question and do not be loose with the interpretation.

Task: {context}
Passage: {passage}
Question: {question}

After the examination (do not repeat the sentences below):
- Briefly justify your answer.
- Conclude using the format: "Answer: Yes"/"Answer: No".
""".strip()

    messages: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message.strip()},
    ]

    response = generate_from_openai_chat_completion(
        model=GPT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        context_length=0,
    ).lower()

    if f"answer: {answer.lower()}" in response:
        return response, 1.0
    else:
        return response, 0.0
