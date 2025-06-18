from collections import Counter

import pandas as pd
import requests


def search_tag(keyword, category_id: int, limit=20):
    """Danbooruのタグを検索する"""

    url = "https://danbooru.donmai.us/tags.json"
    params = {
        "search[category]": category_id,
        "search[order]": "count",
        "search[name_matches]": f"*{keyword}*",
        "limit": limit,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("APIリクエストに失敗しました。")
        return []

    results = response.json()
    if not results:
        print("No results found.")
        return []

    df = pd.DataFrame(results)

    # created_at と updated_atからyyyymmdd形式の日付を抽出
    df["created_at"] = df.created_at.apply(lambda x: x.split("T")[0])
    df["updated_at"] = df.updated_at.apply(lambda x: x.split("T")[0])

    return df[["id", "name", "post_count", "created_at", "updated_at", "words"]]


def get_character_tags_from_work_tag(work_tag, max_posts=100):
    url = "https://danbooru.donmai.us/posts.json"
    params = {
        "tags": work_tag,
        "limit": 100,  # 1回のAPIで最大100件。max_posts超える場合はページング対応要
    }
    char_counter = Counter()
    post_count = 0

    while post_count < max_posts:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("APIリクエスト失敗")
            break
        posts = response.json()
        if not posts:
            break
        for post in posts:
            for tag in post["tag_string_character"].split():
                char_counter[tag] += 1
        post_count += len(posts)
        if len(posts) < 100:
            break
        # 次ページ用にIDをずらす（API仕様に合わせて調整）
        params["page"] = post_count // 100 + 1

    # 上位キャラクターtagを返す
    result = char_counter.most_common(20)

    return pd.DataFrame(result, columns=["tag", "count"])


# Danbooruのキャラクタータグを検索するときのcategory_idマスタ
category_master = pd.DataFrame(
    [
        {"category_id": 0, "content": "General", "detail": "一般(普通の単語・形容など)"},
        {"category_id": 1, "content": "Artist", "detail": "作者(絵師)タグ"},
        {"category_id": 3, "content": "Copyright", "detail": "作品・シリーズ・版権名"},
        {"category_id": 4, "content": "Character", "detail": "キャラクタータグ"},
        {"category_id": 5, "content": "Metadata", "detail": "メタデータ(例: rating:explicit など)"},
    ]
)


def example():
    keyword = "princess_connect"
    display(
        search_tag(keyword, category_id=4, limit=10),  # danbooruタグをキーワードから検索
        get_character_tags_from_work_tag(keyword, 10),  # 作品名からキャラクター名を検索
    )
    keyword = "blue_archive"
    display(
        search_tag(keyword, category_id=4, limit=10),  # danbooruタグをキーワードから検索
        get_character_tags_from_work_tag(keyword, 10),  # 作品名からキャラクター名を検索
    )


if __name__ == "__main__":
    example()
