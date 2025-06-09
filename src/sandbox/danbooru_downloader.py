import math
import random

from dataclasses import dataclass
from pathlib import Path
from time import sleep

import requests

from config import DanbooruConfig
from tqdm import tqdm


@dataclass
class DownloadParams:
    tags: str
    title: str  # タイトル名。保存先ディレクトリ名に使う
    limit: int = 30  # 取得枚数
    sample: bool = False  # サンプリング有効
    verbose: bool = True


class DanbooruDownloader:
    BASE_URL = "https://danbooru.donmai.us/posts.json"
    COUNT_URL = "https://danbooru.donmai.us/counts/posts.json"

    def __init__(self, config: DanbooruConfig):
        self.username = config.username
        self.api_key = config.api_key
        self.output_dir = config.output_dir
        self.sleep_time = config.sleep_time

    def get_post_count(self, tags: str) -> int:
        """指定したタグの投稿数を取得する"""
        params = {
            "tags": tags,
            "login": self.username,
            "api_key": self.api_key,
        }
        resp = requests.get(self.COUNT_URL, params=params)
        resp.raise_for_status()
        return resp.json()["counts"]["posts"]

    def fetch_random_posts(self, tags: str, limit: int) -> list:
        """ランダムに投稿を取得する"""
        total_count = self.get_post_count(tags)
        if total_count == 0:
            print("No posts found!")
            return []

        per_page = 20
        max_page = math.ceil(total_count / per_page)
        sampled_pages = random.sample(range(1, max_page + 1), k=min(limit, max_page))
        posts = []
        for page in sampled_pages:
            params = {
                "tags": tags,
                "limit": per_page,
                "page": page,
                "login": self.username,
                "api_key": self.api_key,
            }
            resp = requests.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            page_posts = resp.json()
            if page_posts:
                post = random.choice(page_posts)
                posts.append(post)
            if len(posts) >= limit:
                break
        return posts

    def fetch_first_n_posts(self, tags: str, limit: int) -> list:
        """最初のN件の投稿を取得する"""
        # 普通に前からlimit件集める
        posts = []
        page = 1
        per_page = 20
        while len(posts) < limit:
            params = {
                "tags": tags,
                "limit": per_page,
                "page": page,
                "login": self.username,
                "api_key": self.api_key,
            }
            resp = requests.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            page_posts = resp.json()
            if not page_posts:
                break
            posts.extend(page_posts)
            page += 1
        return posts[:limit]

    def select_posts(self, tags: str, limit: int, sample: bool) -> list:
        """sampleがTrueならランダムに、Falseなら最初のN件のポストを取得"""
        if sample:
            return self.fetch_random_posts(tags, limit)
        else:
            return self.fetch_first_n_posts(tags, limit)

    def download_file(self, file_url: str, output_path: Path, verbose: bool = True):
        """指定されたURLからファイルをダウンロードし、指定されたパスに保存"""
        try:
            img_data = requests.get(file_url).content
            with open(output_path, "wb") as f:
                f.write(img_data)
            if verbose:
                print(f"Downloaded: {output_path}")
            return True
        except Exception as e:
            print(f"Failed to download {file_url}: {e}")
            return False

    def download(self, params: DownloadParams):
        """指定されたパラメータで画像をダウンロード"""
        output_dir = self.output_dir / str(params.title)
        output_dir.mkdir(exist_ok=True)

        posts = self.select_posts(params.tags, params.limit, params.sample)

        downloaded = 0
        failed_files = []
        for post in tqdm(posts, total=len(posts)):
            file_url = post.get("file_url")
            if not file_url:
                continue
            output_file = output_dir / f"{post['id']}_{Path(file_url).name}"
            if self.download_file(file_url, output_file, verbose=params.verbose):
                downloaded += 1
                sleep(self.sleep_time)
            else:
                failed_files.append(file_url)
            if downloaded >= params.limit:
                break
        if failed_files:
            print(f"Failed to download {len(failed_files)} files:")
            for file_url in failed_files:
                print(file_url)

