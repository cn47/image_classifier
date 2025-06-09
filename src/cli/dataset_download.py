from pathlib import Path

import yaml

from config import DanbooruConfig
from sandbox.danbooru_downloader import DanbooruDownloader, DownloadParams


if __name__ == "__main__":
    downloader = DanbooruDownloader(DanbooruConfig())

    with (Path(__file__).parent / "danbooru_dl.yml").open() as fp:
        params = yaml.safe_load(fp)

    params = [DownloadParams(**p) for p in params]
    for param in params:
        print(f"Downloading images({param.title}) with tags: {param.tags}")
        downloader.download(param)
