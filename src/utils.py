#!/usr/bin/env python3
import hashlib, socket
from datetime import datetime


def now():
    at = datetime.now().astimezone()
    # return f"{at}".replace(" ", "T")
    return at.strftime("%Y-%m-%dT%H:%M:%S%z")


def file_md5(file_path):
    #with open(filepath, 'rb') as f:
    #    content = f.read()
    #return hashlib.md5(content).hexdigest()

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def get_local_ip():
    try:
        # 连接到一个外部地址（不需要真的连通，只是用于获取绑定的IP）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS，任意公网IP即可
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # 回退为本地环回地址


def chunk_list(lst, size=10):
    return [lst[i : i + size] for i in range(0, len(lst), size)]

    #### Usage
    #arr = list(range(1, 35))
    #chunks = chunk_list(arr, 10)


def chunk_generator(lst, size=10):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

    #### Usage
    #for chunk in chunk_generator(arr, 10):
    #    print(chunk)
