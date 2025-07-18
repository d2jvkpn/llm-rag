#!/usr/bin/env python3
import os, argparse, json
from pathlib import Path

from src.process_doc import process_doc
from src.utils import file_md5

import yaml
from qdrant_client import QdrantClient # models as qmodels


if len(os.sys.argv) == 1:
    print("!!! command is required: delete_collection, read_doc")
    os.sys.exit(1)

command = os.sys.argv[1]

parser = argparse.ArgumentParser(
    description="parse commandline arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--config", help="config filepath", default=Path("configs") / "local.yaml")
parser.add_argument("--force", help="without interactive", action="store_true")
parser.add_argument('--values', nargs='+')

args = parser.parse_args(os.sys.argv[2:])


with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


addr = config['qdrant']['addr']
#collection = config['qdrant']['collection']
_model = Path(config['embedding']['model']).name.replace(':', '--')
collection = f"{config['embedding']['provider']}__{_model}"

client = QdrantClient(url=addr, prefer_grpc=True, https=False, timeout=30)

if command == "delete_collection":
    if client.collection_exists(collection):
        # print(f"--> deleting collection: {collection}")
        if not args.force:
            ans = input(f"Delete collection {collection}? (yes/no)").strip()
            if ans != "yes":
                os.sys.exit(0)

        client.delete_collection(collection)
    else:
        print(f"!!! qdrant collecton exixts: {collection}")
elif command == "read_doc":
    for p in args.values:
        md5 = file_md5(p)
        doc = process_doc.document2chunks(p, f"md5-{md5}")
        text = json.dumps(doc, ensure_ascii=False, indent=2)
        print(text)
else:
    print(f"!!! unknown command: {command}")
    os.sys.exit(1)
