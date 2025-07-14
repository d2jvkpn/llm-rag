#!/usr/bin/env python3
import os, argparse
from pathlib import Path

import yaml
from qdrant_client import QdrantClient # models as qmodels


if len(os.sys.argv) == 1:
    print("!!! command is required: delete")
    os.sys.exit(1)

command = os.sys.argv[1]

parser = argparse.ArgumentParser(
    description="parse commandline arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--config", help="config filepath", default=Path("configs") / "local.yaml")
parser.add_argument("--force", help="without interactive", action="store_true")

args = parser.parse_args(os.sys.argv[2:])


with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


addr = config['qdrant']['addr']
#collection = config['qdrant']['collection']
_model = Path(config['embedding']['model']).name.replace(':', '--')
collection = f"{config['embedding']['provider']}__{_model}"

client = QdrantClient(url=addr, prefer_grpc=True, https=False, timeout=30)

if command == "delete":
    if client.collection_exists(collection):
        # print(f"--> deleting collection: {collection}")
        if not args.force:
            ans = input(f"Delete collection {collection}? (yes/no)").strip()
            if ans != "yes":
                os.sys.exit(0)

        client.delete_collection(collection)
    else:
        print(f"!!! qdrant collecton exixts: {collection}")
else:
    print(f"!!! unknown command: {command}")
    os.sys.exit(1)
