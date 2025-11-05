from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASSWORD")
ES_INDEX = os.getenv("ES_INDEX")
FIELDS=[]
FIELDS = [
    "ClusterId",
    "Attempt",
    "AttemptEndTime",
    "TransferTotalBytes",
    "TransferStartTime",
    "TransferEndTime",
    "TransferUrl",
    "AttemptTime",
    "machineattrglidein_site0",
    "Endpoint"
]


def read_job_ids(filename="slow_transfers"):
    """Read job IDs from the slow_transfers file.

    Args:
        filename: Path to file containing job IDs (one per line)

    Returns:
        List of job IDs as strings
    """
    job_ids = []
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Job ID file not found: {filename}")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                job_ids.append(line)

    return job_ids


def get_query(job_ids=None):
    """Build Elasticsearch query for transfer history.

    Args:
        job_ids: List of job IDs to query for. If None, reads from slow_transfers file.

    Returns:
        Query dictionary for Elasticsearch
    """
    if job_ids is None:
        job_ids = read_job_ids()

    query = {
        "index": ES_INDEX,
        "scroll": "30s",
        "size": 500,
        "body": {
            "_source": FIELDS,
            "query": {
                "bool": {
                    "filter": [
                        {
                            "terms": {
                                "ClusterId": job_ids
                            }
                        }
                    ],
                },
            },
        },
    }
    return query


def print_csv(docs):
    print(",".join(FIELDS))
    for doc in docs:
        print(",".join([str(doc.get(field,"UNKNOWN")) for field in FIELDS]))


def main():
    # Validate that credentials are loaded
    if not ES_USER or not ES_PASS:
        raise ValueError(
            "ES_USER and ES_PASSWORD must be set in .env file.\n"
            "Create a .env file with:\n"
            "ES_USER=your_username\n"
            "ES_PASSWORD=your_password"
        )

    client = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS))
    query = get_query()
    docs = []
    for doc in scan(client=client, query=query.pop("body"), **query):
        # import pdb; pdb.set_trace()
        docs.append(doc["_source"])
    print_csv(docs)

if __name__ == "__main__":
    main()
