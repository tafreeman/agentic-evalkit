import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": request["input"],
        "metadata": {"fixture": "echo"},
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)
