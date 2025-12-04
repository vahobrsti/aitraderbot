import os
import json
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta, timezone

load_dotenv()

API_KEY = os.getenv("SANTIMENT_API_KEY")
URL = "https://api.santiment.net/graphql"


def fetch_mvrv_1d_graphql():
    # pick a recent window fully inside your allowed interval
    now = datetime.now(timezone.utc)
    to_dt = now
    from_dt = now - timedelta(days=5)

    from_str = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_str = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = f"""
    {{
      getMetric(metric: "mvrv_usd_1d") {{
        timeseriesDataJson(
          slug: "bitcoin"
          from: "{from_str}"
          to: "{to_str}"
          interval: "1d"
        )
      }}
    }}
    """

    headers = {
        "Authorization": f"Apikey {API_KEY}",
        "Content-Type": "application/graphql",
    }

    resp = requests.post(URL, headers=headers, data=query)
    resp.raise_for_status()
    payload = resp.json()

    if "errors" in payload:
        print("Santiment returned errors:")
        print(json.dumps(payload["errors"], indent=2))
        raise RuntimeError("GraphQL returned errors")

    metric_data = payload.get("data", {}).get("getMetric")
    json_str = metric_data.get("timeseriesDataJson")

    points = json.loads(json_str)  # list of { "datetime": "...", "value": ... }
    return points


if __name__ == "__main__":
    data = fetch_mvrv_1d_graphql()
    print(data)
