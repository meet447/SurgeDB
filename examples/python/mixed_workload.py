import json
import random
import time
import urllib.request

BASE_URL = "http://localhost:3000"
COLLECTION = "mixed_workload"
DIMENSIONS = 384
PREFILL = 10000
OPS = 10000
SEARCH_RATIO = 0.7
INSERT_RATIO = 0.3
K = 10


def request(method, path, payload=None):
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.read()


def random_vector(rng):
    return [rng.random() for _ in range(DIMENSIONS)]


def main():
    rng = random.Random(123)
    request("DELETE", f"/collections/{COLLECTION}")
    status, _ = request(
        "POST",
        "/collections",
        {
            "name": COLLECTION,
            "dimensions": DIMENSIONS,
            "distance_metric": "Cosine",
        },
    )
    if status >= 400:
        raise SystemExit("Failed to create collection")

    batch = []
    for i in range(PREFILL):
        batch.append(
            {
                "id": f"vec_{i}",
                "vector": random_vector(rng),
                "metadata": {"tag": "even" if i % 2 == 0 else "odd"},
            }
        )
        if len(batch) == 200:
            request(
                "POST",
                f"/collections/{COLLECTION}/vectors/batch",
                {"vectors": batch},
            )
            batch = []
    if batch:
        request(
            "POST",
            f"/collections/{COLLECTION}/vectors/batch",
            {"vectors": batch},
        )

    search_lat = []
    insert_lat = []
    start = time.perf_counter()

    for i in range(OPS):
        if rng.random() < SEARCH_RATIO:
            payload = {
                "vector": random_vector(rng),
                "k": K,
                "include_metadata": False,
            }
            t0 = time.perf_counter()
            request("POST", f"/collections/{COLLECTION}/search", payload)
            search_lat.append(time.perf_counter() - t0)
        else:
            payload = {
                "id": f"insert_{i}",
                "vector": random_vector(rng),
                "metadata": {"tag": "live"},
            }
            t0 = time.perf_counter()
            request("POST", f"/collections/{COLLECTION}/vectors", payload)
            insert_lat.append(time.perf_counter() - t0)

    duration = time.perf_counter() - start
    qps = OPS / duration

    def percentile(data, pct):
        if not data:
            return 0.0
        data_ms = sorted([x * 1000.0 for x in data])
        return data_ms[int(pct * (len(data_ms) - 1))]

    print(f"Mixed QPS: {qps:.2f}")
    print(
        f"Search P50/P95: {percentile(search_lat, 0.5):.2f} / {percentile(search_lat, 0.95):.2f} ms"
    )
    print(
        f"Insert P50/P95: {percentile(insert_lat, 0.5):.2f} / {percentile(insert_lat, 0.95):.2f} ms"
    )


if __name__ == "__main__":
    main()
