import json
import random
import time
import urllib.request

BASE_URL = "http://localhost:3000"
COLLECTION = "filter_heavy"
DIMENSIONS = 384
PREFILL = 20000
K = 10
QUERIES = 5000


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
    rng = random.Random(7)
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
                "metadata": {
                    "tag": "even" if i % 2 == 0 else "odd",
                    "score": float(i),
                },
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

    latencies = []
    start = time.perf_counter()
    for _ in range(QUERIES):
        payload = {
            "vector": random_vector(rng),
            "k": K,
            "filter": {"Exact": ["tag", "even"]},
            "include_metadata": False,
        }
        t0 = time.perf_counter()
        request("POST", f"/collections/{COLLECTION}/search", payload)
        latencies.append(time.perf_counter() - t0)

    duration = time.perf_counter() - start
    qps = QUERIES / duration
    latencies_ms = sorted([x * 1000.0 for x in latencies])
    p50 = latencies_ms[int(0.5 * (len(latencies_ms) - 1))]
    p95 = latencies_ms[int(0.95 * (len(latencies_ms) - 1))]

    print(f"Filter-heavy QPS: {qps:.2f}")
    print(f"P50: {p50:.2f} ms | P95: {p95:.2f} ms")


if __name__ == "__main__":
    main()
