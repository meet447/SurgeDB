# Python Workload Examples

These scripts model common workloads for performance testing.

Start the server first:

```bash
cargo run --release -p surgedb-server
```

Search-heavy:

```bash
python3 examples/python/search_heavy.py
```

Mixed workload (search + insert):

```bash
python3 examples/python/mixed_workload.py
```

Filter-heavy:

```bash
python3 examples/python/filter_heavy.py
```
