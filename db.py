import lmdb
import json
import os

LMDB_PATH = "./database"
os.makedirs(LMDB_PATH, exist_ok=True)

env = lmdb.open(
    LMDB_PATH,
    map_size=1024 * 1024 * 1024,   # 1GB
    max_dbs=1,
    lock=True
)


# ----------------------------------------
# LMDB HELPERS
# ----------------------------------------
def lmdb_get_json(key: str):
    """Return parsed JSON stored at key or None."""
    with env.begin() as txn:
        raw = txn.get(key.encode())
        if not raw:
            return None
        try:
            return json.loads(raw.decode())
        except:
            return None


def lmdb_put_json(key: str, value):
    """Store value (dict/list) as JSON."""
    with env.begin(write=True) as txn:
        txn.put(key.encode(), json.dumps(value).encode())


if __name__ == "__main__":
    print("\n=== LMDB FULL DUMP ===")

    with env.begin() as txn:
        cursor = txn.cursor()
        found_any = False

        for key_bytes, val_bytes in cursor:
            found_any = True
            key = key_bytes.decode()

            print("\n--------------------------------------")
            print(f"KEY: {key}")
            print("--------------------------------------")

            raw = val_bytes.decode()

            # Try parsing JSON
            try:
                parsed = json.loads(raw)
                print(json.dumps(parsed, indent=2))
            except Exception:
                print(raw)

        if not found_any:
            print("LMDB is empty.")

    print("\n=== END ===\n")
