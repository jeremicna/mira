# transcription_store.py

import json
import lvldb

DB_PATH = "transcriptions.ldb"

# Open LevelDB once, globally
db = lvldb.DB(DB_PATH, create_if_missing=True)


def _load_json_list(raw_value):
    if raw_value is None:
        return []

    # lvldb can give you bytes or str depending on usage
    if isinstance(raw_value, bytes):
        text = raw_value.decode("utf-8")
    else:
        text = raw_value

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # If the value is corrupted or not a list, reset
        return []

    if isinstance(data, list):
        return data

    # If someone stored a single object before, wrap it
    return [data]


def append_transcription(uuid_str: str, transcription: dict) -> None:
    """
    Append one transcription dict to the list stored at key 'transcriptions:<uuid>'.

    - Keeps chronological order by always appending at the end
    - Serializes as JSON array
    """

    key = f"transcriptions:{uuid_str}"

    existing = db.get(key)
    conversations = _load_json_list(existing)

    conversations.append(transcription)

    db.put(key, json.dumps(conversations, ensure_ascii=False).encode("utf-8"))


def get_transcriptions(uuid_str: str):
    """
    Return the list of transcriptions for this uuid.
    Empty list if none.
    """
    key = f"transcriptions:{uuid_str}"
    raw = db.get(key)
    return _load_json_list(raw)


def delete_transcriptions(uuid_str: str):
    """
    Delete all stored transcriptions for this uuid.
    """
    key = f"transcriptions:{uuid_str}"
    db.delete(key)


def close_db():
    """
    Call this on clean shutdown if you want.
    """
    db.close()