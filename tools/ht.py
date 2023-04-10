import json
import sys
import hashlib


def pprint_file(f):
    """Pretty prints f's contents if it's JSON, else just cats it."""
    try:
        with open(f) as h:
            data = h.read()
            if f.lower().endswith('.json'):
                j = json.loads(data)
                print(json.dumps(j, indent=True))
            else:
                print(data)
    except (IOError, ValueError) as e:
        print(f"Could not read file: {e}")


def generate_auth_token(s):
    m = hashlib.sha256()
    m.update(s.encode('utf-8'))
    print(m.hexdigest())


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'print':
        pprint_file(sys.argv[2])
        return
    elif len(sys.argv) == 3 and sys.argv[1] == 'auth-token':
        generate_auth_token(sys.argv[2])
        return


if __name__ == "__main__":
    main()
