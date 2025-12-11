import time
import requests
import sys

API_URL = "http://localhost:7860/docs"

def wait_for_server(timeout=60):
    print("⏳ Waiting for API to start...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(API_URL)
            if r.status_code == 200:
                print("✅ API is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("❌ API did not start within the timeout.")
    return False


if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)

    sys.exit(0)