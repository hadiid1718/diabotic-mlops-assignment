import json
import urllib.request
import urllib.error

URL = 'http://127.0.0.1:8000/predict'

def post(payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode('utf-8')
            print('Status:', resp.status)
            print('Body:', body)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8')
        except Exception:
            body = ''
        print('HTTPError Status:', e.code)
        print('Body:', body)
    except Exception as e:
        print('Error:', e)

if __name__ == '__main__':
    tests = [
        # Test 1 - diabetic
        {"age":65, "urea":7.5, "cr":52.0, "hba1c":11.2, "chol":6.1, "tg":2.8, "hdl":0.9, "ldl":3.5, "vldl":1.2, "bmi":32.5, "gender":"M"},
        # Test 2 - non-diabetic
        {"age":28, "urea":4.2, "cr":48.0, "hba1c":5.1, "chol":4.0, "tg":1.2, "hdl":1.8, "ldl":2.1, "vldl":0.6, "bmi":22.0, "gender":"F"},
        # Test 3 - invalid gender
        {"age":45, "urea":5.0, "cr":50.0, "hba1c":6.0, "chol":5.0, "tg":1.5, "hdl":1.2, "ldl":2.5, "vldl":0.8, "bmi":25.0, "gender":"X"},
        # Test 4 - missing fields
        {"age":50, "urea":5.0, "cr":50.0}
    ]

    for i, t in enumerate(tests, 1):
        print('\n=== Test', i, 'payload ===')
        print(json.dumps(t))
        post(t)
