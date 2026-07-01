import subprocess
import time
import requests
import sys

# Start Flask application in background
print("Starting Flask app for testing...")
process = subprocess.Popen(
    [sys.executable, 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
time.sleep(2.5)

BASE_URL = "http://127.0.0.1:5000/api"

try:
    # 1. GET /api/books (Seed checks)
    print("\n--- Test 1: GET /api/books ---")
    r = requests.get(f"{BASE_URL}/books")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    books = r.json()
    print(f"Found {len(books)} seed books.")
    assert len(books) >= 5, "Seed books should be present."
    print("Test 1 Passed!")

    # 2. POST /api/books (Create book)
    print("\n--- Test 2: POST /api/books (Create) ---")
    new_book_payload = {
        "title": "Fahrenheit 451",
        "author": "Ray Bradbury",
        "isbn": "9781451673319"
    }
    r = requests.post(f"{BASE_URL}/books", json=new_book_payload)
    assert r.status_code == 201, f"Expected 201, got {r.status_code}"
    new_book = r.json()
    assert new_book["title"] == "Fahrenheit 451"
    assert new_book["status"] == "Available"
    book_id = new_book["id"]
    print(f"Created book successfully. ID is {book_id}")
    print("Test 2 Passed!")

    # 3. GET /api/books/<id> (Retrieve single book)
    print("\n--- Test 3: GET /api/books/<id> (Retrieve) ---")
    r = requests.get(f"{BASE_URL}/books/{book_id}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    book = r.json()
    assert book["author"] == "Ray Bradbury"
    print("Test 3 Passed!")

    # 4. POST /api/books/<id>/borrow (Borrow)
    print("\n--- Test 4: POST /api/books/<id>/borrow (Borrow) ---")
    borrow_payload = {
        "borrower_name": "Bob Martin",
        "due_date": "2026-07-15"
    }
    r = requests.post(f"{BASE_URL}/books/{book_id}/borrow", json=borrow_payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    borrowed_book = r.json()
    assert borrowed_book["status"] == "Borrowed"
    assert borrowed_book["borrower"] == "Bob Martin"
    assert borrowed_book["due_date"] == "2026-07-15"
    print("Test 4 Passed!")

    # 5. POST /api/books/<id>/return (Return)
    print("\n--- Test 5: POST /api/books/<id>/return (Return) ---")
    r = requests.post(f"{BASE_URL}/books/{book_id}/return")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    returned_book = r.json()
    assert returned_book["status"] == "Available"
    assert returned_book["borrower"] is None
    print("Test 5 Passed!")

    # 6. PUT /api/books/<id> (Update)
    print("\n--- Test 6: PUT /api/books/<id> (Update) ---")
    update_payload = {
        "title": "Fahrenheit 451 (Special Edition)",
        "author": "Ray Bradbury Jr.",
        "isbn": "9781451673319"
    }
    r = requests.put(f"{BASE_URL}/books/{book_id}", json=update_payload)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    updated_book = r.json()
    assert updated_book["title"] == "Fahrenheit 451 (Special Edition)"
    assert updated_book["author"] == "Ray Bradbury Jr."
    print("Test 6 Passed!")

    # 7. DELETE /api/books/<id> (Delete)
    print("\n--- Test 7: DELETE /api/books/<id> ---")
    r = requests.delete(f"{BASE_URL}/books/{book_id}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    
    # Confirm it was deleted
    r = requests.get(f"{BASE_URL}/books/{book_id}")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"
    print("Test 7 Passed!")

    print("\n==============================")
    print("All backend API tests passed!")
    print("==============================")

except Exception as e:
    print(f"\n❌ Error during verification: {e}")
    process.terminate()
    sys.exit(1)

finally:
    # Cleanup background process
    print("Shutting down Flask server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    print("Server stopped.")
