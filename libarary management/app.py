import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__, static_folder='static', static_url_path='')

DATABASE = 'library.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    # Create books table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            isbn TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'Available',
            borrower TEXT,
            borrow_date TEXT,
            due_date TEXT
        )
    ''')
    
    # Check if table is empty to insert initial seed data
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM books')
    if cursor.fetchone()[0] == 0:
        seed_books = [
            ("The Great Gatsby", "F. Scott Fitzgerald", "9780743273565", "Available", None, None, None),
            ("To Kill a Mockingbird", "Harper Lee", "9780061120084", "Borrowed", "Alice Smith", "2026-06-25", "2026-07-09"),
            ("1984", "George Orwell", "9780451524935", "Available", None, None, None),
            ("The Hobbit", "J.R.R. Tolkien", "9780547928227", "Available", None, None, None),
            ("Pride and Prejudice", "Jane Austen", "9780141439518", "Available", None, None, None)
        ]
        conn.executemany('''
            INSERT INTO books (title, author, isbn, status, borrower, borrow_date, due_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', seed_books)
        conn.commit()
    conn.close()

# Initialize the database
init_db()

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Helper to serialize database rows to dict
def serialize_book(row):
    return {
        'id': row['id'],
        'title': row['title'],
        'author': row['author'],
        'isbn': row['isbn'],
        'status': row['status'],
        'borrower': row['borrower'],
        'borrow_date': row['borrow_date'],
        'due_date': row['due_date']
    }

# API: Get all books (with optional search filter)
@app.route('/api/books', methods=['GET'])
def get_books():
    search_query = request.args.get('search', '').strip()
    conn = get_db_connection()
    if search_query:
        cursor = conn.execute('''
            SELECT * FROM books 
            WHERE title LIKE ? OR author LIKE ? OR isbn LIKE ?
        ''', (f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'))
    else:
        cursor = conn.execute('SELECT * FROM books')
    
    books = [serialize_book(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(books)

# API: Get specific book
@app.route('/api/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({'error': 'Book not found'}), 404
    return jsonify(serialize_book(row))

# API: Create new book
@app.route('/api/books', methods=['POST'])
def add_book():
    data = request.get_json() or {}
    title = data.get('title', '').strip()
    author = data.get('author', '').strip()
    isbn = data.get('isbn', '').strip()

    if not title or not author or not isbn:
        return jsonify({'error': 'Title, Author, and ISBN are required fields.'}), 400

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO books (title, author, isbn)
            VALUES (?, ?, ?)
        ''', (title, author, isbn))
        conn.commit()
        book_id = cursor.lastrowid
        row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
        new_book = serialize_book(row)
        conn.close()
        return jsonify(new_book), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': f'Book with ISBN {isbn} already exists.'}), 400

# API: Update book details
@app.route('/api/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    data = request.get_json() or {}
    title = data.get('title', '').strip()
    author = data.get('author', '').strip()
    isbn = data.get('isbn', '').strip()

    if not title or not author or not isbn:
        return jsonify({'error': 'Title, Author, and ISBN are required fields.'}), 400

    conn = get_db_connection()
    row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({'error': 'Book not found'}), 404

    try:
        conn.execute('''
            UPDATE books 
            SET title = ?, author = ?, isbn = ?
            WHERE id = ?
        ''', (title, author, isbn, book_id))
        conn.commit()
        updated_row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
        updated_book = serialize_book(updated_row)
        conn.close()
        return jsonify(updated_book)
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': f'Book with ISBN {isbn} already exists.'}), 400

# API: Delete book
@app.route('/api/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({'error': 'Book not found'}), 404

    conn.execute('DELETE FROM books WHERE id = ?', (book_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': f'Book "{row["title"]}" successfully deleted.'})

# API: Borrow book
@app.route('/api/books/<int:book_id>/borrow', methods=['POST'])
def borrow_book(book_id):
    data = request.get_json() or {}
    borrower = data.get('borrower_name', '').strip()
    due_date = data.get('due_date', '').strip()

    if not borrower or not due_date:
        return jsonify({'error': 'Borrower Name and Due Date are required.'}), 400

    conn = get_db_connection()
    row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({'error': 'Book not found'}), 404

    if row['status'] == 'Borrowed':
        conn.close()
        return jsonify({'error': 'Book is already borrowed.'}), 400

    borrow_date = datetime.now().strftime('%Y-%m-%d')
    conn.execute('''
        UPDATE books 
        SET status = 'Borrowed', borrower = ?, borrow_date = ?, due_date = ?
        WHERE id = ?
    ''', (borrower, borrow_date, due_date, book_id))
    conn.commit()
    
    updated_row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    updated_book = serialize_book(updated_row)
    conn.close()
    return jsonify(updated_book)

# API: Return book
@app.route('/api/books/<int:book_id>/return', methods=['POST'])
def return_book(book_id):
    conn = get_db_connection()
    row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({'error': 'Book not found'}), 404

    if row['status'] == 'Available':
        conn.close()
        return jsonify({'error': 'Book is already available.'}), 400

    conn.execute('''
        UPDATE books 
        SET status = 'Available', borrower = NULL, borrow_date = NULL, due_date = NULL
        WHERE id = ?
    ''', (book_id,))
    conn.commit()

    updated_row = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    updated_book = serialize_book(updated_row)
    conn.close()
    return jsonify(updated_book)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
