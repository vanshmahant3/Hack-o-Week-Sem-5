// Frontend Controller for Athena Library Management System

const API_BASE = '/api';

// DOM Elements
const booksGrid = document.getElementById('books-grid');
const searchInput = document.getElementById('search-input');
const btnOpenAddModal = document.getElementById('btn-open-add-modal');
const inventoryCount = document.getElementById('inventory-count');

// Stats Counters
const countTotal = document.getElementById('count-total');
const countAvailable = document.getElementById('count-available');
const countBorrowed = document.getElementById('count-borrowed');

// Modals
const addModal = document.getElementById('add-modal');
const editModal = document.getElementById('edit-modal');
const borrowModal = document.getElementById('borrow-modal');

// Forms
const addBookForm = document.getElementById('add-book-form');
const editBookForm = document.getElementById('edit-book-form');
const borrowBookForm = document.getElementById('borrow-book-form');

// State
let allBooks = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    fetchBooks();
    setupEventListeners();
});

function setupEventListeners() {
    // Search listener with debounce
    let debounceTimer;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            fetchBooks(e.target.value);
        }, 300);
    });

    // Modal Open Add
    btnOpenAddModal.addEventListener('click', () => {
        openModal('add-modal');
        addBookForm.reset();
    });

    // Form Submissions
    addBookForm.addEventListener('submit', handleAddBook);
    editBookForm.addEventListener('submit', handleEditBook);
    borrowBookForm.addEventListener('submit', handleBorrowBook);
}

// Fetch Books from API
async function fetchBooks(searchQuery = '') {
    try {
        const url = searchQuery 
            ? `${API_BASE}/books?search=${encodeURIComponent(searchQuery)}`
            : `${API_BASE}/books`;
            
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch books');
        
        allBooks = await response.json();
        renderBooks(allBooks);
        updateStats(allBooks);
    } catch (error) {
        console.error(error);
        showToast('Error loading books from server.', 'error');
        booksGrid.innerHTML = `
            <div class="empty-state">
                <span class="empty-state-icon">⚠️</span>
                <h3>Database Error</h3>
                <p>Failed to connect to the backend API. Please make sure Flask is running.</p>
            </div>
        `;
    }
}

// Update Dashboard Statistics
function updateStats(books) {
    const total = books.length;
    const available = books.filter(b => b.status === 'Available').length;
    const borrowed = books.filter(b => b.status === 'Borrowed').length;

    countTotal.textContent = total;
    countAvailable.textContent = available;
    countBorrowed.textContent = borrowed;
    inventoryCount.textContent = `${total} item${total !== 1 ? 's' : ''}`;
}

// Render Book Cards into the Grid
function renderBooks(books) {
    if (books.length === 0) {
        booksGrid.innerHTML = `
            <div class="empty-state">
                <span class="empty-state-icon">🔍</span>
                <h3>No Books Found</h3>
                <p>No titles match your current criteria. Try adjusting your search query or add a new book!</p>
            </div>
        `;
        return;
    }

    booksGrid.innerHTML = books.map(book => {
        const isBorrowed = book.status === 'Borrowed';
        const badgeClass = isBorrowed ? 'status-badge borrowed' : 'status-badge available';
        const cardClass = isBorrowed ? 'book-card borrowed' : 'book-card';
        
        // Dynamic borrow actions or return actions
        const mainActionBtn = isBorrowed 
            ? `<button class="btn btn-warning btn-sm" onclick="returnBook(${book.id})">Return</button>`
            : `<button class="btn btn-success btn-sm" onclick="openBorrowModal(${book.id}, '${escapeHtml(book.title)}')">Borrow</button>`;

        // Borrow details HTML if borrowed
        const borrowDetails = isBorrowed
            ? `<div class="borrow-info">
                 <p><span class="info-label">Borrowed by:</span> <span class="info-val">${escapeHtml(book.borrower)}</span></p>
                 <p><span class="info-label">Due Date:</span> <span class="info-val">${book.due_date}</span></p>
               </div>`
            : '';

        return `
            <div class="${cardClass}">
                <div class="card-header">
                    <h3>${escapeHtml(book.title)}</h3>
                    <span class="${badgeClass}">${book.status}</span>
                </div>
                <div class="card-body">
                    <p class="author">by ${escapeHtml(book.author)}</p>
                    <span class="isbn-tag">🔢 ISBN: ${escapeHtml(book.isbn)}</span>
                    ${borrowDetails}
                </div>
                <div class="card-actions">
                    ${mainActionBtn}
                    <button class="btn btn-secondary btn-sm" onclick="openEditModal(${book.id}, '${escapeHtml(book.title)}', '${escapeHtml(book.author)}', '${escapeHtml(book.isbn)}')">Edit</button>
                    <button class="btn btn-danger btn-sm" onclick="deleteBook(${book.id})" title="Delete Book">🗑️</button>
                </div>
            </div>
        `;
    }).join('');
}

// Add Book
async function handleAddBook(e) {
    e.preventDefault();
    const title = document.getElementById('add-title').value;
    const author = document.getElementById('add-author').value;
    const isbn = document.getElementById('add-isbn').value;

    try {
        const response = await fetch(`${API_BASE}/books`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, author, isbn })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to add book');

        showToast(`Book "${title}" added successfully!`, 'success');
        closeModal('add-modal');
        fetchBooks(searchInput.value);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// Edit Book Form Pre-fill & Open
function openEditModal(id, title, author, isbn) {
    document.getElementById('edit-book-id').value = id;
    document.getElementById('edit-title').value = title;
    document.getElementById('edit-author').value = author;
    document.getElementById('edit-isbn').value = isbn;
    openModal('edit-modal');
}

async function handleEditBook(e) {
    e.preventDefault();
    const id = document.getElementById('edit-book-id').value;
    const title = document.getElementById('edit-title').value;
    const author = document.getElementById('edit-author').value;
    const isbn = document.getElementById('edit-isbn').value;

    try {
        const response = await fetch(`${API_BASE}/books/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, author, isbn })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to update book');

        showToast('Book details updated successfully!', 'success');
        closeModal('edit-modal');
        fetchBooks(searchInput.value);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// Borrow Book Modal Setup & Open
function openBorrowModal(id, title) {
    document.getElementById('borrow-book-id').value = id;
    document.getElementById('borrow-title-display').textContent = title;
    
    // Set default due date to 14 days in the future
    const today = new Date();
    const futureDate = new Date(today.getTime() + 14 * 24 * 60 * 60 * 1000);
    document.getElementById('due-date').value = futureDate.toISOString().split('T')[0];
    
    document.getElementById('borrower-name').value = '';
    openModal('borrow-modal');
}

async function handleBorrowBook(e) {
    e.preventDefault();
    const id = document.getElementById('borrow-book-id').value;
    const borrowerName = document.getElementById('borrower-name').value;
    const dueDate = document.getElementById('due-date').value;

    try {
        const response = await fetch(`${API_BASE}/books/${id}/borrow`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ borrower_name: borrowerName, due_date: dueDate })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to borrow book');

        showToast(`Book checked out by ${borrowerName}!`, 'success');
        closeModal('borrow-modal');
        fetchBooks(searchInput.value);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// Return Book
async function returnBook(id) {
    try {
        const response = await fetch(`${API_BASE}/books/${id}/return`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to return book');

        showToast(`Book returned successfully to shelf.`, 'success');
        fetchBooks(searchInput.value);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// Delete Book
async function deleteBook(id) {
    if (!confirm('Are you sure you want to delete this book from the library catalog?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/books/${id}`, {
            method: 'DELETE'
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to delete book');

        showToast('Book removed from database.', 'info');
        fetchBooks(searchInput.value);
    } catch (error) {
        showToast(error.message, 'error');
    }
}

// Modal Toggle Helpers
function openModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// Toast Notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Simple HTML Escaper to prevent XSS
function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
