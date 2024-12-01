document.addEventListener('DOMContentLoaded', () => {
    const addIncomeButton = document.getElementById('add-income');
    const addExpenseButton = document.getElementById('add-expense');

    addIncomeButton?.addEventListener('click', addIncome);
    addExpenseButton?.addEventListener('click', addExpense);

    function addIncome() {
        // Functionality to add income
        alert('Income added!');
    }

    function addExpense() {
        // Functionality to add expense
        alert('Expense added!');
    }
});

function showContent(section) {
    const mainContent = document.getElementById('main-content');
    mainContent.innerHTML = ''; // Clear the current content

    let content = '';
    switch (section) {
        case 'home':
            content = `
                <section class="dashboard">
                <div class="categories">
                    <div class="category" id="Balance">
                        <h2>Balance</h2>
                        <p>INR 3,321.91</p>
                    </div>
                    <div class="category" id="shopping">
                        <h3>INR 33.50</h3>
                        <p>Shopping</p>
                    </div>
                    <div class="category" id="food">
                        <h3>INR 98.39</h3>
                        <p>Food & Drinks</p>
                    </div>
                    <div class="category" id="bills">
                        <h3>INR 93.30</h3>
                        <p>Bills & Utilities</p>
                    </div>
                    <div class="category" id="others">
                        <h3>INR 252.90</h3>
                        <p>Others</p>
                    </div>
                </div>
                <div class="transactions">
                    <h2>Recent Transactions</h2>
                    <table>
                        <tr>
                            <th>Purpose</th>
                            <th>Category</th>
                            <th>Sum</th>
                            <th>Date</th>
                        </tr>
                        <!-- Transactions will be inserted here -->
                    </table>
                </div>
            </section>
            `;
            break;
        case 'incomeExpenseTrack':
            content = `
                <h2>Income/Expense Track</h2>
                <div class="form-container">
                    <form id="income-expense-form">
                        <label for="type">Type:</label>
                        <select id="type" name="type">
                            <option value="income">Income</option>
                            <option value="expense">Expense</option>
                        </select>

                        <label for="category">Category:</label>
                        <select id="category" name="category">
                            <option value="salary">Salary</option>
                            <option value="gifts">Gifts</option>
                            <option value="shopping">Shopping</option>
                            <option value="food">Food & Drinks</option>
                            <option value="bills">Bills</option>
                            <option value="others">Others</option>
                        </select>

                        <label for="amount">Amount:</label>
                        <input type="number" id="amount" name="amount" required>

                        <label for="date">Date:</label>
                        <input type="date" id="date" name="date" required>

                        <label for="purpose">Purpose:</label>
                        <input type="text" id="purpose" name="purpose" required>

                        <button type="submit">Add Entry</button>
                    </form>
                </div>

                <div class="entries-table">
                    <h3>Entries</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Type</th>
                                <th>Category</th>
                                <th>Amount</th>
                                <th>Date</th>
                                <th>Purpose</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="entries-list">
                            <!-- Entries will be dynamically populated here -->
                        </tbody>
                    </table>
                </div>
            `;
            break;
        case 'budget':
            content = `
                <h2>Budget</h2>
                <p>Set and manage your budget effectively.</p>
            `;
            break;
        case 'expensePrediction':
            content = `
                <h2>Expense Prediction</h2>
                <p>Analyze and predict your future expenses.</p>
            `;
            break;
        case 'investmentRecommendation':
            content = `
                <h2>Investment Recommendation</h2>
                <p>Get personalized investment recommendations.</p>
            `;
            break;
        default:
            content = `<h2>Welcome</h2><p>Select a section from the menu.</p>`;
            break;
    }

    mainContent.innerHTML = content;

    if (section === 'incomeExpenseTrack') {
        setupIncomeExpenseForm();
    }
}

function setupIncomeExpenseForm() {
    const form = document.getElementById('income-expense-form');
    form.addEventListener('submit', (event) => {
        event.preventDefault();
        addEntry();
    });
}

function addEntry() {
    const type = document.getElementById('type').value;
    const category = document.getElementById('category').value;
    const amount = document.getElementById('amount').value;
    const date = document.getElementById('date').value;
    const purpose = document.getElementById('purpose').value;

    const newRow = document.createElement('tr');
    newRow.innerHTML = `
        <td>${type}</td>
        <td>${category}</td>
        <td>${amount} INR</td>
        <td>${date}</td>
        <td>${purpose}</td>
        <td><button onclick="deleteEntry(this)">Delete</button></td>
    `;

    document.getElementById('entries-list').appendChild(newRow);
    document.getElementById('income-expense-form').reset();
}

function deleteEntry(button) {
    const row = button.parentNode.parentNode;
    document.getElementById('entries-list').removeChild(row);
}