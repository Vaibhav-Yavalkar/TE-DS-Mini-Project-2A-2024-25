document.addEventListener('DOMContentLoaded', function() {
    fetch('/get_budget_statistics')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.querySelector('#budget-table tbody');
            tableBody.innerHTML = ''; // Clear previous data

            data.budgets.forEach(budget => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${budget.category}</td>
                    <td>${budget.date}</td>
                    <td>${budget.amount}</td>
                    <td>${budget.remaining_amount}</td>
                    <td><button onclick="viewGraph('${budget.id}')">View Graph</button></td>
                `;
                tableBody.appendChild(row);
            });
        });
});

function viewGraph(budgetId) {
    // Open a new window or display a modal with the graph for the selected budget
    window.open(`/view_budget_graph/${budgetId}`, '_blank');
}
