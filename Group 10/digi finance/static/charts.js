// charts.js

function showGraph(category, spentAmount, budgetAmount) {
    // Check if a chart already exists and destroy it
    if (window.budgetChart) {
        window.budgetChart.destroy();
    }

    var ctx = document.getElementById('budgetChart').getContext('2d');
    window.budgetChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Spent Amount'],
            datasets: [{
                label: category,
                data: [spentAmount],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: Math.max(spentAmount, budgetAmount) + 100
                }
            },
            plugins: {
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: budgetAmount,
                            yMax: budgetAmount,
                            borderColor: 'red',
                            borderWidth: 2,
                            label: {
                                content: 'Budget Limit',
                                enabled: true,
                                position: 'end'
                            }
                        }
                    }
                }
            }
        }
    });

    // Show the modal
    var chartModal = new bootstrap.Modal(document.getElementById('chartModal'));
    chartModal.show();
}
