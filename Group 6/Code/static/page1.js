document.addEventListener("DOMContentLoaded", function() {
    // Function to enable dark mode
    function enableDarkMode() {
        document.body.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'enabled');
    }

    // Function to disable dark mode
    function disableDarkMode() {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'disabled');
    }

    // Check for saved dark mode preference in localStorage on page load
    const darkMode = localStorage.getItem('darkMode');
    if (darkMode === 'enabled') {
        enableDarkMode();  // Apply dark mode if it's enabled
    }

    // Toggle dark mode when the switch is clicked
    const darkModeToggle = document.getElementById('mode-changer');
    darkModeToggle.addEventListener('click', () => {
        const darkMode = localStorage.getItem('darkMode');
        if (darkMode !== 'enabled') {
            enableDarkMode();  // Enable dark mode and save preference
        } else {
            disableDarkMode();  // Disable dark mode and save preference
        }
    });

    // Features menu toggle
    const featuresLink = document.querySelector("#features > a");
    const subMenu = document.querySelector("#features .sub-menu");
    featuresLink.addEventListener("click", function(event) {
        event.preventDefault();
        subMenu.classList.toggle("show");
    });
});

function toggleFeedbackForm() {
    const dashboard = document.getElementById('dashboard');
    const container = document.getElementById('feedback-container');

    if (container.style.display === 'none' || container.style.display === '') {
        // Show feedback container and hide the dashboard
        container.style.display = 'block';
        dashboard.style.display = 'none';
    } else {
        // Hide feedback container and show the dashboard
        container.style.display = 'none';
        dashboard.style.display = 'block';
    }
}


document.getElementById('feedback-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Create a new FormData object to send data
    const formData = new FormData(this);

    // Display a loading message or spinner
    const thankYouMessage = document.getElementById('thank-you-message');
    thankYouMessage.innerText = "Submitting your feedback...";
    thankYouMessage.style.display = 'block';

    // Send data to the server
    fetch(this.action, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            // Show thank you message after successful submission
            thankYouMessage.innerText = "Thank you for your feedback!";

            // Wait for a few seconds before redirecting to the dashboard
            setTimeout(() => {
                window.location.href = '/dashboard'; // Adjust to your dashboard URL
            }, 2500); //2.5s
        } else {
            // Display error message if the response is not ok
            thankYouMessage.innerText = "Error submitting feedback. Please try again.";
        }
    })
    .catch(error => {
        console.error('Error:', error);
        thankYouMessage.innerText = "An error occurred. Please try again.";
    });
});
