document.addEventListener("DOMContentLoaded", function () {
    const loginLink = document.querySelector(".login-link");
    const registerLink = document.querySelector(".register-link");
    const loginForm = document.querySelector(".form-box.login");
    const registerForm = document.querySelector(".form-box.register");

    registerLink.addEventListener("click", () => {
        loginForm.classList.remove("active");
        registerForm.classList.add("active");
    });

    loginLink.addEventListener("click", () => {
        registerForm.classList.remove("active");
        loginForm.classList.add("active");
    });

    // Close icon functionality (if needed)
    const closeIcon = document.querySelector(".icon-close");
    closeIcon.addEventListener("click", () => {
        document.querySelector(".wrapper").style.display = "none";
    });
});
