let menu = document.querySelector('#menu-icon');
let navbar = document.querySelector('.navbar');

menu.onclick = () => {
    menu.classList.toggle('bx-x');
    navbar.classList.toggle('active');
}

window.onscroll = () => {
    menu.classList.remove('bx-x');
    navbar.classList.remove('active');
}

    let currentIndex = 0;

    function moveSlide(direction) {
        const slides = document.querySelector('.slides');
        const totalSlides = document.querySelectorAll('.slide').length;

        // Update current index
        currentIndex += direction;

        // Loop around the slides
        if (currentIndex < 0) {
            currentIndex = totalSlides - 1; // Go to last slide
        } else if (currentIndex >= totalSlides) {
            currentIndex = 0; // Go to first slide
        }

        // Move the slides
        slides.style.transform = `translateX(-${currentIndex * 100}%)`;
    }

