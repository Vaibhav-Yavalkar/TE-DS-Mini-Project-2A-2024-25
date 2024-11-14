// var swiper = new Swiper(".slide-content", {
//     slidesPerView: 2,
//     spaceBetween: 25,
//     centerSlide:'true',
//     grabCursor:'true',
//     pagination: {
//       el: ".swiper-pagination",
//       clickable: true,
//       dynamicBullets: true,
//     },
//     navigation: {
//       nextEl: ".swiper-button-next",
//       prevEl: ".swiper-button-prev",
//     },
//   });

// document.addEventListener('DOMContentLoaded', function() {
//   // Handle redirection on click of the redirectDiv
//   var redirectDiv = document.getElementById('redirectDiv');
//   if (redirectDiv) {
//       redirectDiv.onclick = function() {
//           window.location.href = '/mainpage';  // Replace with the appropriate route
//       };
//   }

//   // Initialize Swiper
//   var swiper = new Swiper(".slide-content", {
//       slidesPerView: 2,
//       spaceBetween: 25,
//       centeredSlides: true, // Corrected property name from 'centerSlide' to 'centeredSlides'
//       grabCursor: true,
//       pagination: {
//           el: ".swiper-pagination",
//           clickable: true,
//           dynamicBullets: true,
//       },
//       navigation: {
//           nextEl: ".swiper-button-next",
//           prevEl: ".swiper-button-prev",
//       },
//   });
// });

document.addEventListener('DOMContentLoaded', function () {
    // Initialize Swiper only after the DOM is fully loaded
    var swiper = new Swiper(".slide-content", {
        slidesPerView: 2,
        spaceBetween: 25,
        centerSlide: 'true',
        grabCursor: 'true',
        pagination: {
            el: ".swiper-pagination",
            clickable: true,
            dynamicBullets: true,
        },
        navigation: {
            nextEl: ".swiper-button-next",
            prevEl: ".swiper-button-prev",
        },
    });

    // Handle redirection for the mainpage if applicable
    var redirectDiv = document.getElementById('redirectDiv');
    if (redirectDiv) {
        redirectDiv.onclick = function () {
            window.location.href = '/mainpage';  // Adjust this if necessary
        };
    }
});