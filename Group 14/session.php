<?php
session_start();

// Function to check if user is logged in
function is_logged_in() {
    return isset($_SESSION['username']);
}

// Optionally, you can set a session timeout here
$session_lifetime = 500; //8 min
if (isset($_SESSION['last_activity']) && (time() - $_SESSION['last_activity']) > $session_lifetime) {
    session_unset();
    session_destroy();
}

$_SESSION['last_activity'] = time();
?>
