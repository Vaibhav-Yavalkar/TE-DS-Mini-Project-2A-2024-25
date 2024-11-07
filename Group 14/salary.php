<?php
session_start();

// Check if the user is logged in
if (!isset($_SESSION['logged_in']) || $_SESSION['logged_in'] !== true) {
    // Redirect to login page if session has expired or not logged in
    header("Location: login.php");
    exit();
}
?>
