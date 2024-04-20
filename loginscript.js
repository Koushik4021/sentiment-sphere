function login() {
    var email = document.getElementById('loginEmail').value;
    var password = document.getElementById('loginPassword').value;

    // Retrieve user data from localStorage
    var users = JSON.parse(localStorage.getItem('users'));

    if (users) {
        // Check if there are any users registered
        var user = users.find(function(u) {
            return u.email === email && u.password === password;
        });

        if (user) {
            // If the user is found, login successful
            alert('Login successful!');
            // Redirect to the main page
            window.location.href = "main.html";
        } else {
            // If user not found, display error message
            document.getElementById('errorMessage').textContent = "Invalid email or password. Please try again.";
        }
    } else {
        // If no users are registered, display error message
        document.getElementById('errorMessage').textContent = "No users found. Please sign up.";
    }
}
