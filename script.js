function signup() {
    var email = document.getElementById('signupEmail').value;
    var password = document.getElementById('signupPassword').value;
    var confirmPassword = document.getElementById('confirmPassword').value;

    // Validate if any input field is empty
    if (!email || !password || !confirmPassword) {
        alert('Please enter all required fields.');
        return;
    }

    // Validate password match
    if (password !== confirmPassword) {
        alert('Passwords do not match');
        return;
    }

    // Store user data in localStorage (for demo purposes)
    var user = { email: email, password: password };
    localStorage.setItem('user', JSON.stringify(user));

    alert('User successfully created!');

    // Redirect to login page
    window.location.href = 'index.html';
}

function login() {
    var email = document.getElementById('loginEmail').value;
    var password = document.getElementById('loginPassword').value;

    // Validate if any input field is empty
    if (!email || !password) {
        alert('Please enter all required fields.');
        return;
    }

    // Retrieve user data from localStorage (for demo purposes)
    var storedUser = localStorage.getItem('user');

    if (storedUser) {
        var user = JSON.parse(storedUser);

        // Check if login credentials match
        if (user.email === email && user.password === password) {
            alert('Login successful!');

            // Redirect to main page
            window.location.href = 'main.html';
        } else {
            alert('Invalid login credentials');
        }
    } else {
        alert('User not found. Please sign up.');
    }
}
