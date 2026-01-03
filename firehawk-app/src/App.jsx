import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './Login';
import Dashboard from './Dashboard';
import AlertDetails from './AlertDetails';
import './App.css';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null); // store FULL user object from backend

  // called from Login.jsx: onLogin(data.user)
  const handleLogin = (userFromBackend) => {
    setUser(userFromBackend);      // { id, departmentId, userType, ... }
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
    localStorage.removeItem('user');
  };

  // fallback if user not set yet
  const currentUserType = (user?.userType || 'operator');

  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            isLoggedIn ? (
              <Navigate to="/dashboard" />
            ) : (
              <Login onLogin={handleLogin} />
            )
          }
        />
        <Route
          path="/dashboard"
          element={
            isLoggedIn ? (
              <Dashboard
                userData={user}          // pass full user object to Dashboard
                onLogout={handleLogout}
              />
            ) : (
              <Navigate to="/" />
            )
          }
        />
        <Route
          path="/alert/:alertId"
          element={
            isLoggedIn ? (
              <AlertDetails
                userData={user}        // pass full user object
                onLogout={handleLogout}
              />
            ) : (
              <Navigate to="/" />
            )
          }
        />F
      </Routes>
    </Router>
  );
}

export default App;
