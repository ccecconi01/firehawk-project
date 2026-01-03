import { useState } from 'react';

export default function Login({ onLogin }) {
  const [userType, setUserType] = useState('operator');
  const [departmentId, setDepartmentId] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleToggle = () => {
    setUserType(userType === 'operator' ? 'viewer' : 'operator');
    setPassword('');
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      // For Visualiser, password is optional
      if (userType === 'operator' && !password) {
        alert('Password is required for Operator login');
        setIsLoading(false);
        return;
      }

      const response = await fetch('https://firehawk-project-production.up.railway.app', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          departmentId: parseInt(departmentId),
          password: password || '',
          userType: userType
        })
      });

      const data = await response.json();

      if (data.success) {
        console.log('Login successful:', data.user);

        // Store user data in localStorage (optional - for persistence)
        localStorage.setItem('user', JSON.stringify(data.user));

        // Pass the entire user object to App.jsx
        if (onLogin) {
          onLogin(data.user);  // Pass the full user data from backend
        }
      } else {
        alert('Login failed: ' + data.message);
      }
    } catch (error) {
      console.error('Login error:', error);
      alert('Error connecting to server');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex justify-center mb-8">
          <img
            src="/logo/firehawklogo.png"
            alt="Firehawk logo"
            className="h-16 sm:h-20 md:h-40 w-auto select-none"
            loading="eager"
            draggable="false"
          />
        </div>

        {/* Toggle */}
        <div className="flex items-center justify-center gap-4 mb-8">
          <span className={`text-sm font-medium transition-colors ${userType === 'viewer' ? 'text-gray-900' : 'text-gray-500'}`}>
            Viewer
          </span>
          <button
            onClick={handleToggle}
            className="relative inline-flex h-8 w-16 items-center rounded-full bg-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-red-600 focus:ring-offset-2"
            style={{ backgroundColor: userType === 'operator' ? '#c41e3a' : '#d1d5db' }}
            aria-label="Toggle between Visualiser and Operator"
            aria-pressed={userType === 'operator'}
          >
            <span
              className="inline-block h-6 w-6 transform rounded-full bg-white shadow-lg transition-transform duration-300 ease-in-out"
              style={{ transform: userType === 'operator' ? 'translateX(32px)' : 'translateX(4px)' }}
            />
          </button>
          <span className={`text-sm font-medium transition-colors ${userType === 'operator' ? 'text-gray-900' : 'text-gray-500'}`}>
            Operator
          </span>
        </div>

        {/* Form */}
        <div className="bg-gray-700 rounded-lg shadow-xl p-8">
          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label className="block text-white text-sm font-medium mb-2">Department ID</label>
              <input
                type="text"
                placeholder="Insert ID"
                value={departmentId}
                onChange={(e) => setDepartmentId(e.target.value)}
                className="w-full bg-gray-600 border border-gray-500 text-white placeholder-gray-400 rounded px-4 py-2 focus:outline-none focus:border-red-600"
                required
              />
            </div>

            {userType === 'operator' && (
              <div>
                <label className="block text-white text-sm font-medium mb-2">Password</label>
                <input
                  type="password"
                  placeholder="Insert Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-gray-600 border border-gray-500 text-white placeholder-gray-400 rounded px-4 py-2 focus:outline-none focus:border-red-600"
                  required
                />
              </div>
            )}

            <div className="flex justify-center pt-4">
              <button
                type="submit"
                disabled={isLoading}
                className="bg-red-600 hover:bg-red-700 text-white font-semibold px-8 py-2 rounded-full transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50"
              >
                {isLoading ? 'Logging in...' : userType === 'operator' ? 'Login' : 'Continue'}
              </button>
            </div>
          </form>
        </div>

        <div className="text-center mt-6 text-gray-600 text-xs">
          <p>Login as {userType === 'viewer' ? 'Viewer' : 'Operator'}</p>
        </div>
      </div>
    </div>
  );
}
