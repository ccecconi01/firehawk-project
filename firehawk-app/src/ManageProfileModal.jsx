import { useState } from 'react';

export default function ManageProfileModal({ userData, onClose, onPasswordUpdated }) {
  // Form state for the 3 password inputs
  const [currentPassword, setCurrentPassword] = useState(''); // user's existing password
  const [newPassword, setNewPassword] = useState(''); // desired new password
  const [confirmPassword, setConfirmPassword] = useState(''); // confirmation for newPassword

  // UI state
  const [isLoading, setIsLoading] = useState(false); // disables button + shows "Updating..."
  const [message, setMessage] = useState(''); // feedback message text shown above inputs
  const [messageType, setMessageType] = useState(''); // 'success' | 'error' to style the message box

  /**
   * Handles password update submit.
   * - Prevents default form submission
   * - Validates inputs
   * - Calls backend endpoint /api/update-password
   * - Shows success/error feedback
   * - Clears fields and closes modal on success
   */
  const handleSubmit = async (e) => {
    // NOTE: This function is called from BOTH:
    // 1) <form onSubmit={handleSubmit}>  => receives a real event
    // 2) <button onClick={handleSubmit}> => receives a click event
    // Using e.preventDefault() works for both, but be careful if you ever call handleSubmit() without passing e.
    e.preventDefault();

    // Reset any previous message before validating
    setMessage('');
    setMessageType('');

    // ---------- Client-side validation ----------
    // Ensure all fields are filled
    if (!currentPassword || !newPassword || !confirmPassword) {
      setMessage('All fields are required');
      setMessageType('error');
      return;
    }

    // Ensure new password matches confirmation
    if (newPassword !== confirmPassword) {
      setMessage('New passwords do not match');
      setMessageType('error');
      return;
    }

    // Enforce minimal password length (weak but better than nothing)
    if (newPassword.length < 4) {
      setMessage('New password must be at least 4 characters');
      setMessageType('error');
      return;
    }

    // Prevent reusing the same password
    if (currentPassword === newPassword) {
      setMessage('New password must be different from current password');
      setMessageType('error');
      return;
    }

    // Start loading state (disable button, change label)
    setIsLoading(true);

    try {
      /**
       * POST request to backend to update the password.
       * Payload contains user identity + old/new password:
       * - departmentId: used to locate the user account
       * - userType: viewer/operator (if backend uses separate logic per type)
       * - currentPassword: to verify the user knows their current password
       * - newPassword: the desired new password
       */
      const response = await fetch('http://firehawk-project-production.up.railway.app', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json', // sending JSON body
        },
        body: JSON.stringify({
          departmentId: userData?.departmentId,
          userType: userData?.userType,
          currentPassword,
          newPassword,
        }),
      });

      // Parse JSON response from server
      const data = await response.json();

      // If backend returns { success: true, ... }
      if (data.success) {
        setMessage('Password updated successfully!');
        setMessageType('success');

        // Clear input fields after success
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');

        // Close modal after a short delay (lets user see the success message)
        setTimeout(() => {
          onClose?.(); // close modal (optional chaining for safety)

          // Notify parent component (if it wants to refresh UI/state)
          if (onPasswordUpdated) {
            onPasswordUpdated();
          }
        }, 2000);
      } else {
        // Backend responded but indicated failure (wrong current password, validation, etc.)
        setMessage(data.message || 'Failed to update password');
        setMessageType('error');
      }
    } catch (error) {
      // Network error / server down / CORS issue, etc.
      console.error('Error updating password:', error);
      setMessage('Error connecting to server');
      setMessageType('error');
    } finally {
      // Always stop loading state, even after errors
      setIsLoading(false);
    }
  };

  return (
    // Modal overlay: click outside closes modal
    <div className="modal-overlay" onClick={onClose}>
      {/* Stop click propagation so clicking inside the modal doesn't close it */}
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Modal header */}
        <div className="modal-header">
          <h2 style={{ color: '#fff', margin: 0 }}>Change Password</h2>
        </div>

        {/* Modal body */}
        <div className="modal-body" style={{ paddingTop: '20px' }}>
          {/* Form: pressing Enter inside inputs triggers onSubmit */}
          <form onSubmit={handleSubmit} className="manage-profile-form">
            {/* Message Display (shown only when message is non-empty) */}
            {message && (
              <div
                style={{
                  padding: '10px',
                  marginBottom: '15px',
                  borderRadius: '4px',
                  // Green for success, red for error
                  backgroundColor: messageType === 'success' ? '#d4edda' : '#f8d7da',
                  color: messageType === 'success' ? '#155724' : '#721c24',
                  border: `1px solid ${
                    messageType === 'success' ? '#c3e6cb' : '#f5c6cb'
                  }`,
                  fontSize: '14px',
                }}
              >
                {message}
              </div>
            )}

            {/* Current Password input */}
            <div style={{ marginBottom: '15px' }}>
              <label
                style={{
                  display: 'block',
                  color: '#fff',
                  marginBottom: '5px',
                  fontSize: '14px',
                  fontWeight: '500',
                }}
              >
                Current Password
              </label>
              <input
                type="password" // masks input characters
                placeholder="Enter current password"
                value={currentPassword} // controlled input
                onChange={(e) => setCurrentPassword(e.target.value)} // update state
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #555',
                  borderRadius: '4px',
                  backgroundColor: '#555',
                  color: '#fff',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              />
            </div>

            {/* New Password input */}
            <div style={{ marginBottom: '15px' }}>
              <label
                style={{
                  display: 'block',
                  color: '#fff',
                  marginBottom: '5px',
                  fontSize: '14px',
                  fontWeight: '500',
                }}
              >
                New Password
              </label>
              <input
                type="password"
                placeholder="Enter new password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #555',
                  borderRadius: '4px',
                  backgroundColor: '#555',
                  color: '#fff',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              />
            </div>

            {/* Confirm Password input */}
            <div style={{ marginBottom: '15px' }}>
              <label
                style={{
                  display: 'block',
                  color: '#fff',
                  marginBottom: '5px',
                  fontSize: '14px',
                  fontWeight: '500',
                }}
              >
                Confirm New Password
              </label>
              <input
                type="password"
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid #555',
                  borderRadius: '4px',
                  backgroundColor: '#555',
                  color: '#fff',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              />
            </div>

            {/* NOTE:
               There is no submit button inside the <form>.
               That’s okay because you call handleSubmit from the footer button,
               and Enter will still submit because onSubmit exists,
               but if you want “Enter” to always work reliably, add:
               <button type="submit" style={{ display: 'none' }} />
            */}
          </form>
        </div>

        {/* Modal footer buttons */}
        <div className="modal-footer">
          {/* Update Password button */}
          <button
            onClick={handleSubmit} // triggers the same submit logic
            disabled={isLoading} // prevents double submissions
            style={{
              backgroundColor: '#c41e3a',
              color: '#fff',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              opacity: isLoading ? 0.6 : 1,
              marginRight: '10px',
              fontSize: '14px',
              fontWeight: '500',
            }}
          >
            {isLoading ? 'Updating...' : 'Update Password'}
          </button>

          {/* Cancel button closes modal */}
          <button
            onClick={onClose}
            style={{
              backgroundColor: '#555',
              color: '#fff',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
