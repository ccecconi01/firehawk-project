import { useState } from 'react';
import ManageProfileModal from './ManageProfileModal';

export default function ProfileModal({ userData, onClose, onLogout }) {
  const [showManageProfile, setShowManageProfile] = useState(false);

  // Normalize from userData (handles "Viewer", "viewer", "OPERATOR", etc.)
  const normalizedType = (userData?.userType || '').toLowerCase();
  const displayUserType = normalizedType === 'viewer' ? 'Viewer' : 'Operator';

  // Department ID from userData
  const departmentId = userData?.departmentId || 'N/A';

  const handleManageProfileClick = () => {
    setShowManageProfile(true);
  };

  const handleManageProfileClose = () => {
    setShowManageProfile(false);
  };

  const handlePasswordUpdated = () => {
    // Optional: You can add logic here after password is updated
    console.log('Password updated successfully');
  };

  if (showManageProfile) {
    return (
      <ManageProfileModal
        userData={userData}
        onClose={handleManageProfileClose}
        onPasswordUpdated={handlePasswordUpdated}
      />
    );
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="user-avatar">
            <img
              src="/img/user.png"
              alt="Profile"
              className="h-10 w-10 object-cover"
              loading="eager"
              draggable="false"
            />
          </div>
        </div>

        <div className="modal-body">
          <div className="info-row">
            <span className="info-label">Department ID</span>
            <span className="info-value">{departmentId}</span>
          </div>

          <div className="info-row">
            <span className="info-label">User type</span>
            <span className="info-value">{displayUserType}</span>
          </div>
        </div>

        <div className="modal-footer">
          {/* Only show for operators */}
          {normalizedType === 'operator' && (
            <button
              className="manage-profile-btn"
              onClick={handleManageProfileClick}
            >
              Manage Profile
            </button>
          )}
          <button className="logout-btn" onClick={onLogout}>
            Log out
          </button>
        </div>
      </div>
    </div>
  );
}