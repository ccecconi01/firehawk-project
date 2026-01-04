// Dashboard.jsx

// React hooks:
// - useState: stores local component state (modal open, data arrays, loading flag)
// - useEffect: runs side-effects (fetching fires.json on mount)
import { useState, useEffect } from 'react';

// Config with API URLs
import config from './config';

// React Router:
// - useNavigate: programmatic navigation when clicking a row (go to /alert/:id)
import { useNavigate } from 'react-router-dom';

// React-Leaflet components for the map UI
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet'; import 'leaflet/dist/leaflet.css'; import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'; import markerIcon from 'leaflet/dist/images/marker-icon.png'; import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// Local components + styling
import ProfileModal from './ProfileModal';
import './Dashboard.css';

// Helper that normalizes raw JSON into the shape your UI expects
import { transformFireData } from './dataTransforms';

export default function Dashboard({ userData, onLogout }) {
  // Router navigation function
  const navigate = useNavigate();

  // UI state: profile modal visibility
  const [showProfileModal, setShowProfileModal] = useState(false);

  // Data shown in the table (you fill it with the latest 30)
  const [tableData, setTableData] = useState([]);

  // Full dataset placeholder (currently never set in your effect)
  // Note: as written, allFires stays [], so ongoingFires will always be []
  const [allFires, setAllFires] = useState([]);

  // Loading state for the initial fetch
  const [loading, setLoading] = useState(true);

  // Refreshing state for future use (update incidents button)
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Fix Leaflet's missing icon issue by setting default icon paths
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({ iconRetinaUrl: markerIcon2x, iconUrl: markerIcon, shadowUrl: markerShadow, });


  /**
   * Fetch the local JSON file once when the component mounts.
   * Steps:
   * 1) fetch /data/fires.json
   * 2) transform raw data into UI-friendly objects
   * 3) sort by "lastlyUpdated" (newest first)
   * 4) keep the latest 20
   */
  useEffect(() => {
    fetch('/data/fires.json')
      .then((response) => {
        // If the server returns 404/500/etc, throw to go into .catch
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        // Parse JSON body
        return response.json();
      })
      .then((data) => {
        // Convert raw JSON into your app's normalized rows
        const transformed = transformFireData(data);

        /**
         * Parse your "lastlyUpdated" string into a timestamp number
         * so we can reliably sort it.
         *
         * Expected format: "DD/MM/YYYY, HH:MM"
         * Returns: milliseconds since epoch (number)
         * If parsing fails, returns 0 so it sorts to the end.
         */
        /**
         * Parse your "lastlyUpdated" string into a timestamp number.
         * Handles both "DD/MM/YYYY" and "DD-MM-YYYY".
         */
        const parseDateTime = (value) => {
          if (!value) return 0;

          // Split into date and time (e.g. "12/12/2025, 14:30")
          const [datePart, timePart] = value.split(',').map((s) => s.trim());
          if (!datePart || !timePart) return 0;

          // Normalize separators (replace - with /)
          const normalizedDate = datePart.replace(/-/g, '/');
          
          // Now the split '/' always works
          const [day, month, year] = normalizedDate.split('/').map(Number);
          const [hour, minute] = timePart.split(':').map(Number);

          // Build JS Date
          const dateObj = new Date(year, month - 1, day, hour, minute);
          
          // If the date is invalid, return 0 to sort to the end of the list
          return isNaN(dateObj.getTime()) ? 0 : dateObj.getTime();
        };

        // Sort newest -> oldest by "lastlyUpdated"
        const sorted = [...transformed].sort(
          (a, b) => parseDateTime(b.lastlyUpdated) - parseDateTime(a.lastlyUpdated)
        );

        // Keep the latest 30 rows for the table + map markers
        const latest20 = sorted.slice(0, 20);

        // Update UI state
        setTableData(latest20);

        // Optional: if you want ongoingFires to work, you likely meant:
        // setAllFires(sorted);
        setLoading(false);
      })
      .catch((error) => {
        // Any network/JSON/transform errors end up here
        console.error('Error loading fire data:', error);
        setLoading(false);
      });
  }, []); // [] = run once (on mount)

  /**
   * Logout handler:
   * delegates to parent if provided (keeps this component reusable)
   */
  const handleLogout = () => {
    if (onLogout) {
      onLogout();
    }
  };

  // While data is loading, render a simple placeholder
  if (loading) {
    return <div className="dashboard-container">Loading data...</div>;
  }

  /**
   * Computes ongoing fires from allFires.
   * As written, allFires is never set, so this always returns [].
   * Logic:
   * - must have coordinates
   * - "ongoing" if:
   *    fire.active === true
   *    OR status exists and is not "Conclusão"
   */
  const ongoingFires = allFires.filter((row) => {
    const fire = row.originalData;
    if (!fire) return false;

    const hasCoords =
      typeof fire.lat === 'number' && typeof fire.lng === 'number';

    const isOngoing =
      fire.active === true || (row.status && row.status !== 'Conclusão');

    return hasCoords && isOngoing;
  });

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      
      const response = await fetch(`${config.DATA_API}/api/refresh-data`, {
        method: 'POST',
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Reload the local fires.json file
        //Timestamp forces browser to not use cached version
        fetch(`/data/fires.json?t=${Date.now()}`)
          .then(res => res.json())
          .then(data => {
             const transformed = transformFireData(data);
             
             setTableData(transformed.slice(0, 20));
             alert("Incident data updated!");
          });
      } else {
        alert("Server Error: " + result.error);
      }
    } catch (error) {
      console.error(error);
      alert("Error connecting to server.");
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <div className="dashboard-container">
      {/* Header area: brand + profile button */}
      <header className="dashboard-header">
        {/* Centered logo */}
        <div className="flex justify-center mb-8">
          <img
            src="/logo/firehawklogo.png"
            alt="Firehawk logo"
            className="h-16 sm:h-20 md:h-20 w-auto select-none"
            loading="eager"
            draggable="false"
          />
        </div>

        {/* Profile button on the right */}
        <div className="header-right">
          <button
            className="profile-button"
            onClick={() => setShowProfileModal(true)} // open modal
            title="Profile"
            aria-label="Open profile"
          >
            <img
              src="/img/user.png"
              alt="Profile"
              className="h-10 w-10 object-cover"
              loading="eager"
              draggable="false"
            />
          </button>
          <button 
            onClick={handleRefresh} 
            disabled={isRefreshing}
            className="refresh-button"
            style={{
              padding: '8px 16px',
              backgroundColor: isRefreshing ? '#ccc' : '#e63946',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: isRefreshing ? 'wait' : 'pointer',
              fontWeight: 'bold',
              display: 'flex', 
              alignItems: 'center', 
              gap: '5px'
            }}
          >
            {isRefreshing ? '🔄 Refreshing...' : '⚡ Update Incidents'}
          </button>
        </div>
      </header>

      {/* Main content layout: table (left) + map (right) */}
      <div className="dashboard-content">
        {/* Left side: table listing latest alerts */}
        <div className="table-section">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Lastly updated</th>
                <th>Location</th>
                <th>Units</th>
                <th>Level</th>
                <th>Status</th>
              </tr>
            </thead>

            <tbody>
              {tableData.map((row) => (
                <tr
                  key={row.id} // stable key for React list rendering
                  onClick={() => navigate(`/alert/${row.id}`)} // go to alert details page
                  className="clickable-row"
                >
                  <td>{row.id}</td>
                  <td>{row.lastlyUpdated}</td>
                  <td>{row.location}</td>
                  <td>{row.units}</td>
                  <td>{row.level}</td>
                  <td>{row.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Right side: map showing markers for fires with coordinates */}
        <div className="map-section">
          <MapContainer
            // Default view centered roughly on Portugal
            center={[39.5, -8.0]}
            zoom={7}
            style={{ height: '100%', width: '100%' }}
          >
            {/* Base map tiles (OpenStreetMap) */}
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />

            {/**
             * Create markers from tableData:
             * - must have originalData with lat/lng numbers
             * - must have a status
             * - intended: exclude completed ("Conclusão")
             *
             *  FIX: Changed 'Conclusão' to 'conclusão' in toLowerCase() comparison.
             */}
            {tableData
              .filter(
                (row) =>
                  row.originalData &&
                  typeof row.originalData.lat === 'number' &&
                  typeof row.originalData.lng === 'number' &&
                  row.status &&
                  row.status.toLowerCase() !== 'conclusão'
              )
              .map((row) => (
                <Marker
                  key={row.originalId} // key for markers list
                  position={[row.originalData.lat, row.originalData.lng]} // marker position
                >
                  {/* Popup shown when marker is clicked */}
                  <Popup>
                    <strong>ID:</strong> {row.originalId}
                    <br />
                    <strong>Updated:</strong> {row.lastlyUpdated}
                    <br />
                    <strong>Location:</strong> {row.location}
                    <br />
                    <strong>Units:</strong> {row.units}
                    <br />
                    <strong>Status:</strong> {row.status}
                  </Popup>
                </Marker>
              ))}
          </MapContainer>
        </div>
      </div>

      {/* Profile modal: shown only when showProfileModal is true */}
      {showProfileModal && (
        <ProfileModal
          userData={userData}
          userType={userData?.userType}
          onClose={() => setShowProfileModal(false)} // close modal
          onLogout={handleLogout} // pass logout handler down
        />
      )}

      {/* Footer: shows current mode based on userType */}
      <footer className="dashboard-footer">
        <p>Mode : {userData?.userType === 'viewer' ? 'Viewer' : 'Operator'}</p>
      </footer>
    </div>
  );
}