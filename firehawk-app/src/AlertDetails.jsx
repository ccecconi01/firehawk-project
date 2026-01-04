//importing libs , directories and functions
import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import jsPDF from 'jspdf';

import { transformFireData } from './dataTransforms';
import ProfileModal from './ProfileModal';
import './AlertDetails.css';

export default function AlertDetails({ userData, onLogout }) {
  const { alertId } = useParams();
  const navigate = useNavigate();

  const [showProfileModal, setShowProfileModal] = useState(false);
  const [alertData, setAlertData] = useState(null);
  const [loading, setLoading] = useState(true);
  // Removed horizon state as RF predictions are static, not time-dependent
  // const [horizon, setHorizon] = useState('30m'); 

  const normalizedType = (userData?.userType || '').toLowerCase();
  const isOperator = normalizedType === 'operator';

  // Load fires.json and pick the alert
  useEffect(() => {
    setLoading(true);

    fetch('/data/fires.json')
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        const transformed = transformFireData(data);

        const selectedAlert = transformed.find(
          (a) => a.id === parseInt(alertId, 10)
        );

        if (selectedAlert) {
          setAlertData(selectedAlert);
        } else if (transformed.length > 0) {
          // Fallback to the first alert if the ID is not found (for development/testing)
          setAlertData(transformed[0]);
        } else {
          setAlertData(null);
        }
      })
      .catch((err) => {
        console.error('Error loading fire data:', err);
        setAlertData(null);
      })
      .finally(() => setLoading(false));
  }, [alertId]);

  const handleLogout = () => onLogout?.();
  console.log("DADOS ORIGINAIS:", alertData?.originalData);

  if (loading) {
    return (
      <div className="alert-details-container">
        <div className="loading-message">Loading alert details...</div>
      </div>
    );
  }

  if (!alertData) {
    return (
      <div className="alert-details-container">
        <div className="error-message">Alert not found</div>
        <button onClick={() => navigate('/dashboard')}>Back to Dashboard</button>
      </div>
    );
  }

  const handleBackToDashboard = () => navigate('/dashboard');

  // Coordinates (with default fallback)
  const lat = Number(alertData.originalData?.lat ?? alertData.originalData?.LAT ?? 39.5);
  const lng = Number(alertData.originalData?.lng ?? alertData.originalData?.LON ?? -8.0);

  const fireInfo = {
    altitude: alertData.originalData?.icnf?.altitude 
      ? `${Number(alertData.originalData.icnf.altitude).toFixed(2)} m` 
      : (alertData.originalData?.ALTITUDEMEDIA ? `${Number(alertData.originalData.ALTITUDEMEDIA).toFixed(2)} m` : 
        (alertData.originalData?.altitude ? `${Number(alertData.originalData.altitude).toFixed(2)} m` : 'N/A')),

    fireType: alertData.originalData?.natureza || alertData.originalData?.Natureza || 'N/A',
    
    region: alertData.location || alertData.originalData?.DISTRITO || 'N/A',
    
    // WEATHER: try both new and old field names
    temperature: (alertData.originalData?.temp ?? alertData.originalData?.TEMPERATURA) != null 
      ? `${Number(alertData.originalData?.temp ?? alertData.originalData?.TEMPERATURA).toFixed(1)} °C` : 'N/A',

    relativeHumidity: (alertData.originalData?.humidade ?? alertData.originalData?.HUMIDADERELATIVA) != null 
      ? `${Number(alertData.originalData?.humidade ?? alertData.originalData?.HUMIDADERELATIVA).toFixed(1)}%` : 'N/A',

    windSpeed: (alertData.originalData?.vento ?? alertData.originalData?.VENTOINTENSIDADE) != null 
      ? `${Number(alertData.originalData?.vento ?? alertData.originalData?.VENTOINTENSIDADE).toFixed(1)} km/h` : 'N/A',

    // new weather data from pipeline
    pressure: alertData.originalData?.pressao != null 
      ? `${Number(alertData.originalData.pressao).toFixed(1)} hPa` : 'N/A',

    rain: alertData.originalData?.chuva_24h != null 
      ? `${Number(alertData.originalData.chuva_24h).toFixed(1)} mm` : '0 mm',

    windDirection: alertData.originalData?.direcao_vento != null 
      ? `${Number(alertData.originalData.direcao_vento).toFixed(0)}°` : 'N/A',

    // FWI 
    fwi: (alertData.originalData?.fwi ?? alertData.originalData?.FWI) != null 
       ? Number(alertData.originalData?.fwi ?? alertData.originalData?.FWI).toFixed(1) : 'N/A',
       
    isi: (alertData.originalData?.isi ?? alertData.originalData?.ISI) != null 
       ? Number(alertData.originalData?.isi ?? alertData.originalData?.ISI).toFixed(1) : 'N/A',
    
    // VPD_kPa
    vpd: alertData.originalData?.VPD_kPa != null 
      ? `${Number(alertData.originalData.VPD_kPa).toFixed(2)} kPa` : 'N/A',
  };

  // Real-time resources (Actual values)
  const realTimeResources = {
    firefighters: alertData.originalData?.Real_Homens || alertData.originalData?.Real_Operacionais_Man || alertData.originalData?.Operacionais_Man || '0',
    vehicles: alertData.originalData?.Real_Terrestres || alertData.originalData?.Real_Meios_Terrestres || alertData.originalData?.Meios_Terrestres || '0',
    helicopters: alertData.originalData?.Real_Aereos || alertData.originalData?.Real_Meios_Aereos || alertData.originalData?.Meios_Aereos || '0',
  };

  // Predicted resources (ML Model output)
  // Try reading transformed data name or the one directly from python
  const predictedResources = {
    firefighters: { predicted: alertData.originalData?.Previsto_Operacionais_Man || alertData.originalData?.Prev_Homens || 'N/A' },
    vehicles: { predicted: alertData.originalData?.Previsto_Meios_Terrestres || alertData.originalData?.Prev_Terrestres || 'N/A' },
    helicopters: { predicted: alertData.originalData?.Previsto_Meios_Aereos || alertData.originalData?.Prev_Aereos || 'N/A' },
  };

  // --------- PDF GENERATION ----------
  const handleGenerateReport = () => {
    if (!alertData) return;

    const doc = new jsPDF();
    let y = 15;
    const lineGap = 7;

    const addLine = (text) => {
      doc.text(text, 10, y);
      y += lineGap;
    };

    // Title
    doc.setFontSize(18);
    addLine(`FireHawk - Alert Report`);
    doc.setFontSize(11);
    addLine(`Generated: ${new Date().toLocaleString()}`);
    y += 3;

    // Alert info
    doc.setFontSize(13);
    addLine(`Alert Information`);
    doc.setFontSize(11);
    addLine(`Alert ID: ${alertData.id}`);
    addLine(`Original ID: ${alertData.originalId}`);
    addLine(`Status: ${alertData.status}`);
    addLine(`Location (district): ${alertData.location}`);
    addLine(`Last updated: ${alertData.lastlyUpdated}`);
    addLine(`Coordinates: ${lat.toFixed(4)}, ${lng.toFixed(4)}`);
    y += 3;

    // Fire info
    doc.setFontSize(13);
    addLine(`Fire Details`);
    doc.setFontSize(11);
    addLine(`Fire type: ${fireInfo.fireType}`);
    addLine(`Region: ${fireInfo.region}`);
    addLine(`Altitude: ${fireInfo.altitude}`);
    y += 3;

    // Weather info
    doc.setFontSize(13);
    addLine(`Weather Information`);
    doc.setFontSize(11);
    addLine(`Temperature: ${fireInfo.temperature}`);
    addLine(`Pressure: ${fireInfo.pressure}`);
    addLine(`Relative humidity: ${fireInfo.relativeHumidity}`);
    addLine(`Wind Speed: ${fireInfo.windSpeed} (Dir: ${fireInfo.windDirection})`);
    addLine(`Rain (24h): ${fireInfo.rain}`); 
    addLine(`FWI: ${fireInfo.fwi}`);
    addLine(`ISI: ${fireInfo.isi}`);
    addLine(`VPD: ${fireInfo.vpd}`);
    y += 3;

    // Real-time resources
    doc.setFontSize(13);
    addLine(`Real-time Resources`);
    doc.setFontSize(11);
    addLine(`Firefighters: ${realTimeResources.firefighters}`);
    addLine(`Vehicles: ${realTimeResources.vehicles}`);
    addLine(`Aerial: ${realTimeResources.helicopters}`);
    y += 3;

    // Predicted resources (RF Model)
    doc.setFontSize(13);
    addLine(`Predicted Resources`);
    doc.setFontSize(11);
    addLine(`Firefighters: Predicted=${predictedResources.firefighters.predicted}`);
    addLine(`Vehicles: Predicted=${predictedResources.vehicles.predicted}`);
    addLine(`Aerial: Predicted=${predictedResources.helicopters.predicted}`);
    y += 3;

    // Save file
    const fileName = `FireHawk_Alert_${alertData.id}.pdf`;
    doc.save(fileName);
  };

  return (
    <div className="alert-details-container">
      {/* Header */}
      <header className="alert-header">
        <div className="header-left">
          <button className="back-button" onClick={handleBackToDashboard}>
            <div className="logo">
              <img
                src="/logo/firehawklogo.png"
                alt="Firehawk logo"
                className="h-16 sm:h-20 md:h-20 w-auto select-none"
                loading="eager"
                draggable="false"
              />
            </div>
          </button>
        </div>

        <div className="header-right">
          <button
            className="profile-button"
            onClick={() => setShowProfileModal(true)}
            title="Profile"
          >
            <img
              src="/img/user.png"
              alt="Profile"
              className="h-10 w-10 object-cover"
              loading="eager"
              draggable="false"
            />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="alert-content">
        {/* Weather/Fire Information Panel */}
        <div className="weather-section">
          <h2>Weather Information</h2>
          <div className="weather-info">
            <p>Temperature: {fireInfo.temperature}</p>
            <p>Pressure : {fireInfo.pressure}</p>
            <p>Rain (24h): {fireInfo.rain}</p>
            <p>Wind Direction: {fireInfo.windDirection}</p>
            <p>Wind Speed: {fireInfo.windSpeed}</p>
            <p>Relative Humidity: {fireInfo.relativeHumidity}</p>
            <p>FWI: {fireInfo.fwi}</p>
            <p>ISI: {fireInfo.isi}</p>
            <p>VPD: {fireInfo.vpd}</p>
                
            <hr style={{ margin: '10px 0', borderColor: '#ccc' }} />
            <p><strong>Nature:</strong> {fireInfo.fireType}</p>
            <p><strong>Altitude:</strong> {fireInfo.altitude}</p>
            <p><strong>Region:</strong> {fireInfo.region}</p>
          </div>
        </div>

        {/* Map */}
        <div className="map-section">
          <MapContainer
            center={[lat, lng]}
            zoom={13}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            <Marker position={[lat, lng]}>
              <Popup>
                {alertData.location}
                <br />
                {lat.toFixed(4)}, {lng.toFixed(4)}
              </Popup>
            </Marker>
          </MapContainer>
        </div>

        {/* Right side info */}
        <div className="alert-info-section">
          <div className="alert-header-info">
            <h3>Alert ID: {alertData.id}</h3>
            <p>Location: {alertData.location}</p>
            <p>Status: {alertData.status}</p>
          </div>

          {isOperator && (
            <button className="generate-report-btn" onClick={handleGenerateReport}>
              Generate Report
            </button>
          )}

          {/* Real-Time Resources (Actual) */}
          <div className="resources-box">
            <h4>Real Time Resources </h4>
            <p>Firefighters: {realTimeResources.firefighters}</p>
            <p>Vehicles: {realTimeResources.vehicles}</p>
            <p>Helicopters: {realTimeResources.helicopters}</p>
          </div>

          {/* Predicted Resources (RF Model) */}
          <div className="resources-box">
            <h4>Predicted Resources </h4>
            <p>
              Firefighters: {predictedResources.firefighters.predicted} 
            </p>
            <p>
              Vehicles: {predictedResources.vehicles.predicted} 
            </p>
            <p>
              Helicopters: {predictedResources.helicopters.predicted} 
            </p>
            {/* Removed time buttons as predictions are static */}
          </div>
        </div>
      </div>

      {/* Profile Modal */}
      {showProfileModal && (
        <ProfileModal
          userData={userData}
          onClose={() => setShowProfileModal(false)}
          onLogout={handleLogout}
        />
      )}

      <footer className="alert-footer">
        <p>Mode - {normalizedType === 'viewer' ? 'Viewer' : 'Operator'}</p>
      </footer>
    </div>
  );
}