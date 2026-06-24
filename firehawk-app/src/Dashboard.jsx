// Dashboard.jsx

// React hooks:
// - useState: stores local component state (modal open, data arrays, loading flag)
// - useEffect: runs side-effects (fetching fires.json on mount)
// - useMemo: derives filtered/paginated views without recomputing on every render
import { useState, useEffect, useMemo } from 'react';

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

// Fix Leaflet's missing icon issue by setting default icon paths (module scope: run once)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({ iconRetinaUrl: markerIcon2x, iconUrl: markerIcon, shadowUrl: markerShadow });

// How many incidents to show per page (kept in the 10-15 range requested).
const PAGE_SIZE = 12;

/**
 * Parse a "DD/MM/YYYY, HH:MM" (or with '-') string into a sortable timestamp.
 * Returns 0 when it cannot be parsed so those rows sort to the end.
 */
function parseDateTime(value) {
  if (!value) return 0;
  const [datePart, timePart] = value.split(',').map((s) => s.trim());
  if (!datePart || !timePart) return 0;
  const [day, month, year] = datePart.replace(/-/g, '/').split('/').map(Number);
  const [hour, minute] = timePart.split(':').map(Number);
  const d = new Date(year, month - 1, day, hour, minute);
  return isNaN(d.getTime()) ? 0 : d.getTime();
}

/** Sort transformed rows newest -> oldest by lastlyUpdated. */
function sortByDateDesc(rows) {
  return [...rows].sort(
    (a, b) => parseDateTime(b.lastlyUpdated) - parseDateTime(a.lastlyUpdated)
  );
}

/**
 * Map an incident status string to Tailwind badge classes.
 * Em Curso / Em Resolução / Ativo / Chegada -> red (active)
 * Despacho -> amber, Vigilância -> blue, Conclusão -> gray.
 */
function statusBadgeClass(status) {
  const s = (status || '').toLowerCase();
  if (s.includes('curso') || s.includes('resolu') || s.includes('ativo') || s.includes('chegada'))
    return 'bg-red-100 text-red-700 ring-1 ring-inset ring-red-200';
  if (s.includes('despacho'))
    return 'bg-amber-100 text-amber-800 ring-1 ring-inset ring-amber-200';
  if (s.includes('vigil'))
    return 'bg-blue-100 text-blue-700 ring-1 ring-inset ring-blue-200';
  if (s.includes('conclus'))
    return 'bg-gray-100 text-gray-500 ring-1 ring-inset ring-gray-200';
  return 'bg-slate-100 text-slate-600 ring-1 ring-inset ring-slate-200';
}

export default function Dashboard({ userData, onLogout }) {
  // Router navigation function
  const navigate = useNavigate();

  // UI state: profile modal visibility
  const [showProfileModal, setShowProfileModal] = useState(false);

  // Full, date-sorted dataset (the table paginates/filters over this in memory).
  const [allRows, setAllRows] = useState([]);

  // Loading state for the initial fetch
  const [loading, setLoading] = useState(true);

  // Refreshing state (Update Incidents button)
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Listing controls (page state — React state only, not localStorage)
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('all');
  const [search, setSearch] = useState('');

  /**
   * Fetch the local JSON once on mount: transform -> sort by date desc ->
   * keep the FULL list in state (pagination happens in render).
   */
  useEffect(() => {
    fetch('/data/fires.json')
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response.json();
      })
      .then((data) => {
        setAllRows(sortByDateDesc(transformFireData(data)));
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error loading fire data:', error);
        setLoading(false);
      });
  }, []); // run once

  const handleLogout = () => {
    if (onLogout) onLogout();
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch(`${config.DATA_API}/api/refresh-data`, { method: 'POST' });
      const result = await response.json();

      if (result.success) {
        // Reload the local fires.json (timestamp busts the browser cache)
        const res = await fetch(`/data/fires.json?t=${Date.now()}`);
        const data = await res.json();
        setAllRows(sortByDateDesc(transformFireData(data)));
        setPage(1); // back to the first page after a refresh
      } else {
        alert('Server Error: ' + result.error);
      }
    } catch (error) {
      console.error(error);
      alert('Error connecting to server.');
    } finally {
      setIsRefreshing(false);
    }
  };

  // Distinct statuses present in the data, for the filter dropdown.
  const statusOptions = useMemo(() => {
    const set = new Set();
    allRows.forEach((r) => r.status && set.add(r.status));
    return Array.from(set).sort();
  }, [allRows]);

  // Apply status filter + free-text location search.
  const filteredRows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return allRows.filter((r) => {
      const matchesStatus = statusFilter === 'all' || r.status === statusFilter;
      const matchesSearch = !q || (r.location || '').toLowerCase().includes(q);
      return matchesStatus && matchesSearch;
    });
  }, [allRows, statusFilter, search]);

  // Pagination math (clamp page into range so filters can't strand it).
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const startIdx = (safePage - 1) * PAGE_SIZE;
  const pageRows = filteredRows.slice(startIdx, startIdx + PAGE_SIZE);

  // Markers reflect the CURRENT PAGE (keeps the map in sync with the table you see),
  // excluding completed incidents and rows without valid coordinates.
  const markerRows = pageRows.filter(
    (row) =>
      row.originalData &&
      typeof row.originalData.lat === 'number' &&
      typeof row.originalData.lng === 'number' &&
      row.status &&
      row.status.toLowerCase() !== 'conclusão'
  );

  // While data is loading, render a simple placeholder
  if (loading) {
    return <div className="dashboard-container">Loading data...</div>;
  }

  const rangeFrom = filteredRows.length === 0 ? 0 : startIdx + 1;
  const rangeTo = startIdx + pageRows.length;

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
            onClick={() => setShowProfileModal(true)}
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
            className="inline-flex items-center gap-2 rounded-lg border border-red-200 bg-white px-3.5 py-2 text-sm font-semibold text-red-600 shadow-sm transition hover:bg-red-50 hover:border-red-300 focus:outline-none focus:ring-2 focus:ring-red-300 disabled:cursor-wait disabled:opacity-60"
          >
            <span className={isRefreshing ? 'animate-spin' : ''} aria-hidden="true">
              {isRefreshing ? '⟳' : '⚡'}
            </span>
            {isRefreshing ? 'Refreshing…' : 'Update Incidents'}
          </button>
        </div>
      </header>

      {/* Main content layout: table (left) + map (right) */}
      <div className="dashboard-content">
        {/* Left side: paginated, filterable incident list */}
        <div className="table-section !p-0 !overflow-hidden flex flex-col">
          {/* Toolbar: search + status filter + result count */}
          <div className="flex flex-wrap items-center gap-3 border-b border-gray-200 bg-white/60 px-4 py-3">
            <div className="relative flex-1 min-w-[180px]">
              <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-gray-400" aria-hidden="true">⌕</span>
              <input
                type="text"
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                placeholder="Search location / district…"
                className="w-full rounded-lg border border-gray-300 bg-white py-2 pl-9 pr-3 text-sm text-gray-700 placeholder-gray-400 shadow-sm focus:border-red-300 focus:outline-none focus:ring-2 focus:ring-red-200"
              />
            </div>
            <select
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
              className="rounded-lg border border-gray-300 bg-white py-2 pl-3 pr-8 text-sm text-gray-700 shadow-sm focus:border-red-300 focus:outline-none focus:ring-2 focus:ring-red-200"
              aria-label="Filter by status"
            >
              <option value="all">All statuses</option>
              {statusOptions.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <span className="ml-auto text-xs font-medium text-gray-500">
              {filteredRows.length} incident{filteredRows.length === 1 ? '' : 's'}
            </span>
          </div>

          {/* Scrollable table with a sticky header */}
          <div className="flex-1 overflow-y-auto">
            <table className="w-full border-collapse text-sm">
              <thead className="sticky top-0 z-10">
                <tr className="bg-gray-50 text-left text-xs font-semibold uppercase tracking-wide text-gray-500 shadow-[inset_0_-1px_0_0_#e5e7eb]">
                  <th className="px-4 py-3">ID</th>
                  <th className="px-4 py-3">Lastly updated</th>
                  <th className="px-4 py-3">Location</th>
                  <th className="px-4 py-3">Units</th>
                  <th className="px-4 py-3">Level</th>
                  <th className="px-4 py-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row) => (
                  <tr
                    key={row.id}
                    onClick={() => navigate(`/alert/${row.id}`)}
                    className="cursor-pointer border-b border-gray-100 odd:bg-white even:bg-gray-50/60 transition-colors hover:bg-red-50/70"
                  >
                    <td className="px-4 py-3 font-mono text-xs text-gray-400">{row.id}</td>
                    <td className="px-4 py-3 whitespace-nowrap text-gray-600">{row.lastlyUpdated}</td>
                    <td className="px-4 py-3 font-medium text-gray-800">{row.location}</td>
                    <td className="px-4 py-3 whitespace-nowrap text-gray-600">{row.units}</td>
                    <td className="px-4 py-3 text-gray-600">{row.level}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${statusBadgeClass(row.status)}`}>
                        {row.status}
                      </span>
                    </td>
                  </tr>
                ))}

                {pageRows.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-10 text-center text-sm text-gray-400">
                      No incidents match the current filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination footer */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-gray-200 bg-white/60 px-4 py-3">
            <span className="text-xs text-gray-500">
              {rangeFrom}–{rangeTo} of {filteredRows.length}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={safePage <= 1}
                className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
              >
                ‹ Previous
              </button>
              <span className="px-1 text-sm font-medium text-gray-600">
                Page {safePage} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={safePage >= totalPages}
                className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next ›
              </button>
            </div>
          </div>
        </div>

        {/* Right side: map showing markers for the current page */}
        <div className="map-section">
          <MapContainer
            center={[39.5, -8.0]}
            zoom={7}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />

            {markerRows.map((row) => (
              <Marker
                key={row.originalId}
                position={[row.originalData.lat, row.originalData.lng]}
              >
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
          onClose={() => setShowProfileModal(false)}
          onLogout={handleLogout}
        />
      )}

      {/* Footer: shows current mode based on userType */}
      <footer className="dashboard-footer">
        <p>Mode : {userData?.userType === 'viewer' ? 'Viewer' : 'Operator'}</p>
      </footer>
    </div>
  );
}
