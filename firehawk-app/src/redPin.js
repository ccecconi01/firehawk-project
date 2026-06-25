// redPin.js
// Shared Leaflet marker icon in the Firehawk brand red (#c41e3a — the crimson used
// by the logo title, logout and "Generate Report" buttons). Replaces Leaflet's
// default blue marker. Used by both the Dashboard map and the AlertDetails map.
import L from 'leaflet';

export const BRAND_RED = '#c41e3a';

export const redPinIcon = L.divIcon({
  className: 'firehawk-red-pin', // no default Leaflet styling
  html: `
    <svg width="26" height="38" viewBox="0 0 26 38" xmlns="http://www.w3.org/2000/svg">
      <path d="M13 0C5.82 0 0 5.82 0 13c0 9.6 11 23.2 12.2 24.8a1 1 0 0 0 1.6 0C15 36.2 26 22.6 26 13 26 5.82 20.18 0 13 0z"
            fill="${BRAND_RED}" stroke="#7f1322" stroke-width="1"/>
      <circle cx="13" cy="13" r="4.5" fill="#ffffff" fill-opacity="0.95"/>
    </svg>`,
  iconSize: [26, 38],
  iconAnchor: [13, 38],   // pin tip sits on the coordinate
  popupAnchor: [0, -34],
});
