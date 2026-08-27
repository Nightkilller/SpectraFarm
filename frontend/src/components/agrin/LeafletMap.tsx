import { useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Circle,
  Marker,
  Popup,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export type MapLayer = "satellite" | "ndvi" | "sar" | "stress";

// Solid hex colors for Leaflet SVG rendering
const LAYER_CONFIG: Record<
  MapLayer,
  { stroke: string; fill: string; name: string }
> = {
  satellite: { stroke: "#10b981", fill: "#10b981", name: "Satellite AOI" },
  ndvi: { stroke: "#059669", fill: "#34d399", name: "NDVI Greenness Zone" },
  sar: { stroke: "#2563eb", fill: "#60a5fa", name: "SAR Moisture Zone" },
  stress: { stroke: "#dc2626", fill: "#f87171", name: "Stress Risk Zone" },
};

// Custom high-visibility SVG pin icon
function createCustomPin(color: string) {
  return L.divIcon({
    className: "custom-map-pin",
    html: `
      <div style="position: relative; width: 32px; height: 32px; transform: translate(-50%, -100%);">
        <div style="
          position: absolute;
          width: 32px;
          height: 32px;
          border-radius: 50%;
          background: ${color};
          opacity: 0.35;
          animation: map-ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;
        "></div>
        <div style="
          position: relative;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #ffffff;
          border: 3px solid ${color};
          border-radius: 50%;
          box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        ">
          <div style="width: 10px; height: 10px; border-radius: 50%; background: ${color};"></div>
        </div>
        <div style="
          position: absolute;
          bottom: -6px;
          left: 50%;
          transform: translateX(-50%);
          width: 0;
          height: 0;
          border-left: 6px solid transparent;
          border-right: 6px solid transparent;
          border-top: 8px solid ${color};
        "></div>
      </div>
      <style>
        @keyframes map-ping {
          75%, 100% {
            transform: scale(2.2);
            opacity: 0;
          }
        }
      </style>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 32],
  });
}

function MapController({
  lat,
  lng,
  onMapClick,
}: {
  lat: number;
  lng: number;
  onMapClick?: (coords: { lat: number; lng: number }) => void;
}) {
  const map = useMap();

  useEffect(() => {
    map.setView([lat, lng], map.getZoom() < 14 ? 15 : map.getZoom(), {
      animate: true,
    });
  }, [lat, lng, map]);

  useMapEvents({
    click(e) {
      if (onMapClick) {
        onMapClick({ lat: e.latlng.lat, lng: e.latlng.lng });
      }
    },
  });

  return null;
}

export default function LeafletMap({
  lat,
  lng,
  radius,
  layer,
  farmName = "Target AOI",
  cropName = "Field",
  onCoordsChange,
}: {
  lat: number;
  lng: number;
  radius: number;
  layer: MapLayer;
  farmName?: string;
  cropName?: string;
  onCoordsChange?: (coords: { lat: number; lng: number }) => void;
}) {
  const config = LAYER_CONFIG[layer] || LAYER_CONFIG.satellite;
  const pinIcon = createCustomPin(config.stroke);

  return (
    <MapContainer
      center={[lat, lng]}
      zoom={15}
      scrollWheelZoom
      zoomControl={true}
      className="h-full w-full cursor-crosshair"
    >
      {/* High Resolution Satellite Base Layer */}
      <TileLayer
        attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        maxZoom={19}
      />

      {/* Cartographic Reference Overlay (Roads & Labels) */}
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        maxZoom={19}
        opacity={0.7}
      />

      {/* Layer-specific thematic overlay */}
      {layer !== "satellite" && (
        <TileLayer
          opacity={0.3}
          url="https://tiles.stadiamaps.com/tiles/stamen_terrain_lines/{z}/{x}/{y}.png"
        />
      )}

      {/* AOI Outer Buffer Circle */}
      <Circle
        center={[lat, lng]}
        radius={radius}
        pathOptions={{
          color: config.stroke,
          weight: 3,
          dashArray: "6, 6",
          fillColor: config.fill,
          fillOpacity: layer === "satellite" ? 0.22 : 0.4,
        }}
      >
        <Tooltip permanent={false} direction="top">
          <div className="text-xs font-semibold">
            {farmName} — AOI ({radius}m radius)
          </div>
        </Tooltip>
      </Circle>

      {/* Center Target Precision Circle */}
      <Circle
        center={[lat, lng]}
        radius={Math.max(12, radius * 0.08)}
        pathOptions={{
          color: config.stroke,
          weight: 2,
          fillColor: config.fill,
          fillOpacity: 0.85,
        }}
      />

      {/* Pinpoint Target Marker */}
      <Marker position={[lat, lng]} icon={pinIcon}>
        <Popup>
          <div className="p-1 text-center">
            <p className="font-bold text-sm text-gray-900">{farmName}</p>
            <p className="text-xs text-gray-600 font-medium">Crop: {cropName}</p>
            <p className="text-[11px] font-mono text-emerald-600 mt-1">
              {lat.toFixed(5)}°N, {lng.toFixed(5)}°E
            </p>
            <p className="text-[10px] text-gray-400 mt-0.5">
              AOI Buffer: {radius}m
            </p>
          </div>
        </Popup>
      </Marker>

      <MapController lat={lat} lng={lng} onMapClick={onCoordsChange} />
    </MapContainer>
  );
}
