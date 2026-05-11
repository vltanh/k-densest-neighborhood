const defaultApiBase = () => {
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL;
  if (typeof window !== 'undefined' && window.location?.hostname) {
    const host = window.location.hostname;
    // When served from LAN IP (phone hitting dev machine), target same host on 8000.
    // Localhost stays localhost.
    return `${window.location.protocol}//${host}:8000`;
  }
  return 'http://127.0.0.1:8000';
};

export const API_BASE_URL = defaultApiBase();

export const ORACLE_OPENALEX = 'openalex';
export const ORACLE_SIM = 'sim';

export const ORACLE_MODES = [
  { value: ORACLE_OPENALEX, label: 'OpenAlex' },
  { value: ORACLE_SIM, label: 'Simulation' },
];

export const SIM_DATASETS = ['Cora', 'PubMed', 'DBLP'];

// Curated palette for the common small-class case (<= 10).
export const CLASS_PALETTE = [
  '#3A7CE3', // azure
  '#E8A33A', // amber
  '#C0392B', // vermillion
  '#2CA58D', // teal
  '#8E6FCF', // violet
  '#E07A5F', // terracotta
  '#5FAD56', // sage
  '#D4A373', // ochre
  '#4C6EF5', // indigo
  '#B5838D', // mauve
];

// HSL rainbow generator — stable per label, distinct enough up to ~70 classes.
// Uses golden-angle hue stepping so consecutive labels stay far apart.
const GOLDEN_ANGLE = 137.50776405003785;
export const classColor = (label, totalClasses = null) => {
  if (label == null || label < 0) return '#8297B2';
  if (totalClasses && totalClasses <= CLASS_PALETTE.length) {
    return CLASS_PALETTE[label % CLASS_PALETTE.length];
  }
  const hue = (label * GOLDEN_ANGLE) % 360;
  // Alternate saturation/lightness bands to maximize local contrast.
  const band = label % 3;
  const sat = band === 0 ? 72 : band === 1 ? 58 : 84;
  const light = band === 0 ? 55 : band === 1 ? 48 : 62;
  return `hsl(${hue.toFixed(1)}, ${sat}%, ${light}%)`;
};
