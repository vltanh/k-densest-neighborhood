export const fmtFloat = (x, digits = 6) => {
  if (x == null || Number.isNaN(x)) return null;
  const abs = Math.abs(x);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return x.toExponential(3);
  return x.toFixed(digits);
};

export const fmtInt = (x) => (x == null ? null : Number(x).toLocaleString());
