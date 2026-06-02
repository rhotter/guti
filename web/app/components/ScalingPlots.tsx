"use client";

import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts";

// ── types ────────────────────────────────────────────────────────────────────

interface Variant {
  hash: string;
  num_sensors: number | null;
  source_spacing_mm: number | null;
  grid_resolution_mm: number | null;
  psf_fwhm_mm: number | null;
  bold_contrast: number | null;
  bold_snr: number | null;
  frequency_hz: number | null;
  n_singular_values: number;
  sv_indices: number[];
  singular_values: number[];
  first_sv: number;
  bitrate_today: number | null;
  bitrate_fundamental: number | null;
  bitrate_physical_today: number | null;
  bitrate_physical_fundamental: number | null;
  bitrate_empirical_today: number | null;
  bitrate_empirical_fundamental: number | null;
  bitrate_anchored_today: number | null;
  bitrate_anchored_fundamental: number | null;
  snr_empirical_today: number | null;
}

interface ModalityData {
  modality: string;
  label: string;
  sweep_params: string[];
  default_bitrate_mode?: NoiseMode;
  bitrate_modes?: Record<string, { label: string; description: string }>;
  noise_label_today: string;
  noise_label_fundamental: string;
  source_amplitude: number;
  source_amplitude_units: string;
  typical_signal: number;
  variants: Variant[];
}

type NoiseMode = "physical_detector_floor" | "empirical_observed_snr" | "empirical_anchored";

// ── constants ────────────────────────────────────────────────────────────────

const MODALITIES = [
  { key: "meg_opm", label: "MEG OPM" },
  { key: "meg_squid", label: "MEG SQUID" },
  { key: "eeg_openmeeg", label: "EEG" },
  { key: "cw_fnirs", label: "fNIRS CW" },
  { key: "fmri_bold", label: "fMRI" },
  { key: "us_free_field_analytical_frequency_sweep", label: "Ultrasound" },
];

const PARAM_LABELS: Record<string, string> = {
  num_sensors: "# sensors",
  source_spacing_mm: "source spacing (mm)",
  grid_resolution_mm: "voxel size (mm)",
  psf_fwhm_mm: "PSF FWHM (mm)",
  bold_snr: "BOLD response SNR",
  frequency_hz: "frequency (Hz)",
};

// viridis-ish palette
const COLORS = [
  "#440154", "#3b528b", "#21908d", "#5dc963", "#fde725",
  "#31688e", "#35b779", "#90d743", "#482878",
];

const fmt = (x: number | null) =>
  x === null ? "—" : x >= 1000 ? `${(x / 1000).toFixed(1)}k` : x.toFixed(0);

const bitrateFor = (v: Variant, tier: "today" | "fundamental", mode: NoiseMode) => {
  if (mode === "empirical_observed_snr") {
    return tier === "today" ? v.bitrate_empirical_today : v.bitrate_empirical_fundamental;
  }
  if (mode === "empirical_anchored") {
    return tier === "today" ? v.bitrate_anchored_today : v.bitrate_anchored_fundamental;
  }
  const physical = tier === "today" ? v.bitrate_physical_today : v.bitrate_physical_fundamental;
  return physical ?? (tier === "today" ? v.bitrate_today : v.bitrate_fundamental);
};

// ── main component ────────────────────────────────────────────────────────────

export default function ScalingPlots() {
  const [activeModality, setActiveModality] = useState("meg_opm");
  const [data, setData] = useState<Record<string, ModalityData>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  // chart state
  const [sweepParam, setSweepParam] = useState("num_sensors");
  const [fixedParam, setFixedParam] = useState<string>("source_spacing_mm");
  const [fixedValue, setFixedValue] = useState<number | null>(null);
  const [chartType, setChartType] = useState<"spectra" | "bitrate" | "first_sv">("bitrate");
  const [normalized, setNormalized] = useState(true);
  const [tier, setTier] = useState<"today" | "fundamental">("today");
  const [noiseMode, setNoiseMode] = useState<NoiseMode>("physical_detector_floor");

  // Load data for active modality
  useEffect(() => {
    if (data[activeModality] || loading[activeModality]) return;
    setLoading((l) => ({ ...l, [activeModality]: true }));
    fetch(`/data/${activeModality}.json`)
      .then((r) => r.json())
      .then((d: ModalityData) => {
        setData((prev) => ({ ...prev, [activeModality]: d }));
        setNoiseMode(d.default_bitrate_mode ?? "physical_detector_floor");
        // set default sweep/fixed params
        if (d.sweep_params.length > 0) {
          setSweepParam(d.sweep_params[0]);
          const fixed = d.sweep_params.find((p) => p !== d.sweep_params[0]);
          if (fixed) setFixedParam(fixed);
        }
      })
      .finally(() => setLoading((l) => ({ ...l, [activeModality]: false })));
  }, [activeModality]);

  const modalityData = data[activeModality];

  // Unique values for the fixed parameter
  const fixedValues = useMemo(() => {
    if (!modalityData) return [];
    const seen: Record<string, boolean> = {};
    const vals: number[] = [];
    for (const v of modalityData.variants) {
      const x = (v as any)[fixedParam];
      if (x !== null && !seen[x]) { seen[x] = true; vals.push(x); }
    }
    return vals.sort((a, b) => a - b);
  }, [modalityData, fixedParam]);

  // Set default fixed value when it changes
  useEffect(() => {
    if (fixedValues.length > 0 && (fixedValue === null || !fixedValues.includes(fixedValue))) {
      // pick a middle-ish value
      const mid = fixedValues[Math.floor(fixedValues.length / 3)];
      setFixedValue(mid);
    }
  }, [fixedValues]);

  // Filtered + sorted variants for the current sweep
  const filteredVariants = useMemo(() => {
    if (!modalityData || fixedValue === null) return [];
    return modalityData.variants
      .filter((v) => {
        const fv = (v as any)[fixedParam];
        return fv !== null && Math.abs(fv - fixedValue) < 0.01;
      })
      .sort((a, b) => ((a as any)[sweepParam] ?? 0) - ((b as any)[sweepParam] ?? 0));
  }, [modalityData, sweepParam, fixedParam, fixedValue]);

  // All variants sorted by sweep param (no fixed constraint), deduplicated by sweep val → best bitrate
  const sweepVariants = useMemo(() => {
    if (!modalityData) return [];
    // group by sweep value, pick variant with highest bitrate
    const byVal: Record<number, Variant> = {};
    for (const v of modalityData.variants) {
      const sv = (v as any)[sweepParam];
      if (sv === null) continue;
      const cur = byVal[sv];
      const vBr = bitrateFor(v, tier, noiseMode);
      const curBr = cur ? bitrateFor(cur, tier, noiseMode) : -Infinity;
      if (!cur || (vBr ?? 0) > (curBr ?? 0)) byVal[sv] = v;
    }
    return Object.values(byVal).sort(
      (a, b) => ((a as any)[sweepParam] ?? 0) - ((b as any)[sweepParam] ?? 0)
    );
  }, [modalityData, sweepParam, tier, noiseMode]);

  // Bitrate chart data
  const bitrateData = useMemo(() => {
    return sweepVariants.map((v) => ({
      x: (v as any)[sweepParam],
      today: bitrateFor(v, "today", noiseMode),
      fundamental: bitrateFor(v, "fundamental", noiseMode),
    }));
  }, [sweepVariants, sweepParam, noiseMode]);

  // First SV chart data
  const firstSvData = useMemo(() => {
    return sweepVariants.map((v) => ({
      x: (v as any)[sweepParam],
      first_sv: v.first_sv,
    }));
  }, [sweepVariants, sweepParam]);

  // Spectra chart data: series per variant, x = sv_index, y = sv (or normalized)
  const spectraData = useMemo(() => {
    if (filteredVariants.length === 0) return { series: [], maxPoints: 0 };
    const globalMax = Math.max(...filteredVariants.map((v) => v.singular_values[0]));
    const series = filteredVariants.map((v, i) => {
      const scale = normalized ? globalMax : 1;
      return {
        label: `${PARAM_LABELS[sweepParam] ?? sweepParam}=${(v as any)[sweepParam]}`,
        color: COLORS[i % COLORS.length],
        data: v.sv_indices.map((idx, j) => ({
          idx,
          val: v.singular_values[j] / scale,
        })),
      };
    });
    return { series, maxPoints: Math.max(...filteredVariants.map((v) => v.n_singular_values)) };
  }, [filteredVariants, normalized, sweepParam]);

  // ── render ────────────────────────────────────────────────────────────────

  const isLoading = loading[activeModality];

  return (
    <div className="not-prose w-full" style={{ fontFamily: "system-ui, sans-serif" }}>
      {/* Modality tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 12, flexWrap: "wrap" }}>
        {MODALITIES.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveModality(key)}
            style={{
              padding: "4px 12px",
              borderRadius: 6,
              border: "1px solid #d1d5db",
              background: activeModality === key ? "#1d4ed8" : "#f9fafb",
              color: activeModality === key ? "#fff" : "#374151",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: activeModality === key ? 600 : 400,
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {isLoading && (
        <div style={{ color: "#6b7280", padding: "16px 0" }}>Loading data…</div>
      )}

      {modalityData && !isLoading && (
        <>
          {/* Controls */}
          <div style={{ display: "flex", gap: 12, marginBottom: 12, flexWrap: "wrap", alignItems: "center", fontSize: 13 }}>
            {/* Chart type */}
            <div style={{ display: "flex", gap: 4 }}>
              {(["bitrate", "spectra", "first_sv"] as const).map((ct) => (
                <button
                  key={ct}
                  onClick={() => setChartType(ct)}
                  style={{
                    padding: "3px 10px",
                    borderRadius: 5,
                    border: "1px solid #d1d5db",
                    background: chartType === ct ? "#374151" : "#f9fafb",
                    color: chartType === ct ? "#fff" : "#374151",
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                >
                  {ct === "bitrate" ? "Bitrate" : ct === "spectra" ? "Spectra" : "First SV"}
                </button>
              ))}
            </div>

            {/* Sweep selector */}
            {modalityData.sweep_params.length > 1 && (
              <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#374151" }}>
                Sweep:
                <select
                  value={sweepParam}
                  onChange={(e) => {
                    const np = e.target.value;
                    setSweepParam(np);
                    const nf = modalityData.sweep_params.find((p) => p !== np) ?? "";
                    setFixedParam(nf);
                    setFixedValue(null);
                  }}
                  style={{ border: "1px solid #d1d5db", borderRadius: 4, padding: "2px 6px", fontSize: 12 }}
                >
                  {modalityData.sweep_params.map((p) => (
                    <option key={p} value={p}>{PARAM_LABELS[p] ?? p}</option>
                  ))}
                </select>
              </label>
            )}

            {/* Fixed value (only for spectra) */}
            {chartType === "spectra" && fixedValues.length > 0 && (
              <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#374151" }}>
                Fix {PARAM_LABELS[fixedParam] ?? fixedParam}:
                <select
                  value={fixedValue ?? ""}
                  onChange={(e) => setFixedValue(Number(e.target.value))}
                  style={{ border: "1px solid #d1d5db", borderRadius: 4, padding: "2px 6px", fontSize: 12 }}
                >
                  {fixedValues.map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </label>
            )}

            {/* Noise tier */}
            <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#374151" }}>
              Noise:
              <select
                value={tier}
                onChange={(e) => setTier(e.target.value as "today" | "fundamental")}
                style={{ border: "1px solid #d1d5db", borderRadius: 4, padding: "2px 6px", fontSize: 12 }}
              >
                <option value="today">Today</option>
                <option value="fundamental">Fundamental</option>
              </select>
            </label>

            {/* Noise model */}
            {chartType === "bitrate" && (
              <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#374151" }}>
                Mode:
                <select
                  value={noiseMode}
                  onChange={(e) => setNoiseMode(e.target.value as NoiseMode)}
                  style={{ border: "1px solid #d1d5db", borderRadius: 4, padding: "2px 6px", fontSize: 12 }}
                >
                  <option value="physical_detector_floor">Physical detector floor</option>
                  <option value="empirical_observed_snr">Empirical observed SNR</option>
                  {modalityData.variants[0]?.bitrate_anchored_today != null && (
                    <option value="empirical_anchored">Empirically anchored</option>
                  )}
                </select>
              </label>
            )}

            {/* Normalized toggle (spectra only) */}
            {chartType === "spectra" && (
              <label style={{ display: "flex", alignItems: "center", gap: 4, cursor: "pointer", color: "#374151" }}>
                <input
                  type="checkbox"
                  checked={normalized}
                  onChange={(e) => setNormalized(e.target.checked)}
                />
                Normalized
              </label>
            )}
          </div>

          {/* SNR info line */}
          <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}>
            {noiseMode === "physical_detector_floor" ? (
              <>
                Physical detector floor · noise today: {modalityData.noise_label_today} ·
                source amplitude: {modalityData.source_amplitude} {modalityData.source_amplitude_units}
              </>
            ) : noiseMode === "empirical_anchored" ? (
              <>
                Empirically anchored · boundary voxels excluded; best spatial mode pinned to
                the literature single-channel SNR × √N_eff array gain (single estimate from
                the canonical 256-channel layout)
              </>
            ) : (
              <>
                Empirical observed SNR · SNR today: {modalityData.variants[0]?.snr_empirical_today?.toFixed(2) ?? "—"} ·
                typical signal: {modalityData.typical_signal}
              </>
            )}
          </div>

          {/* Chart */}
          <div style={{ width: "100%", height: 360 }}>
            {chartType === "bitrate" && (
              <BitrateChart data={bitrateData} xKey="x" xLabel={PARAM_LABELS[sweepParam] ?? sweepParam} />
            )}
            {chartType === "first_sv" && (
              <FirstSvChart data={firstSvData} xKey="x" xLabel={PARAM_LABELS[sweepParam] ?? sweepParam} />
            )}
            {chartType === "spectra" && (
              <SpectraChart series={spectraData.series} normalized={normalized} />
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ── sub-charts ────────────────────────────────────────────────────────────────

function BitrateChart({
  data,
  xKey,
  xLabel,
}: {
  data: { x: number; today: number | null; fundamental: number | null }[];
  xKey: string;
  xLabel: string;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 24, left: 16, bottom: 32 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="x" label={{ value: xLabel, position: "insideBottom", offset: -8, fontSize: 12 }} tick={{ fontSize: 11 }} />
        <YAxis label={{ value: "bits/s", angle: -90, position: "insideLeft", fontSize: 12 }} tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v)} />
        <Tooltip formatter={(v) => fmt(v as number)} labelFormatter={(l) => `${xLabel}: ${l}`} />
        <Legend verticalAlign="top" iconSize={10} wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="today" stroke="#1d4ed8" strokeWidth={2} dot={{ r: 4 }} name="Today" connectNulls />
        <Line type="monotone" dataKey="fundamental" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} name="Fundamental limit" connectNulls strokeDasharray="5 3" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function FirstSvChart({
  data,
  xKey,
  xLabel,
}: {
  data: { x: number; first_sv: number }[];
  xKey: string;
  xLabel: string;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 24, left: 16, bottom: 32 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="x" label={{ value: xLabel, position: "insideBottom", offset: -8, fontSize: 12 }} tick={{ fontSize: 11 }} />
        <YAxis label={{ value: "First singular value", angle: -90, position: "insideLeft", fontSize: 12 }} tick={{ fontSize: 11 }} />
        <Tooltip labelFormatter={(l) => `${xLabel}: ${l}`} />
        <Line type="monotone" dataKey="first_sv" stroke="#7c3aed" strokeWidth={2} dot={{ r: 4 }} name="First SV" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function SpectraChart({
  series,
  normalized,
}: {
  series: { label: string; color: string; data: { idx: number; val: number }[] }[];
  normalized: boolean;
}) {
  // Build unified data: array of {idx, series0, series1, ...}
  const seenIdx: Record<number, boolean> = {};
  series.forEach((s) => s.data.forEach((d) => { seenIdx[d.idx] = true; }));
  const sortedIdx = Object.keys(seenIdx).map(Number).sort((a, b) => a - b);

  const chartData = sortedIdx.map((idx) => {
    const row: Record<string, number> = { idx };
    series.forEach((s, i) => {
      const pt = s.data.find((d) => d.idx === idx);
      if (pt) row[`s${i}`] = pt.val;
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 8, right: 24, left: 16, bottom: 32 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="idx"
          scale="log"
          type="number"
          domain={["auto", "auto"]}
          label={{ value: "Singular value index", position: "insideBottom", offset: -8, fontSize: 12 }}
          tick={{ fontSize: 10 }}
          tickFormatter={(v) => v}
        />
        <YAxis
          scale="log"
          type="number"
          domain={["auto", "auto"]}
          label={{ value: normalized ? "SV (normalized)" : "Singular value", angle: -90, position: "insideLeft", fontSize: 11 }}
          tick={{ fontSize: 10 }}
          tickFormatter={(v) => v.toExponential(0)}
        />
        <Tooltip
          formatter={(v) => (v as number).toExponential(3)}
          labelFormatter={(l) => `Index: ${l}`}
        />
        <Legend verticalAlign="top" iconSize={8} wrapperStyle={{ fontSize: 10 }} />
        {series.map((s, i) => (
          <Line
            key={s.label}
            dataKey={`s${i}`}
            stroke={s.color}
            strokeWidth={1.5}
            dot={false}
            name={s.label}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
