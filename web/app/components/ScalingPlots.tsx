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
  capacity_today: number | null;
  capacity_fundamental: number | null;
  output_snr: number | null;
  noise_model_type: string | undefined;
  noise_kernel: string | undefined;
}

interface ModalityData {
  modality: string;
  label: string;
  sweep_params: string[];
  noise_label_today: string;
  noise_label_fundamental: string;
  source_amplitude: number;
  source_amplitude_units: string;
  typical_signal: number;
  variants: Variant[];
}

// Which rate to plot: achievable bitrate vs water-filled channel capacity.
type Metric = "bitrate" | "capacity";

// ── constants ────────────────────────────────────────────────────────────────

const MODALITIES = [
  { key: "meg_opm", label: "MEG OPM" },
  { key: "meg_squid", label: "MEG SQUID" },
  { key: "eeg", label: "EEG" },
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

const CONFIG_PARAMS = [
  "num_sensors",
  "source_spacing_mm",
  "grid_resolution_mm",
  "psf_fwhm_mm",
  "bold_contrast",
  "bold_snr",
  "frequency_hz",
] as const;

// viridis-ish palette
const COLORS = [
  "#440154", "#3b528b", "#21908d", "#5dc963", "#fde725",
  "#31688e", "#35b779", "#90d743", "#482878",
];

const fmt = (x: number | null) =>
  x === null ? "—" : x >= 1000 ? `${(x / 1000).toFixed(1)}k` : x.toFixed(0);

const valueLabel = (param: string, value: number | null) => {
  if (value === null) return null;
  if (param === "frequency_hz" && value >= 1000) return `${value / 1000} kHz`;
  return `${value}`;
};

const configKeyForSweep = (variant: Variant, sweepParam: string) => {
  return CONFIG_PARAMS
    .filter((param) => param !== sweepParam)
    .map((param) => `${param}:${(variant as any)[param] ?? "null"}`)
    .join("|");
};

const configLabelForSweep = (variant: Variant, sweepParam: string) => {
  return CONFIG_PARAMS
    .filter((param) => param !== sweepParam)
    .map((param) => {
      const value = (variant as any)[param] as number | null;
      const label = valueLabel(param, value);
      return label === null ? null : `${PARAM_LABELS[param] ?? param}: ${label}`;
    })
    .filter(Boolean)
    .join(", ");
};

// Higher = more physically accurate noise model.  spatial_covariance always beats
// scalar_iid; within covariance models the spherical/volume Johnson kernels beat
// parameterised exponential/gaussian approximations.
// johnson_volume (FEM volumetric) and exponential L=18.688 mm agree at N=3000 to ~1%.
// gaussian L=5mm overestimates by ~50%; spherical_johnson overestimates by ~3.5×
// (likely a calibration issue vs the volumetric ground truth).
const NOISE_KERNEL_PRIORITY: Record<string, number> = {
  johnson_volume: 5,
  exponential: 4,
  gaussian: 3,
  spherical_johnson: 2,
};

function noisePriority(v: Variant): number {
  if (v.noise_model_type !== "spatial_covariance") return 0;
  return NOISE_KERNEL_PRIORITY[v.noise_kernel ?? ""] ?? 1;
}

const metricFor = (v: Variant, tier: "today" | "fundamental", metric: Metric) => {
  if (metric === "capacity") {
    return tier === "today" ? v.capacity_today : v.capacity_fundamental;
  }
  return tier === "today" ? v.bitrate_today : v.bitrate_fundamental;
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
  const [metric, setMetric] = useState<Metric>("bitrate");

  // Load data for active modality
  useEffect(() => {
    if (data[activeModality] || loading[activeModality]) return;
    setLoading((l) => ({ ...l, [activeModality]: true }));
    fetch(`/data/${activeModality}.json`)
      .then((r) => r.json())
      .then((d: ModalityData) => {
        setData((prev) => ({ ...prev, [activeModality]: d }));
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

  // Coherent sweep: hold every non-swept configuration parameter fixed. This
  // prevents isolated points from a different source/grid setup from appearing
  // as a false dip in a sensor-count trend.
  const sweepVariants = useMemo(() => {
    if (!modalityData) return { variants: [] as Variant[], heldConfig: "", omitted: 0 };
    const groups = new Map<
      string,
      { variants: Variant[]; sweepValues: Set<number>; score: number; exemplar: Variant }
    >();

    for (const v of modalityData.variants) {
      const sv = (v as any)[sweepParam];
      if (sv === null) continue;
      const key = configKeyForSweep(v, sweepParam);
      const group = groups.get(key) ?? {
        variants: [],
        sweepValues: new Set<number>(),
        score: 0,
        exemplar: v,
      };
      group.variants.push(v);
      group.sweepValues.add(sv);
      group.score += metricFor(v, tier, metric) ?? 0;
      groups.set(key, group);
    }

    const bestGroup = Array.from(groups.values()).sort((a, b) => {
      const countDelta = b.sweepValues.size - a.sweepValues.size;
      if (countDelta !== 0) return countDelta;
      return b.score - a.score;
    })[0];

    if (!bestGroup) return { variants: [] as Variant[], heldConfig: "", omitted: 0 };

    const byVal: Record<number, Variant> = {};
    for (const v of bestGroup.variants) {
      const sv = (v as any)[sweepParam];
      if (sv === null) continue;
      const cur = byVal[sv];
      const vPri = noisePriority(v);
      const curPri = cur ? noisePriority(cur) : -1;
      const vBr = metricFor(v, tier, metric) ?? 0;
      const curBr = cur ? (metricFor(cur, tier, metric) ?? 0) : -Infinity;
      if (!cur || vPri > curPri || (vPri === curPri && vBr > curBr)) byVal[sv] = v;
    }
    const variants = Object.values(byVal).sort(
      (a, b) => ((a as any)[sweepParam] ?? 0) - ((b as any)[sweepParam] ?? 0)
    );
    const allSweepVariants = modalityData.variants.filter((v) => (v as any)[sweepParam] !== null);
    return {
      variants,
      heldConfig: configLabelForSweep(bestGroup.exemplar, sweepParam),
      omitted: allSweepVariants.length - bestGroup.variants.length,
    };
  }, [modalityData, sweepParam, tier, metric]);

  // Bitrate chart data. When the sweep has any spatial_covariance variant, suppress
  // "today" for scalar_iid points — they would otherwise create misleading spikes
  // where the correlated-noise model saturates but uncorrelated data still exists.
  const bitrateData = useMemo(() => {
    const hasSpatialCov = sweepVariants.variants.some(
      (v) => v.noise_model_type === "spatial_covariance"
    );
    return sweepVariants.variants.map((v) => ({
      x: (v as any)[sweepParam],
      today: hasSpatialCov && v.noise_model_type !== "spatial_covariance"
        ? null
        : metricFor(v, "today", metric),
      fundamental: metricFor(v, "fundamental", metric),
    }));
  }, [sweepVariants, sweepParam, metric]);

  // First SV chart data
  const firstSvData = useMemo(() => {
    return sweepVariants.variants.map((v) => ({
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

            {/* Rate metric: achievable bitrate vs water-filled capacity */}
            {chartType === "bitrate" && (
              <label style={{ display: "flex", alignItems: "center", gap: 4, color: "#374151" }}>
                Rate:
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value as Metric)}
                  style={{ border: "1px solid #d1d5db", borderRadius: 4, padding: "2px 6px", fontSize: 12 }}
                >
                  <option value="bitrate">Bitrate (achievable)</option>
                  <option value="capacity">Capacity (water-filled)</option>
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

          {/* Method info line */}
          <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}>
            <>
              {metric === "capacity" ? "Water-filled channel capacity" : "Achievable bitrate"} ·
              output signal {modalityData.typical_signal} {modalityData.source_amplitude_units} vs
              detector noise {tier === "today" ? modalityData.noise_label_today : modalityData.noise_label_fundamental}
              {modalityData.variants[0]?.output_snr != null
                ? ` · SNR ${modalityData.variants[0].output_snr.toFixed(1)}`
                : ""}
              {(() => {
                const ex = sweepVariants.variants[0];
                if (!ex) return null;
                const label = ex.noise_model_type === "spatial_covariance"
                  ? ` · correlated noise (${ex.noise_kernel ?? "covariance"})`
                  : " · uncorrelated noise (IID)";
                return <span style={{ color: ex.noise_model_type === "spatial_covariance" ? "#059669" : "#9ca3af" }}>{label}</span>;
              })()}
            </>
            {chartType !== "spectra" && sweepVariants.heldConfig && (
              <>
                <br />
                Sweep holds {sweepVariants.heldConfig}
                {sweepVariants.omitted > 0 ? ` · omitted ${sweepVariants.omitted} mismatched exported variants` : ""}
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
