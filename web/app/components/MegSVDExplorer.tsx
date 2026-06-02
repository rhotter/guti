"use client";

import type { CSSProperties, PointerEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

interface MegNode {
  x: number;
  y: number;
  z: number;
  amp: number;
  signed: number;
  vx?: number;
  vy?: number;
  vz?: number;
  bx?: number;
  by?: number;
  bz?: number;
}

interface MegMode {
  component: number;
  singularValue: number;
  relativeGain: number;
  sources: MegNode[];
  detectors: MegNode[];
}

interface MegDataset {
  key: string;
  label: string;
  nSensors: number;
  nSources: number;
  sourceSpacingMm: number;
  sensorOffsetMm: number;
  brainRadiusMm: number;
  scalpRadiusMm: number;
  singularValues: number[];
  modes: MegMode[];
}

interface MegPayload {
  datasets: MegDataset[];
}

interface ViewState {
  azimuthDeg: number;
  elevationDeg: number;
}

type SampleMode = "volume" | "shell";

interface FieldSample {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  amp: number;
  signed: number;
}

const panelStyle: CSSProperties = {
  border: "1px solid #e4e4e0",
  borderRadius: 8,
  padding: 10,
  background: "#fff",
  minWidth: 0,
};

const buttonBase: CSSProperties = {
  borderWidth: 1,
  borderStyle: "solid",
  borderColor: "#d7d5cd",
  background: "#fff",
  color: "#222",
  fontFamily: "var(--sans)",
  fontSize: 12,
  lineHeight: 1,
  padding: "6px 8px",
  cursor: "pointer",
};

const activeButton: CSSProperties = {
  ...buttonBase,
  background: "#202020",
  color: "#fff",
  borderColor: "#202020",
};

const fmt = (x: number) => {
  if (Math.abs(x) >= 1e-2) return x.toFixed(3);
  return x.toExponential(2);
};

function wrapDegrees(value: number) {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function signedRgb(value: number): [number, number, number] {
  return value >= 0 ? [181, 58, 42] : [30, 101, 136];
}

function project3d(
  point: { x: number; y: number; z: number },
  radiusMm: number,
  azimuthDeg: number,
  elevationDeg: number,
  size: number,
  margin: number,
) {
  const az = (Math.PI / 180) * azimuthDeg;
  const el = (Math.PI / 180) * elevationDeg;
  const cosAz = Math.cos(az);
  const sinAz = Math.sin(az);
  const cosEl = Math.cos(el);
  const sinEl = Math.sin(el);
  const xr = point.x * cosAz - point.y * sinAz;
  const yr = point.x * sinAz + point.y * cosAz;
  const screenY = point.z * cosEl - yr * sinEl;
  const depth = yr * cosEl + point.z * sinEl;
  const scale = (size - margin * 2) / (2.1 * radiusMm);

  return {
    x: size / 2 + xr * scale,
    y: size / 2 - screenY * scale,
    depth,
    scale,
  };
}

function hemisphereGuideLines(radiusMm: number) {
  const angles = Array.from({ length: 73 }, (_, idx) => (idx / 72) * Math.PI * 2);
  const base = angles.map((theta) => ({
    x: radiusMm * Math.cos(theta),
    y: radiusMm * Math.sin(theta),
    z: 0,
  }));
  const latitude = [0.35, 0.65].map((fraction) => {
    const z = radiusMm * fraction;
    const r = Math.sqrt(radiusMm * radiusMm - z * z);
    return angles.map((theta) => ({ x: r * Math.cos(theta), y: r * Math.sin(theta), z }));
  });
  const meridians = [0, 45, 90, 135].map((deg) => {
    const phi = (Math.PI / 180) * deg;
    return Array.from({ length: 37 }, (_, idx) => {
      const theta = (idx / 36) * Math.PI;
      return {
        x: radiusMm * Math.cos(phi) * Math.cos(theta),
        y: radiusMm * Math.sin(phi) * Math.cos(theta),
        z: radiusMm * Math.sin(theta),
      };
    });
  });

  return [base, ...latitude, ...meridians];
}

function volumeFieldPoints(radiusMm: number) {
  const points: Array<{ x: number; y: number; z: number }> = [];
  const xyCount = 25;
  const zLevels = [0.04, 0.13, 0.22, 0.31, 0.4, 0.49, 0.58, 0.67, 0.76, 0.85, 0.93];

  for (const zFrac of zLevels) {
    const z = radiusMm * zFrac;
    const sliceRadius = Math.sqrt(Math.max(0, radiusMm * radiusMm - z * z));
    for (let ix = 0; ix < xyCount; ix += 1) {
      for (let iy = 0; iy < xyCount; iy += 1) {
        const x = radiusMm * (-0.92 + (1.84 * ix) / (xyCount - 1));
        const y = radiusMm * (-0.92 + (1.84 * iy) / (xyCount - 1));
        if (x * x + y * y <= sliceRadius * sliceRadius) {
          points.push({ x, y, z });
        }
      }
    }
  }

  return points;
}

function shellFieldPoints(radiusMm: number) {
  const points: Array<{ x: number; y: number; z: number }> = [];
  const rings = [
    { zFrac: 0.03, count: 96 },
    { zFrac: 0.12, count: 96 },
    { zFrac: 0.21, count: 92 },
    { zFrac: 0.3, count: 88 },
    { zFrac: 0.39, count: 84 },
    { zFrac: 0.48, count: 76 },
    { zFrac: 0.57, count: 68 },
    { zFrac: 0.66, count: 58 },
    { zFrac: 0.75, count: 48 },
    { zFrac: 0.84, count: 36 },
    { zFrac: 0.92, count: 24 },
    { zFrac: 0.97, count: 12 },
  ];

  for (const ring of rings) {
    const z = radiusMm * ring.zFrac;
    const r = Math.sqrt(Math.max(0, radiusMm * radiusMm - z * z));
    for (let idx = 0; idx < ring.count; idx += 1) {
      const theta = (idx / ring.count) * Math.PI * 2;
      points.push({
        x: r * Math.cos(theta),
        y: r * Math.sin(theta),
        z,
      });
    }
  }

  return points;
}

function interpolateNodeVector(
  point: { x: number; y: number; z: number },
  nodes: MegNode[],
  vectorKeys: [keyof MegNode, keyof MegNode, keyof MegNode],
) {
  const [xKey, yKey, zKey] = vectorKeys;
  let weightSum = 0;
  let vx = 0;
  let vy = 0;
  let vz = 0;

  for (const node of nodes) {
    const dx = point.x - node.x;
    const dy = point.y - node.y;
    const dz = point.z - node.z;
    const dist2 = dx * dx + dy * dy + dz * dz + 64;
    const weight = 1 / Math.pow(dist2, 1.65);
    weightSum += weight;
    vx += weight * Number(node[xKey] ?? 0);
    vy += weight * Number(node[yKey] ?? 0);
    vz += weight * Number(node[zKey] ?? 0);
  }

  return { vx: vx / weightSum, vy: vy / weightSum, vz: vz / weightSum };
}

function buildFieldSamples(
  nodes: MegNode[],
  vectorKeys: [keyof MegNode, keyof MegNode, keyof MegNode],
  radiusMm: number,
  sampleMode: SampleMode,
) {
  const points = sampleMode === "volume" ? volumeFieldPoints(radiusMm) : shellFieldPoints(radiusMm);
  const samples: FieldSample[] = points.map((point) => {
    const vector = interpolateNodeVector(point, nodes, vectorKeys);
    const radius = Math.max(1e-9, Math.hypot(point.x, point.y, point.z));
    return {
      ...point,
      ...vector,
      amp: Math.hypot(vector.vx, vector.vy, vector.vz),
      signed: (vector.vx * point.x + vector.vy * point.y + vector.vz * point.z) / radius,
    };
  });
  const maxAmp = Math.max(1e-12, ...samples.map((sample) => sample.amp));
  const maxSigned = Math.max(1e-12, ...samples.map((sample) => Math.abs(sample.signed)));

  return samples.map((sample) => ({
    ...sample,
    amp: sample.amp / maxAmp,
    signed: sample.signed / maxSigned,
  }));
}

function drawArrow(
  context: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  headSize: number,
) {
  const angle = Math.atan2(y2 - y1, x2 - x1);

  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();

  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(
    x2 - Math.cos(angle - Math.PI / 7) * headSize,
    y2 - Math.sin(angle - Math.PI / 7) * headSize,
  );
  context.lineTo(
    x2 - Math.cos(angle + Math.PI / 7) * headSize,
    y2 - Math.sin(angle + Math.PI / 7) * headSize,
  );
  context.closePath();
  context.fill();
}

function SvdPlot3D({
  title,
  nodes,
  radiusMm,
  view,
  isDragging,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  vectorKeys,
  dotScale,
  sampleMode,
}: {
  title: string;
  nodes: MegNode[];
  radiusMm: number;
  view: ViewState;
  isDragging: boolean;
  onPointerDown: (event: PointerEvent<HTMLDivElement>) => void;
  onPointerMove: (event: PointerEvent<HTMLDivElement>) => void;
  onPointerUp: (event: PointerEvent<HTMLDivElement>) => void;
  vectorKeys: [keyof MegNode, keyof MegNode, keyof MegNode];
  dotScale: number;
  sampleMode: SampleMode;
}) {
  const margin = 14;
  const size = 320;
  const { azimuthDeg, elevationDeg } = view;
  const [xKey, yKey, zKey] = vectorKeys;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const projectedNodes = useMemo(() => {
    const projected = nodes.map((node, idx) => ({
      node,
      idx,
      projected: project3d(node, radiusMm, azimuthDeg, elevationDeg, size, margin),
    }));
    const depths = projected.map((item) => item.projected.depth);
    const minDepth = Math.min(...depths);
    const maxDepth = Math.max(...depths);
    const span = Math.max(1e-9, maxDepth - minDepth);

    return projected
      .map((item) => ({
        ...item,
        depthT: (item.projected.depth - minDepth) / span,
      }))
      .sort((a, b) => a.projected.depth - b.projected.depth);
  }, [nodes, radiusMm, azimuthDeg, elevationDeg]);

  const fieldSamples = useMemo(
    () => buildFieldSamples(nodes, [xKey, yKey, zKey], radiusMm, sampleMode),
    [nodes, xKey, yKey, zKey, radiusMm, sampleMode],
  );

  const projectedField = useMemo(() => {
    const projected = fieldSamples.map((sample, idx) => ({
      sample,
      idx,
      projected: project3d(sample, radiusMm, azimuthDeg, elevationDeg, size, margin),
    }));
    const depths = projected.map((item) => item.projected.depth);
    const minDepth = Math.min(...depths);
    const maxDepth = Math.max(...depths);
    const span = Math.max(1e-9, maxDepth - minDepth);

    return projected
      .map((item) => ({
        ...item,
        depthT: (item.projected.depth - minDepth) / span,
      }))
      .sort((a, b) => a.projected.depth - b.projected.depth);
  }, [fieldSamples, radiusMm, azimuthDeg, elevationDeg]);

  const maxVector = useMemo(() => {
    return Math.max(
      1e-12,
      ...fieldSamples.map((sample) => Math.hypot(sample.vx, sample.vy, sample.vz)),
    );
  }, [fieldSamples]);

  const guides = useMemo(() => hemisphereGuideLines(radiusMm), [radiusMm]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = "100%";
    canvas.style.height = "auto";
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, size, size);
    context.fillStyle = "#fff";
    context.fillRect(0, 0, size, size);

    guides.forEach((line, idx) => {
      context.beginPath();
      line.forEach((point, pointIdx) => {
        const projected = project3d(point, radiusMm, azimuthDeg, elevationDeg, size, margin);
        if (pointIdx === 0) context.moveTo(projected.x, projected.y);
        else context.lineTo(projected.x, projected.y);
      });
      context.strokeStyle =
        sampleMode === "shell" ? (idx === 0 ? "rgba(217,214,203,0.55)" : "rgba(236,233,223,0.48)") : idx === 0 ? "#d9d6cb" : "#ece9df";
      context.lineWidth = idx === 0 ? 1 : 0.7;
      context.stroke();
    });

    for (const { sample, projected, depthT } of projectedField) {
      const [r, g, b] = signedRgb(sample.signed);
      const blobRadius = (sampleMode === "volume" ? 9.5 : 18) * (0.82 + sample.amp * 0.5);
      const alpha =
        sampleMode === "volume"
          ? (0.045 + depthT * 0.065) * (0.45 + sample.amp * 0.55)
          : (0.032 + depthT * 0.052) * (0.6 + sample.amp * 0.4);
      const gradient = context.createRadialGradient(
        projected.x,
        projected.y,
        0,
        projected.x,
        projected.y,
        blobRadius,
      );
      gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${alpha})`);
      gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
      context.fillStyle = gradient;
      context.beginPath();
      context.arc(projected.x, projected.y, blobRadius, 0, Math.PI * 2);
      context.fill();
    }

    const textureStep = sampleMode === "volume" ? 3 : 1;
    projectedField.forEach(({ sample, idx, projected, depthT }) => {
      if (idx % textureStep !== 0) return;
      const end = project3d(
        {
          x: sample.x + (sample.vx / maxVector) * radiusMm * 0.16,
          y: sample.y + (sample.vy / maxVector) * radiusMm * 0.16,
          z: sample.z + (sample.vz / maxVector) * radiusMm * 0.16,
        },
        radiusMm,
        azimuthDeg,
        elevationDeg,
        size,
        margin,
      );
      const dx = end.x - projected.x;
      const dy = end.y - projected.y;
      const [r, g, b] = signedRgb(sample.signed);
      const alpha = sampleMode === "volume" ? 0.28 + depthT * 0.3 : 0.38 + depthT * 0.34;
      const x1 = projected.x - dx * 0.43;
      const y1 = projected.y - dy * 0.43;
      const x2 = projected.x + dx * 0.43;
      const y2 = projected.y + dy * 0.43;

      context.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      context.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      context.lineWidth = sampleMode === "volume" ? 0.7 + sample.amp * 0.45 : 0.85 + sample.amp * 0.5;
      context.lineCap = "round";
      drawArrow(context, x1, y1, x2, y2, sampleMode === "volume" ? 2.4 : 2.8);
    });

    for (const { node, projected, depthT } of projectedNodes) {
      const [r, g, b] = signedRgb(node.signed);
      context.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.06 + depthT * 0.08})`;
      context.beginPath();
      context.arc(projected.x, projected.y, 0.4 + node.amp * 0.8, 0, Math.PI * 2);
      context.fill();
    }
  }, [
    azimuthDeg,
    dotScale,
    elevationDeg,
    guides,
    margin,
    maxVector,
    projectedField,
    projectedNodes,
    radiusMm,
    sampleMode,
    size,
  ]);

  return (
    <div
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      style={{
        ...panelStyle,
        cursor: isDragging ? "grabbing" : "grab",
        touchAction: "none",
        userSelect: "none",
      }}
    >
      <div style={{ fontFamily: "var(--sans)", fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
        {title}
      </div>
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        style={{ display: "block", width: "100%", height: "auto" }}
      />
    </div>
  );
}

export default function MegSVDExplorer() {
  const [payload, setPayload] = useState<MegPayload | null>(null);
  const [datasetKey, setDatasetKey] = useState("opm");
  const [component, setComponent] = useState(1);
  const [view, setView] = useState<ViewState>({ azimuthDeg: -35, elevationDeg: 24 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{
    pointerId: number;
    x: number;
    y: number;
    view: ViewState;
  } | null>(null);

  useEffect(() => {
    fetch("/data/meg_svd_components.json")
      .then((res) => res.json())
      .then((data: MegPayload) => setPayload(data));
  }, []);

  const dataset = payload?.datasets.find((item) => item.key === datasetKey) ?? payload?.datasets[0];
  const mode = dataset?.modes.find((item) => item.component === component) ?? dataset?.modes[0];

  useEffect(() => {
    if (dataset && !dataset.modes.some((item) => item.component === component)) {
      setComponent(dataset.modes[0]?.component ?? 1);
    }
  }, [dataset, component]);

  useEffect(() => {
    if (!dataset || dataset.modes.length < 2) return undefined;
    const interval = window.setInterval(() => {
      setComponent((current) => {
        const currentIdx = dataset.modes.findIndex((item) => item.component === current);
        const nextIdx = currentIdx >= 0 ? (currentIdx + 1) % dataset.modes.length : 0;
        return dataset.modes[nextIdx].component;
      });
    }, 3000);

    return () => window.clearInterval(interval);
  }, [dataset]);

  useEffect(() => {
    if (isDragging) return undefined;
    const interval = window.setInterval(() => {
      setView((current) => ({
        ...current,
        azimuthDeg: wrapDegrees(current.azimuthDeg + 0.28),
      }));
    }, 40);

    return () => window.clearInterval(interval);
  }, [isDragging]);

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      view,
    };
    setIsDragging(true);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    setView({
      azimuthDeg: wrapDegrees(drag.view.azimuthDeg - dx * 0.35),
      elevationDeg: clamp(drag.view.elevationDeg + dy * 0.28, -15, 72),
    });
  };

  const handlePointerUp = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag?.pointerId === event.pointerId) {
      dragRef.current = null;
      setIsDragging(false);
    }
  };

  if (!payload || !dataset || !mode) {
    return (
      <div style={{ fontFamily: "var(--sans)", fontSize: 13, color: "#666" }}>
        Loading MEG SVD components...
      </div>
    );
  }

  return (
    <div
      style={{
        fontFamily: "var(--sans)",
        fontSize: 13,
        lineHeight: 1.4,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
          marginBottom: 10,
        }}
      >
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {payload.datasets.map((item, idx) => (
            <button
              key={item.key}
              type="button"
              aria-pressed={dataset.key === item.key}
              onClick={() => setDatasetKey(item.key)}
              style={{
                ...(dataset.key === item.key ? activeButton : buttonBase),
                borderRadius: idx === 0 ? "6px 2px 2px 6px" : "2px 6px 6px 2px",
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "auto minmax(150px, 240px) auto",
            alignItems: "center",
            gap: 8,
            flex: "1 1 260px",
            maxWidth: 380,
          }}
        >
          <span style={{ color: "#555" }}>component</span>
          <input
            type="range"
            min={dataset.modes[0]?.component ?? 1}
            max={dataset.modes[dataset.modes.length - 1]?.component ?? 1}
            step={1}
            value={mode.component}
            onChange={(event) => setComponent(Number(event.target.value))}
            style={{ width: "100%" }}
          />
          <strong style={{ fontSize: 13 }}>{mode.component}</strong>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "stretch",
          flexWrap: "wrap",
        }}
      >
        <div style={{ flex: "1 1 280px", minWidth: 250 }}>
          <SvdPlot3D
            title="Right singular vector: source pattern"
            nodes={mode.sources}
            radiusMm={dataset.brainRadiusMm}
            view={view}
            isDragging={isDragging}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            vectorKeys={["vx", "vy", "vz"]}
            dotScale={3.4}
            sampleMode="volume"
          />
        </div>
        <div
          aria-hidden="true"
          style={{
            alignSelf: "center",
            flex: "0 0 42px",
            display: "grid",
            placeItems: "center",
          }}
        >
          <svg viewBox="0 0 48 24" style={{ width: 42, height: 24, display: "block" }}>
            <path
              d="M 4 12 H 39 M 31 5 L 40 12 L 31 19"
              fill="none"
              stroke="#9a9588"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div style={{ flex: "1 1 280px", minWidth: 250 }}>
          <SvdPlot3D
            title="Left singular vector: detector pattern"
            nodes={mode.detectors}
            radiusMm={dataset.scalpRadiusMm + dataset.sensorOffsetMm}
            view={view}
            isDragging={isDragging}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            vectorKeys={["bx", "by", "bz"]}
            dotScale={3.6}
            sampleMode="shell"
          />
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
          gap: 6,
          marginTop: 8,
          color: "#4b4b46",
        }}
      >
        <div>component {mode.component}</div>
        <div>sigma {fmt(mode.singularValue)}</div>
        <div>relative gain {(100 * mode.relativeGain).toFixed(1)}%</div>
        <div>
          {dataset.nSensors} sensors, {dataset.nSources} sources
        </div>
      </div>
    </div>
  );
}
