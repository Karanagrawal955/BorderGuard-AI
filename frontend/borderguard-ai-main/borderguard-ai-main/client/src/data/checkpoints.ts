/*
 * BorderGuard AI visual system: all globe content is explicitly fictional demo data.
 * Coordinates are used only to place stylized nodes and arcs in the interactive scene.
 */
export type CheckpointStatus = "CLEAR" | "REVIEW" | "ELEVATED";

export type Checkpoint = {
  id: string;
  name: string;
  country: string;
  region: string;
  lat: number;
  lon: number;
  status: CheckpointStatus;
  throughput: string;
  processing: string;
  screened: string;
  note: string;
};

export const checkpoints: Checkpoint[] = [
  { id: "CP-DEL-03", name: "Indira Gateway", country: "India", region: "New Delhi", lat: 28.61, lon: 77.21, status: "CLEAR", throughput: "842 / hr", processing: "3.2 sec", screened: "12,480", note: "Primary demo lane · document flow nominal" },
  { id: "CP-MUM-02", name: "Western Harbour", country: "India", region: "Mumbai", lat: 19.07, lon: 72.87, status: "ELEVATED", throughput: "624 / hr", processing: "4.1 sec", screened: "9,208", note: "Elevated review queue · fictional signal" },
  { id: "CP-AMS-01", name: "North Sea Gate", country: "Netherlands", region: "Amsterdam", lat: 52.37, lon: 4.9, status: "CLEAR", throughput: "711 / hr", processing: "2.8 sec", screened: "10,774", note: "Demo route handshake stable" },
  { id: "CP-SIN-04", name: "Strait Sentinel", country: "Singapore", region: "Singapore", lat: 1.35, lon: 103.82, status: "REVIEW", throughput: "488 / hr", processing: "5.6 sec", screened: "7,141", note: "Review-required examples active" },
  { id: "CP-LHR-02", name: "Crown Terminal", country: "United Kingdom", region: "London", lat: 51.5, lon: -0.12, status: "CLEAR", throughput: "956 / hr", processing: "2.5 sec", screened: "14,862", note: "High-throughput demo checkpoint" },
  { id: "CP-JFK-01", name: "Atlantic Entry", country: "United States", region: "New York", lat: 40.71, lon: -74.0, status: "ELEVATED", throughput: "903 / hr", processing: "3.8 sec", screened: "13,206", note: "Cross-route sample signal" },
];

export const checkpointRoutes = [
  ["CP-DEL-03", "CP-MUM-02"], ["CP-DEL-03", "CP-SIN-04"], ["CP-DEL-03", "CP-AMS-01"],
  ["CP-AMS-01", "CP-LHR-02"], ["CP-LHR-02", "CP-JFK-01"], ["CP-JFK-01", "CP-SIN-04"],
] as const;
