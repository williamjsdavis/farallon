export type RockUnit = {
  id: number;
  name: string;
  color: string;
  rgb: number[];
};
export type Target = {
  title: string;
  labeledFraction: number;
  palette: RockUnit[];
  bounds: { xmin: number; xmax: number; ymin: number; ymax: number };
};
export type Volume = {
  bounds: number[];
  shape: number[];
  data: string;
  surface?: { shape: number[]; data: string; encoding: string };
};
export type Terrain = {
  datum_m: number;
  elevation_min_m: number;
  elevation_max_m: number;
  relief_m: number;
  source: { name: string; url: string };
  registration: { method: string; approximate: boolean };
};
export type Metrics = {
  miou: number;
  accuracy: number;
  boundary_error_px: number;
  score: number;
  per_unit: Record<string, number>;
  prediction_contact_pixels: number;
};
export type ModelResult = {
  id: string;
  scene_id?: string;
  baseline_id?: string;
  program: string;
  history: { events: ({ type: string } & Record<string, unknown>)[] };
  metrics: Metrics;
  volume: Volume;
  map_image: string;
  observed_map_image: string;
  error_image: string;
  timings: { generation_ms: number; total_ms: number };
};
export type Proposal = {
  headline: string;
  observation: string;
  expected_effect: string;
  program: string;
  parameters_to_refine: string[];
};
export type RunRecord = {
  scene_id?: string;
  baseline_id?: string;
  proposal: Proposal;
  result: ModelResult;
  before: Metrics;
  accepted: boolean;
  model_ms: number;
  elapsed_ms: number;
  search: { evaluations: number; elapsed_ms: number };
  model: string;
};
