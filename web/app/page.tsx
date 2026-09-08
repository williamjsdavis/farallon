'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import {
  ArrowRight,
  Check,
  ChevronRight,
  Code2,
  FlaskConical,
  Layers3,
  Loader2,
  Play,
  RotateCcw,
  Sparkles,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { GeologyBlock } from '@/components/geology-block';
import type {
  ModelResult,
  Proposal,
  RunRecord,
  Target,
  Terrain,
} from '@/lib/geology-types';

const percent = (v: number) => `${(v * 100).toFixed(1)}%`;
type SavedRun = { id: string; headline: string };
type Bootstrap = {
  target: Target;
  baseline: ModelResult;
  api_key_available: boolean;
  model: string;
  terrain?: Terrain;
  scene_id?: string;
  baseline_id?: string;
};
type SessionRun = RunRecord & { origin: 'live' | 'replay' };
async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `/api/${path}`,
    body
      ? {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }
      : undefined,
  );
  if (!response.ok) {
    const value = (await response.json().catch(() => ({}))) as {
      detail?: unknown;
    };
    throw new Error(
      typeof value.detail === 'string'
        ? value.detail
        : `Request failed (${response.status})`,
    );
  }
  return response.json() as Promise<T>;
}

export default function Home() {
  const [target, setTarget] = useState<Target>();
  const [terrain, setTerrain] = useState<Terrain>();
  const [verticalScale, setVerticalScale] = useState(1);
  const [baseline, setBaseline] = useState<ModelResult>();
  const [best, setBest] = useState<ModelResult>();
  const [selected, setSelected] = useState<ModelResult>();
  const [program, setProgram] = useState('');
  const [runs, setRuns] = useState<SessionRun[]>([]);
  const [displayAttempt, setDisplayAttempt] = useState<SessionRun>();
  const [proposal, setProposal] = useState<Proposal>();
  const [busy, setBusy] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState('');
  const [keyAvailable, setKeyAvailable] = useState(false);
  const [model, setModel] = useState('gpt-6-astra');
  const [cut, setCut] = useState(0.12);
  const [mapTab, setMapTab] = useState('units');
  const [lowerTab, setLowerTab] = useState('reasoning');
  const [saved, setSaved] = useState<{ id: string; headline: string }[]>([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [mode, setMode] = useState<'live' | 'replay' | 'manual' | 'test'>(
    'live',
  );
  const [counterfactual, setCounterfactual] = useState('');
  const [ablationParent, setAblationParent] = useState<ModelResult>();
  const isBusy = busy !== '';
  const beginWork = (message: string) => {
    setElapsed(0);
    setBusy(message);
  };

  const show = (result: ModelResult) => {
    setSelected(result);
    setProgram(result.program);
    setCounterfactual('');
    setAblationParent(undefined);
  };
  const inspect = (result: ModelResult) => {
    show(result);
    const run = runs.find((item) => item.result.id === result.id);
    setProposal(run?.proposal);
    setDisplayAttempt(run);
    setMode(run?.origin || (result.id === baseline?.id ? 'live' : 'manual'));
  };
  useEffect(() => {
    request<Bootstrap>('bootstrap')
      .then((data) => {
        setTarget(data.target);
        setTerrain(data.terrain);
        setBaseline(data.baseline);
        setBest(data.baseline);
        show(data.baseline);
        setKeyAvailable(data.api_key_available);
        setModel(data.model);
      })
      .catch((e) => setError(e.message));
    request<SavedRun[]>('runs')
      .then(setSaved)
      .catch(() => {});
  }, []);
  useEffect(() => {
    if (!isBusy) return;
    const start = Date.now();
    const timer = setInterval(
      () => setElapsed((Date.now() - start) / 1000),
      100,
    );
    return () => clearInterval(timer);
  }, [isBusy]);

  async function iterate() {
    if (!best || busy) return;
    setError('');
    setProposal(undefined);
    setDisplayAttempt(undefined);
    setMode('live');
    show(best);
    beginWork('Reading the map and comparing the geology');
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    let complete = false;
    try {
      const response = await fetch('/api/iterate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          program: best.program,
          scene_id: best.scene_id,
          baseline_id: best.baseline_id,
          previous: runs.slice(-4).map((r) => ({
            headline: r.proposal.headline,
            program: r.result.program,
            metrics: r.result.metrics,
            accepted: r.accepted,
            scene_id: r.scene_id,
            baseline_id: r.baseline_id,
          })),
          refine: true,
        }),
      });
      if (!response.ok) {
        const data = (await response.json()) as { detail?: string };
        throw new Error(data.detail || 'Could not start iteration');
      }
      if (!response.body)
        throw new Error('The response stream is unavailable.');
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const processLine = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === 'status') setBusy(event.message);
        if (event.type === 'proposal') {
          setProposal(event);
          setProgram(event.program);
        }
        if (event.type === 'error') throw new Error(event.message);
        if (event.type === 'result') {
          complete = true;
          const record: SessionRun = { ...event, origin: 'live' };
          setRuns((previous) => [...previous, record]);
          setProposal(record.proposal);
          setDisplayAttempt(record);
          if (event.accepted) {
            setBest(event.result);
            show(event.result);
          } else {
            show(best);
          }
        }
      };
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        lines.forEach(processLine);
        if (done) {
          processLine(buffer);
          break;
        }
      }
      if (!complete)
        throw new Error(
          'The connection ended before a measured result arrived. The previous best is retained.',
        );
      request<SavedRun[]>('runs')
        .then(setSaved)
        .catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Iteration failed');
      if (!complete) setProgram(best.program);
    } finally {
      if (reader) {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
      setBusy('');
    }
  }

  async function generate(refine = false) {
    if (busy || !best) return;
    beginWork(
      refine
        ? 'Tuning the parameters of this history'
        : 'Generating your geological history',
    );
    setError('');
    try {
      const result = await request<ModelResult>(refine ? 'refine' : 'render', {
        program,
        scene_id: best.scene_id,
        baseline_id: best.baseline_id,
      });
      show(result);
      setMode('manual');
      setProposal(undefined);
      setDisplayAttempt(undefined);
      if (result.metrics.score > best.metrics.score) setBest(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setBusy('');
    }
  }

  async function ablate(feature: 'plunge' | 'asymmetry') {
    if (!selected || busy) return;
    beginWork('Testing the role of this fold feature');
    setError('');
    const parent = ablationParent || selected;
    const events = structuredClone(parent.history.events);
    for (const event of events)
      if (event.type === 'anticline') {
        if (feature === 'plunge') {
          event.plunge_nw = 0;
          event.plunge_se = 0;
        } else {
          const dip = (Number(event.dip_ne) + Number(event.dip_sw)) / 2;
          event.dip_ne = dip;
          event.dip_sw = dip;
        }
      }
    const executable = events
      .map(
        (event) =>
          `${event.type}(${Object.entries(event)
            .filter(([key]) => key !== 'type')
            .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
            .join(', ')})`,
      )
      .join('\n');
    try {
      const result = await request<ModelResult>('render', {
        program: executable,
        scene_id: parent.scene_id,
        baseline_id: parent.baseline_id,
      });
      show(result);
      setAblationParent(parent);
      setMode('test');
      setProposal(undefined);
      setDisplayAttempt(undefined);
      setCounterfactual(
        feature === 'plunge' ? 'Plunge removed' : 'Limb dips made symmetric',
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not test feature');
    } finally {
      setBusy('');
    }
  }

  async function replay() {
    if (busy || !saved.length) return;
    beginWork('Loading a recorded GPT-6 proposal');
    setError('');
    try {
      const data = await request<RunRecord>(
        `runs/${saved[replayIndex % saved.length].id}`,
      );
      const record: SessionRun = { ...data, origin: 'replay' };
      setMode('replay');
      setProposal(record.proposal);
      setDisplayAttempt(record);
      setRuns((previous) =>
        previous.some((r) => r.result.id === record.result.id)
          ? previous
          : [...previous, record],
      );
      show(record.result);
      if (
        record.accepted &&
        (!best || record.result.metrics.score > best.metrics.score)
      )
        setBest(record.result);
      setReplayIndex((v) => v + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load recording');
    } finally {
      setBusy('');
    }
  }

  function reset() {
    if (!baseline) return;
    setBest(baseline);
    show(baseline);
    setRuns([]);
    setProposal(undefined);
    setDisplayAttempt(undefined);
    setMode('live');
    setReplayIndex(0);
    setError('');
  }
  const last = runs.at(-1);
  const activeRun = displayAttempt;
  const reference = ablationParent || baseline;
  const delta =
    selected && reference ? selected.metrics.miou - reference.metrics.miou : 0;

  return (
    <main className="workbench">
      <header className="topbar">
        <div className="brand">
          <span className="brand-symbol">
            <Layers3 size={23} strokeWidth={1.5} />
          </span>
          <div>
            <strong>
              farallon<span className="brand-dot">.</span>
            </strong>
            <span className="brand-caption">GEOLOGICAL HYPOTHESIS LAB</span>
          </div>
        </div>
        <div className="location">
          <span className="live-dot" /> Sheep Mountain, Wyoming{' '}
          <span className="location-detail">NW fold nose</span>
        </div>
        <div className="top-actions">
          <Button
            variant="ghost"
            disabled={!!busy || !baseline}
            onClick={reset}
            title="Return to the starting hypothesis"
          >
            <RotateCcw size={14} /> Reset
          </Button>
          <Button
            className="run-button"
            disabled={!!busy || !best || !keyAvailable}
            onClick={iterate}
          >
            {busy ? <Loader2 className="spin" /> : <Sparkles />}
            {busy ? 'Testing hypothesis' : 'Let GPT-6 investigate'}
            {!busy && <ArrowRight size={16} />}
          </Button>
        </div>
      </header>
      <section className="intro">
        <div>
          <p className="eyebrow">FROM A MAP TO A POSSIBLE HISTORY</p>
          <h1>What happened beneath the surface?</h1>
          <p>
            Read the rocks. Write a history. Generate a world. Test it against
            the evidence.
          </p>
        </div>
        <div className="model-label">
          <span className={keyAvailable ? 'live-dot' : 'offline-dot'} />
          {model}
          <small>
            {mode === 'replay'
              ? 'RECORDED RUN'
              : mode === 'manual'
                ? 'MANUAL SIMULATION'
                : mode === 'test'
                  ? 'FEATURE TEST'
                  : 'LIVE INFERENCE'}
          </small>
        </div>
      </section>
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setError('')}
            aria-label="Dismiss error"
          >
            <X />
          </Button>
        </div>
      )}
      {!selected || !target ? (
        <div className="loading-state">
          <Loader2 className="spin" />
          <h2>Warming up the geological simulator</h2>
          <p>
            {error
              ? 'Check the local Python server on port 8000, then reload.'
              : 'Preparing the map, seven rock units, and the first 3D model.'}
          </p>
        </div>
      ) : (
        <>
          <section className="main-grid">
            <div className="evidence-panel panel">
              <div className="panel-heading">
                <div>
                  <span className="step-number">01</span>
                  <h2>The surface evidence</h2>
                </div>
                <Tabs
                  value={mapTab}
                  onValueChange={(value) => setMapTab(String(value))}
                >
                  <TabsList className="compact-tabs">
                    <TabsTrigger value="units">Units</TabsTrigger>
                    <TabsTrigger value="source">Source</TabsTrigger>
                    {terrain && (
                      <TabsTrigger value="terrain">Terrain</TabsTrigger>
                    )}
                    <TabsTrigger value="error">Mismatch</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
              <div className="map-pair">
                <div className="map-frame">
                  <div className="map-caption">
                    <span>
                      {mapTab === 'terrain' ? 'MEASURED ELEVATION' : 'OBSERVED'}
                    </span>
                    <span>
                      {mapTab === 'terrain' && terrain
                        ? `${Math.round(terrain.elevation_min_m)}–${Math.round(terrain.elevation_max_m)} m · USGS 3DEP`
                        : 'NW fold nose'}
                    </span>
                  </div>
                  <div
                    className="map-image"
                    style={{
                      aspectRatio: `${(target.bounds.xmax - target.bounds.xmin) / (target.bounds.ymax - target.bounds.ymin)}`,
                    }}
                  >
                    <Image
                      unoptimized
                      width={256}
                      height={256}
                      src={
                        mapTab === 'source'
                          ? '/api/image/source'
                          : mapTab === 'terrain'
                            ? '/api/image/terrain'
                            : '/api/image/target'
                      }
                      alt={
                        mapTab === 'terrain'
                          ? 'USGS measured elevation, dark low areas to pale high areas'
                          : 'Observed geological units at the northwest nose of Sheep Mountain'
                      }
                    />
                    <span className="north-arrow">↑ N</span>
                    <div
                      className="scale-bar"
                      style={{
                        width: `${100 / (target.bounds.xmax - target.bounds.xmin)}%`,
                      }}
                    >
                      <i />1 km
                    </div>
                  </div>
                </div>
                <div className="map-frame">
                  <div className="map-caption">
                    <span>
                      {mapTab === 'error' ? 'DISAGREEMENT' : 'PREDICTED'}
                    </span>
                    <span>
                      {counterfactual ||
                        (selected.id === baseline?.id
                          ? 'Starting hypothesis'
                          : selected.id === best?.id
                            ? 'Best hypothesis'
                            : 'Candidate hypothesis')}
                    </span>
                  </div>
                  <div
                    className={`map-image ${mapTab === 'error' ? 'difference' : ''}`}
                    style={{
                      aspectRatio: `${(target.bounds.xmax - target.bounds.xmin) / (target.bounds.ymax - target.bounds.ymin)}`,
                    }}
                  >
                    <Image
                      unoptimized
                      width={256}
                      height={256}
                      src={
                        mapTab === 'error'
                          ? selected.error_image
                          : selected.map_image
                      }
                      alt={
                        mapTab === 'error'
                          ? 'Red pixels show disagreement with observed geology'
                          : 'Full predicted geological surface; only observed target pixels contribute to the score'
                      }
                    />
                    <span className="north-arrow">↑ N</span>
                    <div
                      className="scale-bar"
                      style={{
                        width: `${100 / (target.bounds.xmax - target.bounds.xmin)}%`,
                      }}
                    >
                      <i />1 km
                    </div>
                  </div>
                </div>
              </div>
              <div className="map-note">
                <span className="mask-swatch" />
                {mapTab === 'error'
                  ? 'Red = mismatch · green = agreement'
                  : `${percent(target.labeledFraction)} of this crop is observed.`}
                <span>
                  {mapTab === 'terrain'
                    ? 'Printed map coordinates give approximate registration.'
                    : 'Full prediction shown; gaps excluded from scoring.'}
                </span>
              </div>
            </div>
            <div className="model-panel panel">
              <div className="panel-heading">
                <div>
                  <span className="step-number">02</span>
                  <h2>A world that could explain it</h2>
                </div>
                <div className="volume-controls">
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={() =>
                      setVerticalScale((value) => (value === 1 ? 2 : 1))
                    }
                    title="Toggle display-only vertical exaggeration; scoring is unchanged"
                  >
                    Vertical {verticalScale}×
                  </Button>
                  <span className="resolution-tag">96³ VOXELS</span>
                </div>
              </div>
              <div className="block-stage">
                <GeologyBlock
                  volume={selected.volume}
                  surfaceImage={selected.map_image}
                  palette={target.palette}
                  cut={cut}
                  verticalScale={verticalScale}
                />
                <div className="block-top-note">
                  <span className="tiny-dot" />{' '}
                  {counterfactual ||
                    (terrain
                      ? `Real terrain · ${Math.round(terrain.relief_m)} m relief`
                      : 'Generated from executable history')}
                </div>
                <div className="block-hint">Drag to orbit · scroll to zoom</div>
                <div className="depth-label">
                  1.8 km
                  <br />
                  <span>
                    {terrain ? 'BELOW TERRAIN DATUM' : 'BELOW SURFACE'}
                  </span>
                </div>
              </div>
              <div className="cutaway-control">
                <span>Cut into the mountain</span>
                <Slider
                  aria-label="Geological cutaway position"
                  value={[cut]}
                  min={0}
                  max={0.85}
                  step={0.01}
                  onValueChange={(value) =>
                    setCut(Array.isArray(value) ? value[0] : value)
                  }
                />
                <span className="mono">{Math.round(cut * 100)}%</span>
              </div>
              <div className="feature-tests">
                <span>TEST A FEATURE</span>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={
                    !!busy ||
                    !(ablationParent || selected).history.events.some(
                      (event) => event.type === 'anticline',
                    )
                  }
                  onClick={() => ablate('plunge')}
                >
                  Remove plunge
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={
                    !!busy ||
                    !(ablationParent || selected).history.events.some(
                      (event) => event.type === 'anticline',
                    )
                  }
                  onClick={() => ablate('asymmetry')}
                >
                  Make symmetric
                </Button>
                {selected.id !== best?.id && (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={!!busy}
                    onClick={() => best && inspect(best)}
                  >
                    Restore best <RotateCcw size={12} />
                  </Button>
                )}
              </div>
            </div>
          </section>
          <div className="legend">
            <span className="legend-label">OLDEST</span>
            {target.palette.map((unit) => (
              <span key={unit.id}>
                <i style={{ background: unit.color }} />
                {unit.name}
              </span>
            ))}
            <span className="legend-label">YOUNGEST</span>
          </div>
          <section className="metrics-strip">
            <div>
              <span className="metric-label">
                Mean unit overlap{' '}
                <span title="Mean intersection over union across mapped geological units, evaluated only on the fixed observation mask.">
                  ⓘ
                </span>
              </span>
              <strong>{percent(selected.metrics.miou)}</strong>
              <span className={delta >= 0 ? 'delta' : 'delta negative'}>
                {delta >= 0 ? '+' : ''}
                {(delta * 100).toFixed(1)} points{' '}
                {ablationParent ? 'versus original model' : 'from start'}
              </span>
            </div>
            <div>
              <span className="metric-label">Contact error</span>
              <strong>
                {selected.metrics.prediction_contact_pixels === 0 ? (
                  '—'
                ) : (
                  <>
                    {selected.metrics.boundary_error_px.toFixed(1)}
                    <small> px</small>
                  </>
                )}
              </strong>
              <span>
                {selected.metrics.prediction_contact_pixels === 0
                  ? 'No predicted contacts'
                  : 'Lower is better'}
              </span>
            </div>
            <div>
              <span className="metric-label">3D + surface generation</span>
              <strong>
                {selected.timings.generation_ms.toFixed(0)}
                <small> ms</small>
              </strong>
              <span>Measured on this laptop · warm engine</span>
            </div>
            <div>
              <span className="metric-label">Visual hypotheses tested</span>
              <strong>{runs.length.toString().padStart(2, '0')}</strong>
              <span>
                {last
                  ? `${(last.model_ms / 1000).toFixed(1)} s for last model proposal`
                  : 'Ready for the first investigation'}
              </span>
            </div>
          </section>
          <section className="reasoning-grid">
            <div className="reasoning-panel panel">
              <div className="panel-heading">
                <div>
                  <span className="step-number">03</span>
                  <h2>The hypothesis loop</h2>
                </div>
                <Tabs
                  value={lowerTab}
                  onValueChange={(value) => setLowerTab(String(value))}
                >
                  <TabsList className="compact-tabs">
                    <TabsTrigger value="reasoning">Reasoning</TabsTrigger>
                    <TabsTrigger value="code">
                      <Code2 size={13} /> History
                    </TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
              {busy && (
                <output className="progress-strip">
                  <Loader2 className="spin" size={15} />
                  <span>{busy}</span>
                  <strong className="mono">{elapsed.toFixed(1)} s</strong>
                </output>
              )}
              {lowerTab === 'reasoning' ? (
                <div className="reasoning-content">
                  <div className="hypothesis-icon">
                    <FlaskConical size={23} />
                  </div>
                  <div>
                    <p className="eyebrow">
                      {mode === 'replay'
                        ? 'RECORDED GPT-6 PROPOSAL'
                        : proposal
                          ? 'GPT-6 VISUAL HYPOTHESIS'
                          : mode === 'manual'
                            ? 'YOUR EXECUTABLE HISTORY'
                            : mode === 'test'
                              ? 'CONTROLLED FEATURE TEST'
                              : 'THE STARTING QUESTION'}
                    </p>
                    <h3>
                      {proposal?.headline ||
                        (mode === 'manual'
                          ? 'Your program, measured against the map'
                          : mode === 'test'
                            ? counterfactual
                            : 'What deformation is missing from these flat layers?')}
                    </h3>
                    <p>
                      {proposal?.observation ||
                        (mode === 'manual'
                          ? 'This model was generated from the history editor. The displayed measurements compare its surface against the same fixed geological observations.'
                          : mode === 'test'
                            ? 'A single feature was changed from the original model. Compare the outcrop pattern and overlap score to see whether that feature helps explain the map.'
                            : 'We begin with seven horizontal sedimentary packages and no fold. GPT-6 sees the mapped outcrops, real topography, and the mismatch, then writes the missing geological events.')}
                    </p>
                    <div className="testable-effect">
                      <ArrowRight size={15} />
                      <span>
                        <b>Expected: </b>
                        {proposal?.expected_effect ||
                          (mode === 'test'
                            ? 'Removing an explanatory feature should make the fit worse. Each feature test begins from the same original model.'
                            : mode === 'manual'
                              ? 'Use the measured fit to decide whether this executable history explains the outcrops.'
                              : 'Find the missing geometry that makes the colored bands close around the northwest nose.')}
                      </span>
                    </div>
                    {activeRun && !busy && (
                      <div
                        className={`decision ${activeRun.accepted ? 'accepted' : 'rejected'}`}
                      >
                        {activeRun.accepted ? (
                          <Check size={14} />
                        ) : (
                          <X size={14} />
                        )}{' '}
                        {activeRun.accepted
                          ? mode === 'replay'
                            ? 'Accepted in this recorded investigation'
                            : 'Improved the combined fit — hypothesis accepted'
                          : mode === 'replay'
                            ? 'Rejected in this recorded investigation'
                            : 'Did not improve the combined fit — previous best retained'}
                        <span>
                          {activeRun.search.evaluations} numerical trials
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="code-panel">
                  <div className="code-caption">
                    <span>history.geo</span>
                    <span>
                      Oldest → youngest · km relative to terrain datum
                    </span>
                  </div>
                  <Textarea
                    className="history-editor"
                    aria-label="Executable geological history"
                    spellCheck={false}
                    value={program}
                    onChange={(event) => setProgram(event.target.value)}
                  />
                  <div className="code-actions">
                    <span>Fixed primitives. Real executable geometry.</span>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!!busy}
                      onClick={() => generate(true)}
                    >
                      Tune parameters
                    </Button>
                    <Button
                      size="sm"
                      disabled={!!busy}
                      onClick={() => generate()}
                    >
                      <Play size={13} /> Generate
                    </Button>
                  </div>
                </div>
              )}
            </div>
            <div className="history-panel panel">
              <div className="panel-heading">
                <h2>The investigation</h2>
                <span className="resolution-tag">
                  {mode === 'replay'
                    ? 'REPLAY'
                    : mode === 'manual'
                      ? 'MANUAL'
                      : mode === 'test'
                        ? 'FEATURE TEST'
                        : 'LIVE'}
                </span>
              </div>
              <div className="run-list">
                <Button
                  variant="ghost"
                  className={`run-row ${selected.id === baseline?.id ? 'selected' : ''}`}
                  disabled={!!busy}
                  onClick={() => baseline && inspect(baseline)}
                >
                  <span className="run-index">00</span>
                  <div>
                    <strong>Undeformed sedimentary layers</strong>
                    <small>Starting hypothesis</small>
                  </div>
                  <span>{baseline ? percent(baseline.metrics.miou) : ''}</span>
                  <ChevronRight size={14} />
                </Button>
                {runs.map((run, index) => (
                  <Button
                    variant="ghost"
                    key={`${run.result.id}-${index}`}
                    className={`run-row ${selected.id === run.result.id ? 'selected' : ''}`}
                    disabled={!!busy}
                    onClick={() => {
                      show(run.result);
                      setProposal(run.proposal);
                      setDisplayAttempt(run);
                      setMode(run.origin);
                    }}
                  >
                    <span className="run-index">
                      {String(index + 1).padStart(2, '0')}
                    </span>
                    <div>
                      <strong>{run.proposal.headline}</strong>
                      <small>
                        {run.origin === 'replay' ? 'Recorded · ' : ''}
                        {run.accepted ? 'Accepted' : 'Rejected'} ·{' '}
                        {(run.model_ms / 1000).toFixed(1)} s proposal
                      </small>
                    </div>
                    <span className={run.accepted ? 'delta' : ''}>
                      {percent(run.result.metrics.miou)}
                    </span>
                    <ChevronRight size={14} />
                  </Button>
                ))}
              </div>
              <div className="replay-control">
                <span>Recorded runs are a transparent backup.</span>
                <Button
                  size="xs"
                  variant="ghost"
                  disabled={!!busy || !saved.length}
                  onClick={replay}
                >
                  <Play size={11} /> Replay next ({saved.length})
                </Button>
              </div>
            </div>
          </section>
          <footer>
            <span>
              Sheep Mountain · NW fold nose ·{' '}
              <a
                href="https://doi.org/10.1130/GES00088.1"
                target="_blank"
                rel="noreferrer"
              >
                Fiore Allwardt et al., 2007 ↗
              </a>
            </span>
            <span>
              {terrain ? (
                <>
                  <a href={terrain.source.url} target="_blank" rel="noreferrer">
                    USGS 3DEP terrain ↗
                  </a>{' '}
                  · approximate registration ·{' '}
                </>
              ) : null}
              One possible subsurface history · {verticalScale}× vertical
              display
            </span>
          </footer>
        </>
      )}
    </main>
  );
}
