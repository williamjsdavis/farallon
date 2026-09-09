'use client';

import { useEffect, useRef, useState } from 'react';
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
  Square,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  NativeSelect,
  NativeSelectOption,
} from '@/components/ui/native-select';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { GeologyBlock } from '@/components/geology-block';
import { readEventStream } from '@/lib/event-stream';
import type {
  ModelResult,
  ModelPreset,
  Proposal,
  RunRecord,
  Target,
  Terrain,
} from '@/lib/geology-types';

const percent = (v: number) => `${(v * 100).toFixed(1)}%`;
const processingTier = (tier?: string | null) =>
  tier === 'fast' || tier === 'priority'
    ? 'Fast confirmed'
    : tier === 'default'
      ? 'Standard served'
      : tier
        ? `${tier} served`
        : 'Tier not recorded';
type SavedRun = { id: string; headline: string };
type ReplayPlaylist = {
  title: string;
  scene_id: string;
  baseline_id: string;
  runs: SavedRun[];
};
type Bootstrap = {
  target: Target;
  baseline: ModelResult;
  api_key_available: boolean;
  model: string;
  model_presets: ModelPreset[];
  default_model_preset: ModelPreset['id'];
  requested_service_tier?: string;
  terrain?: Terrain;
  scene_id?: string;
  baseline_id?: string;
};
type SessionRun = RunRecord & { origin: 'live' | 'replay' };
type InvestigationMode = 'live' | 'replay' | 'manual' | 'test' | 'restart';
type LiveEvent =
  | { type: 'status'; message: string }
  | ({ type: 'proposal' } & Proposal)
  | { type: 'error'; message: string }
  | ({ type: 'result' } & RunRecord);
type AutoEvent =
  | Exclude<LiveEvent, { type: 'result' }>
  | { type: 'started'; run_id: string; max_iterations: number }
  | {
      type: 'restart';
      branch: number;
      description: string;
      result: ModelResult;
      global_best: ModelResult;
    }
  | ({
      type: 'result';
      global_best: ModelResult;
      branch_best: ModelResult;
    } & RunRecord)
  | {
      type: 'complete';
      reason: 'limit' | 'stopped' | 'error';
      completed_iterations: number;
      restarts: number;
      best: ModelResult;
    };
type AutoProgress = {
  state: 'running' | 'stopping' | 'limit' | 'stopped' | 'error';
  completed: number;
  limit: number;
  branch: number;
  restarts: number;
};
type AutoControl = {
  controller: AbortController;
  runId?: string;
  stopRequested: boolean;
  stopSent: boolean;
};
type Investigation = {
  best: ModelResult;
  selected: ModelResult;
  program: string;
  runs: SessionRun[];
  displayAttempt?: SessionRun;
  proposal?: Proposal;
  mode: InvestigationMode;
  counterfactual: string;
  ablationParent?: ModelResult;
};
type ReplaySession = { live: Investigation; runs: SessionRun[] };
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
  const [modelPresets, setModelPresets] = useState<ModelPreset[]>([]);
  const [modelPreset, setModelPreset] = useState<ModelPreset['id']>('astra');
  const chosenPreset = modelPresets.find((preset) => preset.id === modelPreset);
  const modelName = (id: string) =>
    modelPresets.find((preset) => preset.model === id)?.label || id;
  const [requestedTier, setRequestedTier] = useState<string>();
  const [cut, setCut] = useState(0.12);
  const [mapTab, setMapTab] = useState('units');
  const [lowerTab, setLowerTab] = useState('reasoning');
  const [playlist, setPlaylist] = useState<ReplayPlaylist>();
  const [replaySession, setReplaySession] = useState<ReplaySession>();
  const replayIndex = replaySession?.runs.length || 0;
  const replayLength = playlist?.runs.length || 0;
  const replayComplete = replayLength > 0 && replayIndex >= replayLength;
  const [mode, setMode] = useState<InvestigationMode>('live');
  const [autoLimit, setAutoLimit] = useState('20');
  const [autoProgress, setAutoProgress] = useState<AutoProgress>();
  const autoControl = useRef<AutoControl | null>(null);
  const autoRunning =
    autoProgress?.state === 'running' || autoProgress?.state === 'stopping';
  const validAutoLimit =
    /^\d+$/.test(autoLimit) &&
    Number(autoLimit) >= 1 &&
    Number(autoLimit) <= 100;
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
    setMode(
      (result.restart ? 'restart' : run?.origin) ||
        (result.id === baseline?.id
          ? replaySession
            ? 'replay'
            : 'live'
          : 'manual'),
    );
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
        setModelPresets(data.model_presets || []);
        setModelPreset(data.default_model_preset || 'astra');
        setRequestedTier(data.requested_service_tier);
      })
      .catch((e) => setError(e.message));
    request<ReplayPlaylist>('replay')
      .then(setPlaylist)
      .catch((e) => setError(`Replay unavailable: ${e.message}`));
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
  useEffect(
    () => () => {
      const control = autoControl.current;
      autoControl.current = null;
      control?.controller.abort();
    },
    [],
  );

  async function iterate() {
    if (!best || busy || replaySession || !chosenPreset) return;
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
          model_preset: modelPreset,
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

  async function sendAutoStop(control: AutoControl) {
    if (!control.runId || control.stopSent) return;
    control.stopSent = true;
    try {
      const response = await fetch(`/api/auto/${control.runId}/stop`, {
        method: 'POST',
        signal: control.controller.signal,
      });
      // The run can finish between the click and delivery of the stop request.
      if (!response.ok && response.status !== 404)
        throw new Error('The stop request failed.');
    } catch {
      if (autoControl.current === control) {
        setError(
          'The stop signal could not be delivered. Disconnected the auto run; completed results are retained.',
        );
        control.controller.abort();
      }
    }
  }

  function stopAuto() {
    const control = autoControl.current;
    if (!control) return;
    control.stopRequested = true;
    setAutoProgress((value) =>
      value ? { ...value, state: 'stopping' } : value,
    );
    if (control.runId) void sendAutoStop(control);
    else control.controller.abort();
  }

  async function startAuto() {
    if (
      !best ||
      busy ||
      replaySession ||
      autoControl.current ||
      !chosenPreset ||
      !validAutoLimit
    )
      return;
    const limit = Number(autoLimit);
    const control: AutoControl = {
      controller: new AbortController(),
      stopRequested: false,
      stopSent: false,
    };
    autoControl.current = control;
    const knownRuns = [...runs];
    let globalBest = best;
    let complete = false;
    let failureMessage = '';
    const preserveInitialIdentity = (result: ModelResult) =>
      result.program === best.program &&
      result.scene_id === best.scene_id &&
      result.baseline_id === best.baseline_id
        ? best
        : result;
    const showBest = (result: ModelResult) => {
      const run = knownRuns.find((item) => item.result.id === result.id);
      setBest(result);
      show(result);
      setProposal(run?.proposal);
      setDisplayAttempt(run);
      setMode(
        result.restart
          ? 'restart'
          : run?.origin || (result.id === baseline?.id ? 'live' : 'manual'),
      );
    };
    setError('');
    setProposal(undefined);
    setDisplayAttempt(undefined);
    setMode('live');
    show(best);
    setAutoProgress({
      state: 'running',
      completed: 0,
      limit,
      branch: 1,
      restarts: 0,
    });
    beginWork('Starting the automatic investigation');
    try {
      const response = await fetch('/api/auto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: control.controller.signal,
        body: JSON.stringify({
          model_preset: modelPreset,
          program: best.program,
          scene_id: best.scene_id,
          baseline_id: best.baseline_id,
          max_iterations: limit,
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
      await readEventStream<AutoEvent>(response, (event) => {
        if (autoControl.current !== control) return;
        if (event.type === 'started') {
          control.runId = event.run_id;
          if (control.stopRequested) void sendAutoStop(control);
        }
        if (event.type === 'status') setBusy(event.message);
        if (event.type === 'proposal') {
          setMode('live');
          setProposal(event);
          setDisplayAttempt(undefined);
          setProgram(event.program);
        }
        if (event.type === 'restart') {
          globalBest = preserveInitialIdentity(event.global_best);
          setBest(globalBest);
          show(event.result);
          setMode('restart');
          setProposal(undefined);
          setDisplayAttempt(undefined);
          setAutoProgress((value) =>
            value
              ? {
                  ...value,
                  branch: event.branch,
                  restarts: event.branch - 1,
                }
              : value,
          );
        }
        if (event.type === 'result') {
          if (!event.auto)
            throw new Error('The server returned an incomplete auto result.');
          // Keep only one volume per recorded candidate in the session history.
          const { global_best, branch_best, ...stored } = event;
          const record: SessionRun = { ...stored, origin: 'live' };
          knownRuns.push(record);
          setRuns((previous) => [...previous, record]);
          globalBest = preserveInitialIdentity(global_best);
          setBest(globalBest);
          show(preserveInitialIdentity(branch_best));
          setMode('live');
          setProposal(record.proposal);
          setDisplayAttempt(record);
          setAutoProgress((value) =>
            value
              ? {
                  ...value,
                  completed: event.auto!.iteration,
                  branch: event.auto!.branch,
                  restarts: event.auto!.branch - 1,
                }
              : value,
          );
        }
        if (event.type === 'error') {
          failureMessage = event.message;
          setError(event.message);
        }
        if (event.type === 'complete') {
          complete = true;
          globalBest = preserveInitialIdentity(event.best);
          showBest(globalBest);
          setAutoProgress((value) =>
            value
              ? {
                  ...value,
                  state: event.reason,
                  completed: event.completed_iterations,
                  restarts: event.restarts,
                }
              : value,
          );
        }
      });
      if (!complete)
        throw new Error(
          failureMessage ||
            'The auto connection ended early. The best completed model is retained.',
        );
    } catch (e) {
      if (autoControl.current === control) {
        showBest(globalBest);
        const stopped = control.controller.signal.aborted;
        if (!stopped)
          setError(e instanceof Error ? e.message : 'Auto mode failed.');
        setAutoProgress((value) =>
          value ? { ...value, state: stopped ? 'stopped' : 'error' } : value,
        );
      }
    } finally {
      if (autoControl.current === control) {
        autoControl.current = null;
        setBusy('');
      }
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
      if (!replaySession && result.metrics.score > best.metrics.score)
        setBest(result);
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
      if (event.type === 'anticline' || event.type === 'syncline') {
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
    if (busy || !playlist || !baseline || !best || !selected || replayComplete)
      return;
    const next = playlist.runs[replayIndex];
    if (!next) return;
    beginWork('Loading a recorded proposal');
    setError('');
    try {
      if (
        playlist.scene_id !== baseline.scene_id ||
        playlist.baseline_id !== baseline.baseline_id
      )
        throw new Error(
          'The rehearsal belongs to a different scene. Reload the app.',
        );
      const data = await request<RunRecord>(`runs/${next.id}`);
      if (
        data.scene_id !== playlist.scene_id ||
        data.baseline_id !== playlist.baseline_id
      )
        throw new Error('The recorded scene changed. Reload the app.');
      const record: SessionRun = { ...data, origin: 'replay' };
      const recordedRuns = [...(replaySession?.runs || []), record];
      // A rehearsal has its own history and incumbent. Preserve live work so
      // entering replay never mixes unrelated proposals or replaces that work.
      setReplaySession({
        live: replaySession?.live || {
          best,
          selected,
          program,
          runs,
          displayAttempt,
          proposal,
          mode,
          counterfactual,
          ablationParent,
        },
        runs: recordedRuns,
      });
      setMode('replay');
      setProposal(record.proposal);
      setDisplayAttempt(record);
      setRuns(recordedRuns);
      show(record.result);
      setBest(
        recordedRuns.reduce(
          (incumbent, attempt) =>
            attempt.accepted ? attempt.result : incumbent,
          baseline,
        ),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load recording');
    } finally {
      setBusy('');
    }
  }

  function returnToLive() {
    if (busy || !replaySession) return;
    const live = replaySession.live;
    setBest(live.best);
    setSelected(live.selected);
    setProgram(live.program);
    setRuns(live.runs);
    setDisplayAttempt(live.displayAttempt);
    setProposal(live.proposal);
    setMode(live.mode);
    setCounterfactual(live.counterfactual);
    setAblationParent(live.ablationParent);
    setReplaySession(undefined);
    setError('');
  }

  function reset() {
    if (!baseline || busy) return;
    setBest(baseline);
    show(baseline);
    setRuns([]);
    setProposal(undefined);
    setDisplayAttempt(undefined);
    setMode(replaySession ? 'replay' : 'live');
    setReplaySession((session) =>
      session ? { ...session, runs: [] } : undefined,
    );
    setError('');
    setAutoProgress(undefined);
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
            title={
              replaySession
                ? 'Restart the recorded rehearsal'
                : 'Return to the starting hypothesis'
            }
          >
            <RotateCcw size={14} /> Reset
          </Button>
          <Button
            className="run-button"
            disabled={
              !!busy ||
              !best ||
              (!replaySession && (!keyAvailable || !chosenPreset))
            }
            onClick={replaySession ? returnToLive : iterate}
          >
            {busy ? (
              <Loader2 className="spin" />
            ) : replaySession ? (
              <RotateCcw />
            ) : (
              <Sparkles />
            )}
            {busy
              ? 'Testing hypothesis'
              : replaySession
                ? 'Return to live'
                : 'Test next hypothesis'}
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
          {mode === 'replay' && !displayAttempt && !proposal
            ? 'Recorded rehearsal'
            : modelName(
                displayAttempt?.model ||
                  proposal?.model ||
                  chosenPreset?.model ||
                  model,
              )}
          <small>
            {mode === 'replay'
              ? 'RECORDED RUN'
              : mode === 'manual'
                ? 'MANUAL SIMULATION'
                : mode === 'test'
                  ? 'FEATURE TEST'
                  : mode === 'restart'
                    ? 'RESTART SEED'
                    : displayAttempt && !busy
                      ? processingTier(
                          displayAttempt.service_tier,
                        ).toUpperCase()
                      : (chosenPreset?.service_tier || requestedTier) ===
                            'fast' ||
                          (chosenPreset?.service_tier || requestedTier) ===
                            'priority'
                        ? 'LIVE · FAST REQUESTED'
                        : 'LIVE INFERENCE'}
          </small>
        </div>
      </section>
      <section className="auto-panel" aria-label="Automatic investigation">
        <div className="model-controls">
          <label htmlFor="model-preset">Model</label>
          <NativeSelect
            id="model-preset"
            className="model-select"
            value={modelPreset}
            disabled={
              !!busy || !!autoRunning || !!replaySession || !modelPresets.length
            }
            aria-describedby="model-preset-note"
            onChange={(event) => {
              const preset = modelPresets.find(
                (item) => item.id === event.target.value,
              );
              if (preset) setModelPreset(preset.id);
            }}
          >
            {modelPresets.map((preset) => (
              <NativeSelectOption key={preset.id} value={preset.id}>
                {preset.label} ·{' '}
                {preset.id === 'luna'
                  ? 'Speed'
                  : preset.id === 'terra'
                    ? 'Balanced'
                    : preset.id === 'sol'
                      ? 'Strong'
                      : 'Strongest'}
              </NativeSelectOption>
            ))}
          </NativeSelect>
        </div>
        <div className="auto-controls">
          <Button
            variant={autoRunning ? 'outline' : 'default'}
            onClick={autoRunning ? stopAuto : startAuto}
            disabled={
              autoRunning
                ? autoProgress?.state === 'stopping'
                : !!busy ||
                  !best ||
                  !keyAvailable ||
                  !chosenPreset ||
                  !!replaySession ||
                  !validAutoLimit
            }
          >
            {autoRunning ? <Square size={14} /> : <Play size={14} />}
            {autoProgress?.state === 'stopping'
              ? 'Finishing iteration…'
              : autoRunning
                ? 'Stop auto'
                : 'Start auto'}
          </Button>
          <label htmlFor="auto-limit">Iterations</label>
          <Input
            id="auto-limit"
            type="number"
            min={1}
            max={100}
            step={1}
            value={autoLimit}
            disabled={!!busy || !!replaySession}
            aria-invalid={autoLimit !== '' && !validAutoLimit}
            onChange={(event) => setAutoLimit(event.target.value)}
          />
          <span>1–100</span>
        </div>
        <span id="model-preset-note" className="model-preset-note">
          {chosenPreset
            ? `Fast mode · ${chosenPreset.reasoning_effort === 'none' ? 'no reasoning tokens' : 'low reasoning'} · applies to single and auto runs`
            : 'Loading model options…'}
        </span>
        <output className="auto-status">
          {autoProgress && !replaySession
            ? `${autoProgress.state === 'running' ? 'Running' : autoProgress.state === 'stopping' ? 'Stopping after this iteration' : autoProgress.state === 'limit' ? 'Limit reached' : autoProgress.state === 'error' ? 'Stopped on error' : 'Stopped'} · ${autoProgress.completed}/${autoProgress.limit} complete · restart ${autoProgress.restarts}${best ? ` · best overall ${percent(best.metrics.miou)} overlap` : ''}`
            : replaySession
              ? 'Return to live to start an automatic investigation.'
              : 'Restarts after 4 stalled attempts or 10 iterations per branch. Keeps the best overall.'}
        </output>
        <span className="auto-note">
          Each iteration makes one API request. Stop finishes the current
          iteration.
        </span>
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
              : 'Preparing the map, eight geological units, and the first 3D model.'}
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
                    className={`map-image ${mapTab === 'units' ? 'cartographic' : ''}`}
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
                            : mapTab === 'units'
                              ? '/api/image/target-cartography'
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
                            : autoRunning
                              ? `Current branch ${autoProgress?.branch}`
                              : 'Candidate hypothesis')}
                    </span>
                  </div>
                  <div
                    className={`map-image ${mapTab === 'error' ? 'difference' : 'cartographic'}`}
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
                          : selected.cartographic_map_image ||
                            selected.map_image
                      }
                      alt={
                        mapTab === 'error'
                          ? 'Red pixels show disagreement with observed geology'
                          : 'Predicted geological surface with black unit contacts and dashed model fold axes; anticline arrows point out, syncline arrows point in'
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
                    : mapTab === 'units' || mapTab === 'source'
                      ? 'Yellow = modeled Quaternary cover. All eight units count toward the fit; source ink is excluded.'
                      : 'Full prediction shown; gaps excluded from scoring.'}
                </span>
              </div>
              {mapTab !== 'error' && (
                <div className="fold-key">
                  <span>
                    Dashed model axes · anticline arrows out · syncline arrows
                    in
                  </span>
                  <span>
                    {selected.fold_axes?.some(
                      (axis) => axis.status === 'unavailable',
                    )
                      ? 'Some axes omitted after later deformation.'
                      : selected.fold_axes?.some(
                            (axis) => axis.inferred_under_cover,
                          )
                        ? 'Older fold axes may continue beneath younger cover.'
                        : selected.fold_axes?.some(
                            (axis) => axis.status === 'visible',
                          )
                        ? 'Axes belong to the proposed fold events.'
                        : 'No active fold axis in this view.'}
                  </span>
                </div>
              )}
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
                      (event) =>
                        event.type === 'anticline' || event.type === 'syncline',
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
                      (event) =>
                        event.type === 'anticline' || event.type === 'syncline',
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
              {last && (
                <span
                  className="api-timing"
                  title={`Requested: ${last.requested_service_tier || 'not recorded'}; actual API response tier: ${last.service_tier || 'not recorded'}. API time includes image input, reasoning and output. Fast mode has no fixed end-to-end speedup.`}
                >
                  {processingTier(last.service_tier)} ·{' '}
                  {(last.search.elapsed_ms / 1000).toFixed(2)} s tuning
                </span>
              )}
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
                      {mode === 'replay' && proposal
                        ? `RECORDED ${modelName(activeRun?.model || proposal?.model || model).toUpperCase()} PROPOSAL`
                        : proposal
                          ? `${modelName(activeRun?.model || proposal.model || chosenPreset?.model || model).toUpperCase()} VISUAL HYPOTHESIS`
                          : mode === 'manual'
                            ? 'YOUR EXECUTABLE HISTORY'
                            : mode === 'test'
                              ? 'CONTROLLED FEATURE TEST'
                              : mode === 'restart'
                                ? 'RANDOMIZED RESTART'
                                : 'THE STARTING QUESTION'}
                    </p>
                    <h3>
                      {proposal?.headline ||
                        (mode === 'manual'
                          ? 'Your program, measured against the map'
                          : mode === 'test'
                            ? counterfactual
                            : mode === 'restart'
                              ? 'Explore a different starting history'
                              : 'What deformation is missing from these flat layers?')}
                    </h3>
                    <p>
                      {proposal?.observation ||
                        (mode === 'manual'
                          ? 'This model was generated from the history editor. The displayed measurements compare its surface against the same fixed geological observations.'
                          : mode === 'test'
                            ? 'A single feature was changed from the original model. Compare the outcrop pattern and overlap score to see whether that feature helps explain the map.'
                            : mode === 'restart'
                              ? selected.restart?.description ||
                                'A different starting history opens a new search branch. The best model from earlier branches is retained.'
                              : 'We begin with seven horizontal bedrock packages. The selected model sees the outcrops, yellow Quaternary cover, real topography, and the mismatch, then proposes folds and younger deposition.')}
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
                              : mode === 'restart'
                                ? 'A fresh starting point may lead to a better explanation. This seed was generated locally; the selected model will test new hypotheses from it.'
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
                        {activeRun.auto
                          ? activeRun.auto.global_improved
                            ? 'New best across all restarts'
                            : activeRun.accepted
                              ? 'Improved this branch; overall best retained'
                              : 'Did not improve this branch; branch best retained'
                          : activeRun.accepted
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
                        : mode === 'restart'
                          ? 'RESTART'
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
                        {run.auto
                          ? `Auto ${run.auto.iteration} · branch ${run.auto.branch} · ${run.auto.global_improved ? 'Overall best' : run.accepted ? 'Branch improved' : 'Rejected'}`
                          : run.accepted
                            ? 'Accepted'
                            : 'Rejected'}{' '}
                        · {modelName(run.model)} ·{' '}
                        {(run.model_ms / 1000).toFixed(1)} s proposal
                        {' · '}
                        {processingTier(run.service_tier)}
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
                <output>
                  {replayComplete
                    ? `Rehearsal complete · ${replayLength}/${replayLength}. Reset to replay again.`
                    : playlist
                      ? `${playlist.title} · ${replayIndex}/${replayLength}`
                      : 'Rehearsal unavailable'}
                </output>
                <Button
                  size="xs"
                  variant="ghost"
                  disabled={!!busy || !replayLength || replayComplete}
                  onClick={replay}
                >
                  {replayComplete ? <Check size={11} /> : <Play size={11} />}
                  {replayComplete
                    ? 'Replay complete'
                    : replayLength
                      ? `Replay next (${replayIndex + 1}/${replayLength})`
                      : 'Replay unavailable'}
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
