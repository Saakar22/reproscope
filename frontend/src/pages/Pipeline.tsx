import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import Execution from '../components/Execution'
import ParsedResults from '../components/ParsedResults'
import RepoCard from '../components/RepoCard'
import RunPlans from '../components/RunPlans'
import {
  Badge,
  Card,
  Empty,
  ErrorBox,
  JobStatusBadge,
  Mono,
  STAGE_STATUS_LABEL,
  SectionTitle,
  Spinner,
  StageIcon,
  VerdictBadge,
  fmtBytes,
  fmtTime,
} from '../components/ui'
import { useJobEvents } from '../hooks/useJobEvents'
import type { Job, PipelineEvent, StageId, StageStatus, Verdict } from '../types'

const STAGES: { id: StageId; label: string; what: string }[] = [
  { id: 'ingest', label: 'Ingest', what: 'Store the PDF and repository, extract page text, record the commit' },
  { id: 'extract', label: 'Claim Extraction', what: 'Extract reported results and check each against the PDF text' },
  { id: 'scan', label: 'Repository Scan', what: 'Find scripts, CLI defaults, configs, dependencies and seeds' },
  { id: 'plan', label: 'Run Planning', what: 'Choose the command per claim and validate it against the repo' },
  { id: 'execute', label: 'Experiment Execution', what: 'Run in an isolated CPU-only Docker container' },
  { id: 'parse', label: 'Result Parsing', what: 'Read metrics from output files and logs' },
  { id: 'compare', label: 'Comparison & Diagnosis', what: 'Verdicts, discrepancy checks and evidence' },
]

/** In replay mode the job row already shows the final state, so stage status is rebuilt from events. */
function stagesFromEvents(events: PipelineEvent[], job: Job): Record<StageId, StageStatus> {
  const out = Object.fromEntries(STAGES.map((s) => [s.id, 'pending'])) as Record<StageId, StageStatus>
  for (const e of events) {
    const d = (e.data ?? {}) as { status?: StageStatus; not_implemented?: StageId[] }
    if (e.stage && e.level === 'stage' && d.status) out[e.stage] = d.status
    if (e.stage && e.level === 'error') out[e.stage] = 'failed'
    if (d.not_implemented) for (const s of d.not_implemented) out[s] = 'not_implemented'
    if (e.message === 'job_finished') return job.stages
  }
  return out
}

export default function Pipeline() {
  const { jobId } = useParams<{ jobId: string }>()
  const [params, setParams] = useSearchParams()
  const replay = params.get('replay') === '1'
  const speed = Number(params.get('speed') ?? 4) || 4

  const jobQuery = useQuery({ queryKey: ['job', jobId], queryFn: () => api.job(jobId!), enabled: !!jobId })
  const { events, state, error } = useJobEvents(jobId, replay ? 'replay' : 'live', speed)
  const job = jobQuery.data

  if (jobQuery.isLoading) return <Spinner label="Loading analysis…" />
  if (jobQuery.isError || !job)
    return <ErrorBox title="Analysis not found" message={(jobQuery.error as Error)?.message ?? 'Unknown job'} />

  const stages = replay ? stagesFromEvents(events, job) : job.stages
  const done = STAGES.filter((s) => stages[s.id] === 'done').length
  const finished = ['done', 'failed', 'incomplete'].includes(job.status)

  return (
    <div className="animate-fade-in space-y-6">
      {replay && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-info/30 bg-info-soft px-4 py-2.5 text-sm text-info">
          <span>
            <strong>Replay mode</strong> — re-streaming the stored events of a finished analysis at {speed}× speed. Nothing is
            being executed.
          </span>
          <button type="button" onClick={() => setParams({})} className="font-semibold underline underline-offset-2">
            Exit replay
          </button>
        </div>
      )}

      <JobHeader job={job} />

      <Card className="p-5">
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="font-medium">
            {done} of {STAGES.length} stages complete
          </span>
          <span className="text-xs text-faint">
            {state === 'open' && !finished && 'Live'}
            {state === 'connecting' && 'Connecting…'}
            {state === 'finished' && (replay ? 'Replay finished' : 'Stream closed')}
          </span>
        </div>
        <div className="flex h-2 gap-0.5 overflow-hidden rounded-full" aria-hidden>
          {STAGES.map((s) => (
            <div
              key={s.id}
              className={`flex-1 transition-colors duration-500 ${
                stages[s.id] === 'done'
                  ? 'bg-ok'
                  : stages[s.id] === 'running'
                    ? 'bg-info animate-pulse-dot'
                    : stages[s.id] === 'failed'
                      ? 'bg-bad'
                      : stages[s.id] === 'not_implemented'
                        ? 'bg-warn/40'
                        : 'bg-rule'
              }`}
            />
          ))}
        </div>
      </Card>

      {error && <ErrorBox title="Event stream" message={error} />}
      {job.error && <ErrorBox title="The analysis failed" message={job.error} />}
      {job.status === 'incomplete' && !replay && (
        <div className="rounded-lg border border-warn/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          <strong>Stopped early.</strong> {job.stage_detail}. These stages are reported as not implemented rather than
          simulated; no results were fabricated.
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <div className="space-y-6">
          <Card className="p-5">
            <SectionTitle kicker="Pipeline" title="Stages" />
            <ol className="space-y-1">
              {STAGES.map((s, i) => (
                <li
                  key={s.id}
                  className={`flex gap-3 rounded-lg px-2 py-2.5 ${stages[s.id] === 'running' ? 'bg-info-soft/60' : ''}`}
                >
                  <StageIcon status={stages[s.id]} />
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-baseline gap-x-2 text-sm font-medium">
                      <span className="text-faint">{i + 1}.</span> {s.label}
                      <span className="text-xs font-normal text-faint">{STAGE_STATUS_LABEL[stages[s.id]]}</span>
                    </div>
                    <div className="text-xs text-muted">{s.what}</div>
                  </div>
                </li>
              ))}
            </ol>
          </Card>
          {stages.ingest === 'done' && <IngestSummary jobId={job.id} />}
        </div>

        <EventLog events={events} live={!replay && !finished} />
      </div>

      {stages.extract === 'done' && <ExtractedClaims jobId={job.id} />}
      {stages.scan === 'done' && <RepoCard jobId={job.id} />}
      {stages.plan === 'done' && <RunPlans jobId={job.id} />}
      {stages.execute === 'done' && <Execution jobId={job.id} />}
      {stages.parse === 'done' && <ParsedResults jobId={job.id} />}
      {stages.compare === 'done' && <CompareSummary jobId={job.id} />}
    </div>
  )
}

function CompareSummary({ jobId }: { jobId: string }) {
  const { data } = useQuery({
    queryKey: ['stage', jobId, 'compare'],
    queryFn: () =>
      api.stage<{ counts: Record<Verdict, number>; findings: string[]; hypothesis_tests: unknown[] }>(jobId, 'compare'),
  })
  if (!data) return null
  const order: Verdict[] = ['REPRODUCED', 'PARTIAL', 'NOT_REPRODUCED', 'INCONCLUSIVE', 'NOT_RUN', 'NO_METRIC', 'UNVERIFIED_CLAIM']
  return (
    <Card className="p-5">
      <SectionTitle kicker="Stage 7 output" title="Verdicts">
        <Link
          to={`/jobs/${jobId}/report`}
          className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-accent/90"
        >
          Open the reproducibility report
        </Link>
      </SectionTitle>
      <div className="flex flex-wrap items-center gap-2">
        {order
          .filter((v) => data.counts[v])
          .map((v) => (
            <span key={v} className="inline-flex items-center gap-1.5">
              <VerdictBadge verdict={v} />
              <span className="font-mono text-sm">×{data.counts[v]}</span>
            </span>
          ))}
        <span className="text-sm text-muted">
          · {data.findings.length} finding(s) · {data.hypothesis_tests.length} hypothesis test(s)
        </span>
      </div>
    </Card>
  )
}

function ExtractedClaims({ jobId }: { jobId: string }) {
  const { data, isError, error } = useQuery({ queryKey: ['stage', jobId, 'extract'], queryFn: () => api.extracted(jobId) })
  if (isError) return <ErrorBox title="Could not load extracted claims" message={(error as Error).message} />
  if (!data) return <Spinner label="Loading claims…" />
  const grounded = data.claims.filter((c) => c.grounded).length
  return (
    <Card className="p-5">
      <SectionTitle kicker="Stage 2 output" title="Extracted claims">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
          {data.origin === 'llm' ? (
            <Badge tone="accent">
              {data.model}
              {data.cached && ' · cached'}
            </Badge>
          ) : (
            <Badge tone="warn" title="No LLM key: a regex heuristic produced these, not an LLM">dev-mode heuristic</Badge>
          )}
          <span>
            {grounded} of {data.claims.length} grounded in the PDF
          </span>
          {data.usage?.total_tokens != null && <span>· {data.usage.total_tokens.toLocaleString()} tokens</span>}
        </div>
      </SectionTitle>
      {data.claims.length === 0 ? (
        <Empty title="No quantitative claims were found in the paper" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[44rem] text-sm">
            <thead>
              <tr className="border-b border-rule text-left text-xs text-faint">
                <th className="py-2 pr-3 font-medium">Claim</th>
                <th className="py-2 pr-3 font-medium">Experiment</th>
                <th className="py-2 pr-3 font-medium">Metric</th>
                <th className="py-2 pr-3 text-right font-medium">Reported</th>
                <th className="py-2 pr-3 font-medium">Paper evidence</th>
                <th className="py-2 pr-3 font-medium">Stated hyperparameters</th>
                <th className="py-2 font-medium">Grounding</th>
              </tr>
            </thead>
            <tbody>
              {data.claims.map((c) => (
                <tr key={c.id} className={`border-b border-rule/70 align-top ${c.grounded ? '' : 'text-faint'}`}>
                  <td className="py-2.5 pr-3 font-mono text-xs font-semibold">{c.id}</td>
                  <td className="py-2.5 pr-3">
                    <div className="font-medium">{c.experiment}</div>
                    <div className="text-xs text-faint">{[c.dataset, c.split].filter(Boolean).join(' · ')}</div>
                  </td>
                  <td className="py-2.5 pr-3">{c.metric}</td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums whitespace-nowrap">
                    {c.reported_value}
                    {c.unit === 'percent' && '%'}
                    {c.reported_std != null && <span className="text-faint"> ±{c.reported_std}</span>}
                  </td>
                  <td className="max-w-[18rem] py-2.5 pr-3">
                    <div className="text-xs text-faint">
                      p.{c.page}
                      {c.source && ` · ${c.source}`}
                      {c.grounding.llm_page !== c.page && ` (LLM cited p.${c.grounding.llm_page})`}
                    </div>
                    <div className="truncate font-serif italic" title={c.quote}>“{c.quote}”</div>
                  </td>
                  <td className="py-2.5 pr-3">
                    <div className="flex max-w-[16rem] flex-wrap gap-1">
                      {c.hyperparameters.length === 0 && <span className="text-xs text-faint">none stated</span>}
                      {c.hyperparameters.map((h) => (
                        <span
                          key={h.name}
                          title={h.verified ? `“${h.quote}”` : 'Not found verbatim in the paper; not treated as paper-stated'}
                          className={`rounded px-1.5 py-0.5 font-mono text-[11px] ${h.verified ? 'bg-neutral-soft' : 'bg-info-soft text-info line-through'}`}
                        >
                          {h.name}={h.value}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="py-2.5">
                    {c.grounded ? (
                      <Badge tone="ok" title={`quote match ${c.grounding.quote_score}, number found`}>✓ grounded</Badge>
                    ) : (
                      <Badge tone="info" title={`quote match ${c.grounding.quote_score}; number ${c.grounding.number_found ? 'found' : 'not found'}`}>
                        unverified
                      </Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data.rejected_malformed > 0 && (
        <p className="mt-3 text-xs text-warn">{data.rejected_malformed} malformed item(s) from the LLM were discarded by schema validation.</p>
      )}
    </Card>
  )
}

function JobHeader({ job }: { job: Job }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        <div className="text-[11px] font-semibold tracking-[0.14em] text-faint uppercase">Analysis {job.id}</div>
        <h1 className="mt-1 font-serif text-2xl font-semibold tracking-tight sm:text-3xl">
          {job.paper_title ?? job.pdf_filename ?? 'Untitled paper'}
        </h1>
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted">
          <span className="truncate">{job.repo_url ?? job.repo_filename}</span>
          {job.repo_commit && <Mono className="text-faint">@{job.repo_commit.slice(0, 10)}</Mono>}
          {job.pdf_pages && <span>{job.pdf_pages} pages</span>}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {job.llm_mode === 'dev' && <Badge tone="warn" title="LLM stages use labelled development samples">LLM dev mode</Badge>}
        <JobStatusBadge status={job.status} />
        {job.status === 'done' && (
          <Link
            to={`/jobs/${job.id}/report`}
            className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-semibold text-white hover:bg-accent/90"
          >
            View report
          </Link>
        )}
      </div>
    </div>
  )
}

function IngestSummary({ jobId }: { jobId: string }) {
  const { data, isError } = useQuery({ queryKey: ['stage', jobId, 'ingest'], queryFn: () => api.ingest(jobId) })
  if (isError) return null
  if (!data) return <Spinner />
  const rows: [string, ReactNode][] = [
    ['Paper title', data.pdf.title ?? <span className="text-faint">not detected</span>],
    ['Pages with text', `${data.pdf.pages - data.pdf.empty_pages.length} of ${data.pdf.pages}`],
    ['Repository', data.repo.source === 'git' ? data.repo.url : 'Uploaded ZIP'],
    ['Commit', data.repo.commit ? <Mono>{data.repo.commit}</Mono> : <span className="text-faint">none (no git metadata)</span>],
    ['Files', `${data.repo.files} files · ${data.repo.python_files} Python · ${fmtBytes(data.repo.size_bytes)}`],
  ]
  return (
    <Card className="p-5">
      <SectionTitle kicker="Stage 1 output" title="Ingested inputs" />
      <dl className="grid grid-cols-[8.5rem_minmax(0,1fr)] gap-x-3 gap-y-2 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted">{k}</dt>
            <dd className="min-w-0 break-words">{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
  )
}

const LEVEL_STYLE: Record<PipelineEvent['level'], string> = {
  info: 'text-ink',
  log: 'text-muted',
  stage: 'text-accent font-medium',
  warning: 'text-warn',
  error: 'text-bad font-medium',
}

function EventLog({ events, live }: { events: PipelineEvent[]; live: boolean }) {
  const [filter, setFilter] = useState<'all' | 'problems'>('all')
  const [follow, setFollow] = useState(true)
  const box = useRef<HTMLDivElement>(null)
  const shown = useMemo(
    () =>
      events.filter(
        (e) => e.message !== 'job_finished' && (filter === 'all' || e.level === 'warning' || e.level === 'error'),
      ),
    [events, filter],
  )
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [shown, follow])

  return (
    <Card className="flex min-h-[28rem] flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-rule px-5 py-3">
        <div className="flex items-center gap-2">
          <h2 className="font-serif text-lg font-semibold">Event log</h2>
          {live && <span className="h-2 w-2 rounded-full bg-info animate-pulse-dot" aria-label="live" />}
        </div>
        <div className="flex items-center gap-3 text-xs">
          <label className="flex items-center gap-1.5 text-muted">
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow
          </label>
          <div className="inline-flex rounded-md border border-rule p-0.5">
            {(['all', 'problems'] as const).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={`rounded px-2 py-0.5 ${filter === f ? 'bg-neutral-soft font-semibold text-ink' : 'text-muted'}`}
              >
                {f === 'all' ? 'All' : 'Warnings & errors'}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div ref={box} className="max-h-[36rem] flex-1 overflow-y-auto px-5 py-3 font-mono text-[12.5px] leading-relaxed">
        {shown.length === 0 ? (
          <Empty title={filter === 'all' ? 'Waiting for events…' : 'No warnings or errors'} />
        ) : (
          shown.map((e) => (
            <div key={e.seq} className="flex gap-3 animate-fade-in">
              <span className="shrink-0 text-faint">{fmtTime(e.ts)}</span>
              {e.stage && <span className="w-14 shrink-0 truncate text-faint">{e.stage}</span>}
              <span className={`min-w-0 break-words ${LEVEL_STYLE[e.level]}`}>{e.message}</span>
            </div>
          ))
        )}
      </div>
    </Card>
  )
}
