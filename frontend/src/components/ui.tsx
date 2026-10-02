import type { ReactNode } from 'react'
import type { JobStatus, StageStatus, Verdict } from '../types'

type Tone = 'ok' | 'warn' | 'bad' | 'neutral' | 'info' | 'accent'

const toneClass: Record<Tone, string> = {
  ok: 'bg-ok-soft text-ok',
  warn: 'bg-warn-soft text-warn',
  bad: 'bg-bad-soft text-bad',
  neutral: 'bg-neutral-soft text-neutral',
  info: 'bg-info-soft text-info',
  accent: 'bg-accent-soft text-accent',
}

export function Badge({ tone = 'neutral', children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold tracking-wide whitespace-nowrap ${toneClass[tone]}`}
    >
      {children}
    </span>
  )
}

export const VERDICT_META: Record<Verdict, { label: string; tone: Tone; icon: string; help: string }> = {
  REPRODUCED: { label: 'Reproduced', tone: 'ok', icon: '✓', help: 'Obtained value is within tolerance of the reported value.' },
  PARTIAL: { label: 'Partial', tone: 'warn', icon: '≈', help: 'Outside tolerance but within 3× tolerance.' },
  NOT_REPRODUCED: { label: 'Not reproduced', tone: 'bad', icon: '✕', help: 'Differs by more than 3× tolerance.' },
  INCONCLUSIVE: { label: 'Inconclusive', tone: 'neutral', icon: '?', help: 'A material deviation (e.g. reduced budget) prevents a definitive verdict.' },
  NOT_RUN: { label: 'Not run', tone: 'neutral', icon: '–', help: 'The experiment could not be executed.' },
  NO_METRIC: { label: 'No metric', tone: 'neutral', icon: '∅', help: 'The run finished but no metric value could be found.' },
  UNVERIFIED_CLAIM: { label: 'Unverified claim', tone: 'info', icon: '!', help: 'The claim could not be found in the PDF text; excluded from scoring.' },
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const m = VERDICT_META[verdict]
  return (
    <Badge tone={m.tone} title={m.help}>
      <span aria-hidden>{m.icon}</span>
      {m.label}
    </Badge>
  )
}

const JOB_STATUS: Record<JobStatus, { label: string; tone: Tone }> = {
  queued: { label: 'Queued', tone: 'neutral' },
  running: { label: 'Running', tone: 'info' },
  done: { label: 'Done', tone: 'ok' },
  failed: { label: 'Failed', tone: 'bad' },
  incomplete: { label: 'Incomplete', tone: 'warn' },
}

export function JobStatusBadge({ status }: { status: JobStatus }) {
  const m = JOB_STATUS[status] ?? { label: status, tone: 'neutral' as Tone }
  return (
    <Badge tone={m.tone}>
      {status === 'running' && <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse-dot" />}
      {m.label}
    </Badge>
  )
}

export function StageIcon({ status }: { status: StageStatus }) {
  const base = 'flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold'
  switch (status) {
    case 'done':
      return <span className={`${base} bg-ok text-white`} aria-label="done">✓</span>
    case 'running':
      return (
        <span className={`${base} border-2 border-info`} aria-label="running">
          <span className="h-2 w-2 rounded-full bg-info animate-pulse-dot" />
        </span>
      )
    case 'failed':
      return <span className={`${base} bg-bad text-white`} aria-label="failed">✕</span>
    case 'skipped':
      return <span className={`${base} border border-dashed border-faint text-faint`} aria-label="skipped">–</span>
    case 'not_implemented':
      return <span className={`${base} border border-dashed border-warn text-warn`} aria-label="not implemented">·</span>
    default:
      return <span className={`${base} border border-rule text-faint`} aria-label="pending" />
  }
}

export const STAGE_STATUS_LABEL: Record<StageStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
  skipped: 'Skipped',
  not_implemented: 'Not implemented in this build',
}

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-rule bg-surface shadow-[0_1px_2px_rgba(20,24,32,0.04)] ${className}`}>{children}</div>
}

export function SectionTitle({ kicker, title, children }: { kicker?: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div>
        {kicker && <div className="text-[11px] font-semibold tracking-[0.14em] text-faint uppercase">{kicker}</div>}
        <h2 className="font-serif text-xl font-semibold text-ink">{title}</h2>
      </div>
      {children}
    </div>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-rule border-t-accent" />
      {label}
    </span>
  )
}

export function ErrorBox({ title = 'Something went wrong', message }: { title?: string; message: string }) {
  return (
    <div className="rounded-lg border border-bad/30 bg-bad-soft px-4 py-3 text-sm text-bad" role="alert">
      <div className="font-semibold">{title}</div>
      <div className="mt-0.5 break-words">{message}</div>
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-rule px-4 py-6 text-center">
      <div className="text-sm font-medium text-muted">{title}</div>
      {children && <div className="mt-1 text-xs text-faint">{children}</div>}
    </div>
  )
}

export function Mono({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <code className={`font-mono text-[12.5px] ${className}`}>{children}</code>
}

// ------------------------------------------------------------ formatting

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return Number(v.toFixed(digits)).toString()
}

export function fmtSigned(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined) return '—'
  const s = fmtNum(v, digits)
  return v > 0 ? `+${s}` : s
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 ** 2).toFixed(1)} MB`
}

export function fmtWhen(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

export function fmtTime(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString(undefined, { hour12: false })
}
