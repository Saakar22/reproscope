import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState, type FormEvent } from 'react'
import { api } from '../../api'
import type { ClaimReport, Report, Rerun } from '../../types'
import { Badge, ErrorBox, Mono, Spinner, fmtNum, fmtSigned } from '../ui'

const OUTCOME = {
  supports: { label: 'supports hypothesis', tone: 'ok' },
  partially_supports: { label: 'partially supports', tone: 'warn' },
  does_not_support: { label: 'does not support', tone: 'neutral' },
  inconclusive: { label: 'inconclusive', tone: 'neutral' },
  not_applicable: { label: 'already within tolerance', tone: 'neutral' },
} as const

type Row = { flag: string; value: string }

// Config/output-file flags can't be overridden (same rule as the backend's CONFIG_FLAG_RE / OUTPUT_FLAG_RE).
const FILE_FLAG =
  /^--?(config|cfg|config[_-]file|config[_-]path|conf|c|out|output|outdir|out_dir|output_dir|output-dir|save|save_dir|save-dir|log|logdir|log_dir|log-dir|results|result_file|results_file|metrics_file)$/i

/** User-approved hypothesis rerun: change up to three CLI values of this claim's command and rerun it. */
export default function RerunPanel({ jobId, claim, report }: { jobId: string; claim: ClaimReport; report: Report }) {
  const qc = useQueryClient()
  const plan = report.plans.find((p) => p.claim_id === claim.id)
  const flags = useMemo(
    () =>
      report.repo_facts.filter(
        (f) => f.kind === 'arg' && f.file === plan?.script && f.name.startsWith('-') && !FILE_FLAG.test(f.name),
      ),
    [report.repo_facts, plan?.script],
  )
  const suggested: Row[] = Object.entries(plan?.paper_overrides ?? {}).map(([flag, value]) => ({ flag, value: String(value) }))
  const [rows, setRows] = useState<Row[]>(suggested.length ? suggested.slice(0, 1) : [{ flag: flags[0]?.name ?? '', value: '' }])
  const [note, setNote] = useState('')

  const list = useQuery({
    queryKey: ['reruns', jobId],
    queryFn: () => api.reruns(jobId),
    refetchInterval: (q) => (q.state.data?.some((r) => r.status === 'queued' || r.status === 'running') ? 1500 : false),
  })
  const start = useMutation({
    mutationFn: () =>
      api.startRerun(jobId, {
        claim_id: claim.id,
        overrides: Object.fromEntries(rows.filter((r) => r.flag && r.value).map((r) => [r.flag, r.value.trim()])),
        note: note || undefined,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['reruns', jobId] })
    },
  })

  if (!plan || !plan.command || flags.length === 0) return null
  const mine = (list.data ?? []).filter((r) => r.claim_ids.includes(claim.id))
  const busy = (list.data ?? []).some((r) => r.status === 'queued' || r.status === 'running')
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (rows.some((r) => r.flag && r.value)) start.mutate()
  }

  return (
    <div className="rounded-lg border border-info/25 bg-info-soft/30 p-4">
      <div className="mb-1 text-[11px] font-semibold tracking-wider text-info uppercase">Test a hypothesis</div>
      <p className="mb-3 text-xs text-muted">
        Rerun <Mono>{plan.command}</Mono> with only the values you change, {claim.comparison?.seed_values.length ?? 1}{' '}
        time(s) like the original measurement, and compare against the same tolerance.
      </p>
      <form onSubmit={submit} className="space-y-2">
        {rows.map((row, i) => (
          <div key={i} className="flex flex-wrap items-center gap-2">
            <select
              value={row.flag}
              onChange={(e) => setRows(rows.map((r, j) => (j === i ? { ...r, flag: e.target.value } : r)))}
              className="rounded-md border border-rule bg-surface px-2 py-1 font-mono text-sm"
              aria-label="Flag"
            >
              {flags.map((f) => (
                <option key={f.id} value={f.name}>
                  {f.name} (repo: {f.value ?? 'None'})
                </option>
              ))}
            </select>
            <input
              value={row.value}
              onChange={(e) => setRows(rows.map((r, j) => (j === i ? { ...r, value: e.target.value } : r)))}
              placeholder="new value"
              className="w-32 rounded-md border border-rule px-2 py-1 font-mono text-sm"
              aria-label="Value"
            />
            {rows.length > 1 && (
              <button type="button" onClick={() => setRows(rows.filter((_, j) => j !== i))} className="text-xs text-muted hover:text-bad">
                remove
              </button>
            )}
          </div>
        ))}
        <div className="flex flex-wrap items-center gap-3">
          {rows.length < 3 && (
            <button type="button" onClick={() => setRows([...rows, { flag: flags[0].name, value: '' }])} className="text-xs text-accent hover:underline">
              + change another value
            </button>
          )}
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="why (optional)" maxLength={300}
            className="min-w-0 flex-1 rounded-md border border-rule px-2 py-1 text-sm" />
          <button type="submit" disabled={start.isPending || busy}
            className="rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-white hover:bg-accent/90 disabled:opacity-50">
            {busy ? 'A rerun is running…' : 'Run'}
          </button>
        </div>
        {suggested.length > 0 && (
          <p className="text-[11px] text-muted">
            Prefilled with the paper's stated value where it differs from the repository. Changing one value at a time keeps
            the result attributable.
          </p>
        )}
        {start.isError && <ErrorBox title="Rerun not started" message={(start.error as Error).message} />}
      </form>

      {mine.length > 0 && (
        <ul className="mt-4 space-y-2">
          {mine.map((r) => (
            <RerunItem key={r.id} r={r} claimId={claim.id} jobId={jobId} />
          ))}
        </ul>
      )}
    </div>
  )
}

function RerunItem({ r, claimId, jobId }: { r: Rerun; claimId: string; jobId: string }) {
  const res = r.results[claimId]
  return (
    <li className="rounded-md border border-rule bg-surface px-3 py-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Mono className="font-semibold">{r.id}</Mono>
        <Mono className="text-muted">{Object.entries(r.overrides).map(([k, v]) => `${k} ${v}`).join(' ')}</Mono>
        {r.status === 'running' || r.status === 'queued' ? (
          <Spinner label={`${r.status}… ${r.run_ids.length}/${r.repeats}`} />
        ) : r.status === 'failed' ? (
          <Badge tone="bad">failed</Badge>
        ) : res ? (
          <Badge tone={OUTCOME[res.outcome].tone}>{OUTCOME[res.outcome].label}</Badge>
        ) : null}
        {r.run_ids.map((id) => (
          <a key={id} href={`/api/jobs/${jobId}/runs/${id}/log`} target="_blank" rel="noreferrer" className="font-mono text-xs text-accent hover:underline">
            {id} log
          </a>
        ))}
      </div>
      {res?.gap_after != null && (
        <div className="mt-1 font-mono text-xs text-muted">
          gap {fmtSigned(res.gap_before, 4)} → {fmtSigned(res.gap_after, 4)} (±{fmtNum(res.tolerance, 4)}) · values {res.values.map((v) => fmtNum(v, 4)).join(', ')}
        </div>
      )}
      {res && <p className="mt-1 text-xs text-muted">{res.statement}</p>}
      {r.error && <p className="mt-1 text-xs text-bad">{r.error}</p>}
      {r.note && <p className="mt-1 text-[11px] text-faint">Note: {r.note}</p>}
    </li>
  )
}
