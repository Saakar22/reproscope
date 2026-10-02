import { Fragment, useState, type ReactNode } from 'react'
import type { ClaimReport, FindingOut, Report, RunOut } from '../../types'
import { Badge, Card, Empty, Mono, SectionTitle, VerdictBadge, fmtNum, fmtSigned, fmtWhen } from '../ui'
import RerunPanel from './RerunPanel'

export type EvidenceLinks = {
  /** URL for the paper at a page, or null when no PDF is available (fixture). */
  pdfAt: (page: number) => string | null
  /** Open the in-app paper viewer (null for the fixture). */
  openPaper: ((page: number, quote?: string | null) => void) | null
  /** Open the in-app repository file viewer (null for the fixture). */
  openFile: ((path: string, line?: number | null) => void) | null
  /** Set for real jobs: enables user-approved hypothesis reruns. */
  jobId: string | null
}

function FileRef({ file, line, links }: { file: string; line?: number | null; links: EvidenceLinks }) {
  const label = `${file}${line != null ? `:${line}` : ''}`
  return links.openFile ? (
    <button type="button" onClick={() => links.openFile!(file, line)} className="font-mono text-accent hover:underline"
      title="View this file with the line highlighted">
      {label}
    </button>
  ) : (
    <span className="font-mono">{label}</span>
  )
}

// ------------------------------------------------------------------ A. summary

export function ExecutiveSummary({ report }: { report: Report }) {
  const s = report.summary
  const tiles: { label: string; value: number; tone: string; hint?: string }[] = [
    { label: 'Claims', value: s.total_claims, tone: 'text-ink', hint: `${s.grounded_claims} grounded in the PDF` },
    { label: 'Reproduced', value: s.reproduced, tone: 'text-ok' },
    { label: 'Partial', value: s.partial, tone: 'text-warn' },
    { label: 'Not reproduced', value: s.not_reproduced, tone: 'text-bad' },
    { label: 'Inconclusive / not run', value: s.inconclusive + s.not_run + s.no_metric, tone: 'text-neutral',
      hint: `${s.inconclusive} inconclusive · ${s.not_run} not run · ${s.no_metric} no metric` },
    { label: 'Unverified', value: s.unverified, tone: 'text-info', hint: 'excluded from scoring' },
  ]
  const check = (v: boolean | null) =>
    v === null ? <Badge>unknown</Badge> : v ? <Badge tone="ok">✓ yes</Badge> : <Badge tone="bad">✕ no</Badge>
  return (
    <section>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {tiles.map((t) => (
          <Card key={t.label} className="px-4 py-3.5">
            <div className={`font-serif text-3xl font-semibold tabular-nums ${t.tone}`}>{t.value}</div>
            <div className="mt-0.5 text-xs font-medium text-muted">{t.label}</div>
            {t.hint && <div className="mt-0.5 text-[11px] text-faint">{t.hint}</div>}
          </Card>
        ))}
      </div>
      {s.better_than_reported > 0 && (
        <p className="mt-3 text-sm text-warn">
          {s.better_than_reported} claim(s) scored better than reported beyond tolerance. That is flagged for review (possible
          leakage or a different setup), not counted as reproduced.
        </p>
      )}
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-sm text-muted">
        <span className="flex items-center gap-1.5">Random seed set {check(report.checklist.seed_set)}</span>
        <span className="flex items-center gap-1.5">Dependencies pinned {check(report.checklist.deps_pinned)}</span>
        <span className="flex items-center gap-1.5">All hyperparameters stated {check(report.checklist.all_hparams_stated)}</span>
        <span className="flex items-center gap-1.5">Data split stated {check(report.checklist.split_stated)}</span>
      </div>
    </section>
  )
}

// ------------------------------------------------------------ B/C. claims

export function ClaimsTable({ report, links }: { report: Report; links: EvidenceLinks }) {
  const [open, setOpen] = useState<string | null>(report.claims.find((c) => c.comparison?.verdict !== 'REPRODUCED')?.id ?? null)
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[46rem] text-sm">
        <thead>
          <tr className="border-b border-rule text-left text-xs text-faint">
            <th className="py-2 pr-3 font-medium">Claim</th>
            <th className="py-2 pr-3 font-medium">Experiment</th>
            <th className="py-2 pr-3 font-medium">Metric</th>
            <th className="py-2 pr-3 text-right font-medium">Reported</th>
            <th className="py-2 pr-3 text-right font-medium">Obtained</th>
            <th className="py-2 pr-3 text-right font-medium">Δ</th>
            <th className="py-2 pr-3 text-right font-medium">Tolerance</th>
            <th className="py-2 pr-3 font-medium">Verdict</th>
            <th className="py-2" />
          </tr>
        </thead>
        <tbody>
          {report.claims.map((c) => {
            const cmp = c.comparison
            const isOpen = open === c.id
            // values on a 0–1 scale need more decimals or real differences round away
            const dp = Math.abs(c.reported_value) < 1 ? 4 : 2
            return (
              <Fragment key={c.id}>
                <tr
                  onClick={() => setOpen(isOpen ? null : c.id)}
                  className={`cursor-pointer border-b border-rule/70 align-top transition-colors hover:bg-paper ${!c.grounded ? 'text-faint' : ''}`}
                >
                  <td className="py-2.5 pr-3 font-mono text-xs font-semibold">{c.id}</td>
                  <td className="py-2.5 pr-3">
                    <div className="font-medium">{c.experiment}</div>
                    <div className="text-xs text-faint">{[c.dataset, c.split].filter(Boolean).join(' · ')}</div>
                  </td>
                  <td className="py-2.5 pr-3">
                    {c.metric}
                    {c.unit === 'percent' && <span className="text-faint"> %</span>}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                    {fmtNum(c.reported_value, dp)}
                    {c.reported_std != null && <span className="text-faint"> ±{fmtNum(c.reported_std, dp)}</span>}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">{fmtNum(cmp?.obtained, dp)}</td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums">{fmtSigned(cmp?.abs_delta, dp)}</td>
                  <td className="py-2.5 pr-3 text-right font-mono tabular-nums text-muted">
                    {cmp?.tolerance != null ? `±${fmtNum(cmp.tolerance, dp)}` : '—'}
                  </td>
                  <td className="py-2.5 pr-3">{cmp && <VerdictBadge verdict={cmp.verdict} />}</td>
                  <td className="py-2.5 text-right text-faint" aria-hidden>{isOpen ? '▾' : '▸'}</td>
                </tr>
                {isOpen && (
                  <tr className="border-b border-rule bg-paper/60">
                    <td colSpan={9} className="px-3 py-4">
                      <ClaimDetail claim={c} report={report} links={links} />
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-[11px] font-semibold tracking-wider text-faint uppercase">{label}</div>
      <div className="text-sm">{children}</div>
    </div>
  )
}

function GroundingMeter({ claim }: { claim: ClaimReport }) {
  const score = claim.grounding.quote_score ?? 0
  return (
    <div className="flex items-center gap-3">
      <div className="h-1.5 w-28 overflow-hidden rounded-full bg-rule">
        <div className={`h-full ${claim.grounded ? 'bg-ok' : 'bg-info'}`} style={{ width: `${Math.min(100, score)}%` }} />
      </div>
      <span className="text-xs text-muted">
        quote match {fmtNum(score, 0)} · number {claim.grounding.number_found ? 'found' : 'not found'}
        {claim.grounding.page_found && ` on p.${claim.grounding.page_found}`}
      </span>
      {claim.grounded ? <Badge tone="ok">Grounded</Badge> : <Badge tone="info">Unverified</Badge>}
    </div>
  )
}

function ClaimDetail({ claim, report, links }: { claim: ClaimReport; report: Report; links: EvidenceLinks }) {
  const runs = report.runs.filter((r) => claim.run_ids.includes(r.id))
  const findings = report.findings.filter((f) => claim.finding_ids.includes(f.id))
  const deviations = report.deviations.filter((d) => claim.deviation_ids.includes(d.id))
  const pdf = links.pdfAt(claim.page)
  const hp = Object.entries(claim.hyperparameters)
  return (
    <div className="grid grid-cols-1 gap-5 animate-fade-in lg:grid-cols-2">
      <div className="space-y-4">
        <Field label={`Paper evidence · p.${claim.page}${claim.source ? ` · ${claim.source}` : ''}`}>
          <blockquote className="border-l-2 border-accent/40 pl-3 font-serif text-[15px] italic">“{claim.quote}”</blockquote>
          <div className="mt-2 flex flex-wrap gap-3">
            {links.openPaper && (
              <button type="button" onClick={() => links.openPaper!(claim.page, claim.quote)}
                className="text-xs font-medium text-accent hover:underline">
                View in the paper (p.{claim.page}, quote highlighted)
              </button>
            )}
            {pdf ? (
              <a href={pdf} target="_blank" rel="noreferrer" className="text-xs text-muted hover:underline">
                open PDF in a new tab ↗
              </a>
            ) : (
              <span className="text-xs text-faint">No PDF attached to this sample report.</span>
            )}
          </div>
        </Field>
        <Field label="Grounding check">
          <GroundingMeter claim={claim} />
          {!claim.grounded && (
            <p className="mt-1.5 text-xs text-info">
              The extracted quote or number could not be found in the PDF text, so this claim is not scored.
            </p>
          )}
        </Field>
        <Field label="Hyperparameters stated in the paper">
          {hp.length ? (
            <div className="flex flex-wrap gap-1.5">
              {hp.map(([k, v]) => (
                <Mono key={k} className="rounded bg-neutral-soft px-1.5 py-0.5">{k}={String(v)}</Mono>
              ))}
            </div>
          ) : (
            <span className="text-faint">None stated.</span>
          )}
        </Field>
        {claim.origin !== 'llm' && (
          <p className="text-xs text-warn">
            Extracted by: {claim.origin === 'dev_sample' ? 'development sample (no LLM key)' : claim.origin}
          </p>
        )}
      </div>
      <div className="space-y-4">
        {claim.comparison && (
          <Field label="Comparison">
            <div className="flex flex-wrap items-center gap-2">
              <VerdictBadge verdict={claim.comparison.verdict} />
              {claim.comparison.tolerance_basis && <span className="text-xs text-muted">tolerance: {claim.comparison.tolerance_basis}</span>}
            </div>
            {claim.comparison.note && <p className="mt-1.5 text-sm text-muted">{claim.comparison.note}</p>}
          </Field>
        )}
        {runs.length > 0 && (
          <Field label="Execution">
            {runs.map((r) => (
              <div key={r.id} className="mb-1.5 flex flex-wrap items-center gap-2">
                <Mono className="font-semibold">{r.id}</Mono>
                <Badge tone={r.kind === 'hypothesis' ? 'info' : 'neutral'}>{r.kind}</Badge>
                <RunStatus run={r} />
                <Mono className="break-all text-muted">{r.command}</Mono>
              </div>
            ))}
          </Field>
        )}
        {findings.length > 0 && (
          <Field label="Linked findings">
            <ul className="space-y-1">
              {findings.map((f) => (
                <li key={f.id}>
                  <a href={`#finding-${f.id}`} className="hover:underline">
                    <Mono className="font-semibold text-accent">{f.id}</Mono> <span className="text-muted">{f.summary}</span>
                  </a>
                </li>
              ))}
            </ul>
          </Field>
        )}
        {deviations.length > 0 && (
          <Field label="Deviations made by ReproScope">
            {deviations.map((d) => (
              <div key={d.id} className="text-sm text-warn">
                <Mono className="font-semibold">{d.id}</Mono> {d.detail}
              </div>
            ))}
          </Field>
        )}
      </div>
      {links.jobId && claim.comparison?.obtained != null && (
        <div className="lg:col-span-2">
          <RerunPanel jobId={links.jobId} claim={claim} report={report} />
        </div>
      )}
    </div>
  )
}

function RunStatus({ run }: { run: RunOut }) {
  const tone = run.status === 'ok' ? 'ok' : run.status === 'timeout' || run.status === 'not_runnable' ? 'warn' : run.status === 'error' ? 'bad' : 'neutral'
  return <Badge tone={tone}>{run.status}{run.exit_code != null && ` · exit ${run.exit_code}`}</Badge>
}

// ------------------------------------------------------------ D. findings

const SEVERITY_TONE = { high: 'bad', medium: 'warn', low: 'neutral' } as const

export function FindingsList({ findings, links }: { findings: FindingOut[]; links: EvidenceLinks }) {
  if (!findings.length) return <Empty title="No discrepancies detected" />
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      {findings.map((f) => (
        <Card key={f.id} className="scroll-mt-24 p-4 target:ring-2 target:ring-accent">
          <div id={`finding-${f.id}`} className="flex flex-wrap items-center gap-2">
            <Mono className="font-semibold">{f.id}</Mono>
            <Badge tone={SEVERITY_TONE[f.severity]}>{f.severity}</Badge>
            <span className="font-mono text-xs text-muted">{f.rule}</span>
            {(f.claim_ids?.length ? f.claim_ids : f.claim_id ? [f.claim_id] : []).length > 0 && (
              <span className="ml-auto text-xs text-faint">
                {(f.claim_ids?.length ? f.claim_ids : [f.claim_id!]).join(', ')}
              </span>
            )}
          </div>
          <p className="mt-2 text-sm font-medium">{f.summary}</p>
          {f.impact && <p className="mt-1 text-xs text-muted">{f.impact}</p>}
          <div className="mt-3 space-y-2">
            {f.paper_evidence?.quote && (
              <div className="text-xs">
                <span className="text-faint">Paper p.{f.paper_evidence.page}: </span>
                <span className="font-serif italic">“{f.paper_evidence.quote}”</span>
                {f.paper_evidence.page != null && links.openPaper && (
                  <button type="button" onClick={() => links.openPaper!(f.paper_evidence!.page!, f.paper_evidence!.quote)}
                    className="ml-1.5 text-accent hover:underline">view</button>
                )}
              </div>
            )}
            {f.repo_evidence.map((e, i) => (
              <div key={i} className="overflow-hidden rounded-md border border-rule">
                <div className="bg-neutral-soft px-2.5 py-1 text-[11px] text-muted">
                  <FileRef file={e.file} line={e.line} links={links} />
                </div>
                {e.snippet && <pre className="overflow-x-auto px-2.5 py-1.5 font-mono text-[12px]">{e.snippet}</pre>}
              </div>
            ))}
            {f.run_evidence?.line_text && (
              <div className="overflow-hidden rounded-md border border-rule">
                <div className="bg-neutral-soft px-2.5 py-1 font-mono text-[11px] text-muted">
                  log {f.run_evidence.run_id} line {f.run_evidence.line_no}
                </div>
                <pre className="overflow-x-auto px-2.5 py-1.5 font-mono text-[12px] text-bad">{f.run_evidence.line_text}</pre>
              </div>
            )}
            {Array.isArray((f.run_evidence as { values?: number[] } | null)?.values) && (
              <div className="text-xs text-muted">
                Measured across runs{' '}
                {((f.run_evidence as { run_ids?: string[] }).run_ids ?? []).map((r) => (
                  <Mono key={r} className="mr-1">{r}</Mono>
                ))}
                :{' '}
                <Mono>{(f.run_evidence as { values: number[] }).values.map((v) => fmtNum(v, 4)).join(', ')}</Mono>
              </div>
            )}
          </div>
          {f.test && <HypothesisTestBox test={f.test} />}
        </Card>
      ))}
    </div>
  )
}

const OUTCOME = {
  supports: { label: 'supports hypothesis', tone: 'ok' },
  partially_supports: { label: 'partially supports', tone: 'warn' },
  does_not_support: { label: 'does not support', tone: 'neutral' },
  inconclusive: { label: 'inconclusive', tone: 'neutral' },
  not_applicable: { label: 'already within tolerance', tone: 'neutral' },
} as const

function HypothesisTestBox({ test }: { test: NonNullable<FindingOut['test']> }) {
  const tol = test.tolerance ?? null
  const max = Math.max(Math.abs(test.gap_before ?? 0), Math.abs(test.gap_after ?? 0), tol ?? 0) || 1
  const bar = (v: number | undefined, label: string) => (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-12 text-faint">{label}</span>
      <div className="relative h-2 flex-1 rounded-full bg-rule">
        {tol != null && <div className="absolute inset-y-0 left-0 rounded-full bg-ok-soft" style={{ width: `${(tol / max) * 100}%` }} />}
        <div
          className={`absolute inset-y-0 left-0 rounded-full ${v != null && tol != null && Math.abs(v) <= tol ? 'bg-mark-ok' : 'bg-mark-bad'}`}
          style={{ width: `${(Math.abs(v ?? 0) / max) * 100}%` }}
        />
      </div>
      <Mono className="w-12 text-right">{fmtSigned(v)}</Mono>
    </div>
  )
  return (
    <div className="mt-3 rounded-lg border border-info/25 bg-info-soft/50 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs font-semibold text-info">
        Hypothesis rerun {(test.run_ids ?? (test.run_id ? [test.run_id] : [])).map((r) => <Mono key={r}>{r}</Mono>)}
        {test.override && <Mono className="font-normal">{Object.entries(test.override).map(([k, v]) => `${k} ${v}`).join(' ')}</Mono>}
        {test.supports_hypothesis && (
          <Badge tone={OUTCOME[test.supports_hypothesis].tone}>{OUTCOME[test.supports_hypothesis].label}</Badge>
        )}
      </div>
      {bar(test.gap_before, 'before')}
      <div className="h-1" />
      {bar(test.gap_after, 'after')}
      {test.statement && <p className="mt-2 text-xs text-muted">{test.statement}</p>}
    </div>
  )
}

// ------------------------------------------------------------ E. runs & logs

export function RunsTable({ runs, logUrl }: { runs: RunOut[]; logUrl: (runId: string) => string | null }) {
  if (!runs.length) return <Empty title="No experiments were run" />
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[44rem] text-sm">
        <thead>
          <tr className="border-b border-rule text-left text-xs text-faint">
            <th className="py-2 pr-3 font-medium">Run</th>
            <th className="py-2 pr-3 font-medium">Claim</th>
            <th className="py-2 pr-3 font-medium">Command</th>
            <th className="py-2 pr-3 font-medium">Environment</th>
            <th className="py-2 pr-3 font-medium">Status</th>
            <th className="py-2 pr-3 text-right font-medium">Time</th>
            <th className="py-2 font-medium">Log</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const url = r.log_available ? logUrl(r.id) : null
            return (
              <tr key={r.id} className="border-b border-rule/70 align-top">
                <td className="py-2.5 pr-3">
                  <Mono className="font-semibold">{r.id}</Mono>
                  <div className="text-[11px] text-faint">{r.kind}</div>
                </td>
                <td className="py-2.5 pr-3 font-mono text-xs">{r.claim_id}</td>
                <td className="py-2.5 pr-3">
                  <Mono className="break-all">{r.command}</Mono>
                  {Object.keys(r.overrides).length > 0 && (
                    <div className="mt-0.5 text-[11px] text-info">override: {JSON.stringify(r.overrides)}</div>
                  )}
                </td>
                <td className="py-2.5 pr-3 text-xs text-muted">
                  {r.image_tag && <Mono className="block">{r.image_tag}</Mono>}
                  {Object.entries(r.environment).map(([k, v]) => `${k}=${String(v)}`).join(' · ')}
                </td>
                <td className="py-2.5 pr-3"><RunStatus run={r} /></td>
                <td className="py-2.5 pr-3 text-right font-mono text-xs tabular-nums">{r.seconds != null ? `${fmtNum(r.seconds, 1)}s` : '—'}</td>
                <td className="py-2.5 text-xs">
                  {url ? (
                    <a href={url} target="_blank" rel="noreferrer" className="font-medium text-accent hover:underline">view ↗</a>
                  ) : (
                    <span className="text-faint">not stored</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ------------------------------------------------------------ F. explanations

export function Explanations({ report }: { report: Report }) {
  const withHyp = report.claims.filter((c) => c.hypotheses.length || c.diagnosis_origin)
  const findingIds = new Set(report.findings.map((f) => f.id))
  if (!withHyp.length)
    return (
      <Empty title="No explanations needed">
        Explanations are generated only for claims that did not reproduce (or had no metric), and only from collected
        evidence.
      </Empty>
    )
  return (
    <div className="space-y-4">
      {withHyp.map((c) => (
        <Card key={c.id} className="p-4">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Mono className="font-semibold">{c.id}</Mono>
            <span className="text-sm font-medium">{c.experiment}</span>
            {c.comparison && <VerdictBadge verdict={c.comparison.verdict} />}
            {c.diagnosis_origin === 'dev_sample' && <Badge tone="warn">development sample, not LLM output</Badge>}
          </div>
          {c.hypotheses.length === 0 && (
            <p className="text-sm text-muted">
              <strong>Unexplained.</strong> The collected evidence does not explain this result, so no cause is
              suggested{c.diagnosis_origin === 'error' ? ' (the explanation service was unavailable)' : ''}.
            </p>
          )}
          <ol className="space-y-2">
            {c.hypotheses.map((h, i) => (
              <li key={i} className="flex gap-3 text-sm">
                <span className="mt-0.5 font-mono text-xs text-faint">{i + 1}.</span>
                <div>
                  <div>{h.text}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs">
                    <Badge tone={h.likelihood === 'high' ? 'accent' : 'neutral'}>{h.likelihood} likelihood</Badge>
                    {h.evidence_ids.map((id) =>
                      findingIds.has(id) ? (
                        <a key={id} href={`#finding-${id}`} className="rounded bg-accent-soft px-1.5 py-0.5 font-mono text-accent hover:underline">{id}</a>
                      ) : (
                        <Mono key={id} className="rounded bg-neutral-soft px-1.5 py-0.5">{id}</Mono>
                      ),
                    )}
                    {h.suggested_check && <span className="text-muted">· next check: {h.suggested_check}</span>}
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </Card>
      ))}
    </div>
  )
}

export function ReportMeta({ report }: { report: Report }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted">
      <span className="truncate">{report.job.repo_url ?? report.job.repo_filename ?? '—'}</span>
      {report.job.repo_commit && <Mono className="text-faint">@{report.job.repo_commit.slice(0, 10)}</Mono>}
      <span>generated {fmtWhen(report.generated_at)}</span>
      {report.job.llm_mode === 'dev' && <Badge tone="warn">LLM dev mode</Badge>}
    </div>
  )
}

export { SectionTitle }
