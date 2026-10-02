import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import type { ParsedResult } from '../types'
import { Badge, Card, Empty, ErrorBox, Mono, SectionTitle, Spinner, fmtNum } from './ui'

const SOURCE: Record<string, { label: string; title: string }> = {
  file: { label: 'output file', title: 'Read from a structured file the run wrote' },
  regex: { label: 'log line', title: 'Matched a metric pattern in the run log' },
  llm: { label: 'LLM-located line', title: 'Located by the LLM, then verified: the line exists verbatim in the log and contains the value' },
}

function Evidence({ r, jobId }: { r: ParsedResult; jobId: string }) {
  const ev = r.evidence ?? {}
  return (
    <div className="space-y-0.5 text-xs">
      {ev.file ? (
        <div>
          <Mono>{ev.file}</Mono> <span className="text-faint">key</span> <Mono>{ev.key}</Mono>
        </div>
      ) : (
        <div>
          <a href={`/api/jobs/${jobId}/runs/${r.run_id}/log`} target="_blank" rel="noreferrer" className="text-accent hover:underline">
            {r.run_id} log line {ev.line_no}
          </a>
          <pre className="mt-0.5 overflow-x-auto rounded bg-neutral-soft px-2 py-1 font-mono text-[11.5px]">{ev.line_text}</pre>
        </div>
      )}
      {r.normalisation && <div className="text-muted">{r.normalisation}</div>}
    </div>
  )
}

export default function ParsedResults({ jobId }: { jobId: string }) {
  const parsed = useQuery({ queryKey: ['stage', jobId, 'parse'], queryFn: () => api.parsed(jobId) })
  const claims = useQuery({ queryKey: ['stage', jobId, 'extract'], queryFn: () => api.extracted(jobId) })
  if (parsed.isError) return <ErrorBox title="Could not load parsed results" message={(parsed.error as Error).message} />
  if (!parsed.data || !claims.data) return <Spinner label="Loading results…" />
  const byId = Object.fromEntries(claims.data.claims.map((c) => [c.id, c]))
  const results = parsed.data.results
  const found = results.filter((r) => r.status === 'found').length

  return (
    <Card className="p-5">
      <SectionTitle kicker="Stage 6 output" title="Obtained results">
        <span className="text-xs text-muted">
          {found} of {results.length} claim(s) have a measured value · verdicts are assigned in Stage 7
        </span>
      </SectionTitle>
      {results.length === 0 ? (
        <Empty title="No planned claims" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[46rem] text-sm">
            <thead>
              <tr className="border-b border-rule text-left text-xs text-faint">
                <th className="py-2 pr-3 font-medium">Claim</th>
                <th className="py-2 pr-3 font-medium">Metric</th>
                <th className="py-2 pr-3 text-right font-medium">Reported</th>
                <th className="py-2 pr-3 text-right font-medium">Obtained</th>
                <th className="py-2 pr-3 font-medium">Source</th>
                <th className="py-2 font-medium">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => {
                const c = byId[r.claim_id]
                const pct = r.unit === 'percent' ? '%' : ''
                return (
                  <tr key={r.claim_id} className="border-b border-rule/70 align-top">
                    <td className="py-2.5 pr-3">
                      <Mono className="font-semibold">{r.claim_id}</Mono>
                      <div className="text-xs text-faint">{c?.experiment}</div>
                    </td>
                    <td className="py-2.5 pr-3">{r.metric}</td>
                    <td className="py-2.5 pr-3 text-right font-mono tabular-nums">
                      {c ? `${c.reported_value}${pct}` : '—'}
                      {c?.reported_std != null && <span className="text-faint"> ±{c.reported_std}</span>}
                    </td>
                    <td className="py-2.5 pr-3 text-right font-mono font-semibold tabular-nums">
                      {r.status === 'found' ? `${fmtNum(r.value, 4)}${pct}` : '—'}
                    </td>
                    <td className="py-2.5 pr-3">
                      {r.status === 'found' && r.source ? (
                        <Badge tone={r.source === 'llm' ? 'info' : 'accent'} title={SOURCE[r.source].title}>
                          {SOURCE[r.source].label}
                        </Badge>
                      ) : r.status === 'no_metric' ? (
                        <Badge tone="warn">no metric</Badge>
                      ) : (
                        <Badge tone="neutral">not run</Badge>
                      )}
                    </td>
                    <td className="max-w-[24rem] py-2.5">
                      {r.status === 'found' ? (
                        <>
                          <Evidence r={r} jobId={jobId} />
                          {r.variant_note && <div className="mt-1 text-xs text-warn">⚠ {r.variant_note}</div>}
                        </>
                      ) : (
                        <div className="text-xs text-muted">
                          {r.reason}
                          {r.attempts && (
                            <ul className="mt-1 list-disc pl-4 text-faint">
                              {r.attempts.map((a) => (
                                <li key={a}>{a}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
