import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api'
import type { ExecRun } from '../types'
import { Badge, Card, Empty, ErrorBox, Mono, SectionTitle, Spinner, fmtBytes, fmtNum } from './ui'

const STATUS: Record<ExecRun['status'], { label: string; tone: 'ok' | 'bad' | 'warn' | 'neutral' }> = {
  ok: { label: 'completed', tone: 'ok' },
  error: { label: 'failed', tone: 'bad' },
  timeout: { label: 'timed out', tone: 'warn' },
  not_runnable: { label: 'not runnable', tone: 'warn' },
  not_executed: { label: 'not executed', tone: 'neutral' },
}

const KEY_PACKAGES = /^(torch|torchvision|tensorflow\S*|jax|scikit-learn|numpy|scipy|pandas|transformers|xgboost|lightgbm)==/i

export default function Execution({ jobId }: { jobId: string }) {
  const { data, isError, error } = useQuery({ queryKey: ['stage', jobId, 'execute'], queryFn: () => api.executed(jobId) })
  const [showPkgs, setShowPkgs] = useState(false)
  if (isError) return <ErrorBox title="Could not load execution results" message={(error as Error).message} />
  if (!data) return <Spinner label="Loading runs…" />
  const keyPkgs = data.packages.filter((p) => KEY_PACKAGES.test(p))

  return (
    <Card className="p-5">
      <SectionTitle kicker="Stage 5 output" title="Experiment execution">
        <span className="text-xs text-muted">
          {data.runs.filter((r) => r.status === 'ok').length} of {data.runs.length} run(s) completed · isolated Docker,
          CPU-only, non-root, no network
        </span>
      </SectionTitle>

      {data.docker_available === false && (
        <ErrorBox title="Docker was not available" message="No experiment was executed. Start Docker Desktop and run the analysis again." />
      )}

      {data.image && (
        <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
          <Badge tone="accent">Python {data.python}</Badge>
          {keyPkgs.map((p) => (
            <Mono key={p} className="rounded bg-neutral-soft px-1.5 py-0.5">{p}</Mono>
          ))}
          <button type="button" onClick={() => setShowPkgs(!showPkgs)} className="text-accent hover:underline">
            {showPkgs ? 'hide' : `all ${data.packages.length} packages`}
          </button>
          <span className="font-mono text-[11px] text-faint">{data.image}</span>
        </div>
      )}
      {showPkgs && (
        <pre className="mb-4 max-h-48 overflow-auto rounded bg-neutral-soft px-3 py-2 font-mono text-[11.5px]">{data.packages.join('\n')}</pre>
      )}

      {data.runs.length === 0 ? (
        <Empty title="Nothing was executed">No claim had a runnable plan.</Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[42rem] text-sm">
            <thead>
              <tr className="border-b border-rule text-left text-xs text-faint">
                <th className="py-2 pr-3 font-medium">Run</th>
                <th className="py-2 pr-3 font-medium">Claims</th>
                <th className="py-2 pr-3 font-medium">Command</th>
                <th className="py-2 pr-3 font-medium">Status</th>
                <th className="py-2 pr-3 text-right font-medium">Time</th>
                <th className="py-2 pr-3 font-medium">Outputs</th>
                <th className="py-2 font-medium">Log</th>
              </tr>
            </thead>
            <tbody>
              {data.runs.map((r) => (
                <tr key={r.id} className="border-b border-rule/70 align-top">
                  <td className="py-2.5 pr-3 font-mono text-xs font-semibold">{r.id}</td>
                  <td className="py-2.5 pr-3 font-mono text-xs">{r.claim_ids.join(', ')}</td>
                  <td className="py-2.5 pr-3">
                    <Mono className="break-all">{r.command}</Mono>
                    {(r.attempts?.length ?? 0) > 1 && (
                      <div className="mt-0.5 text-[11px] text-warn">{r.attempts!.length} attempts (see deviations)</div>
                    )}
                  </td>
                  <td className="py-2.5 pr-3">
                    <Badge tone={STATUS[r.status].tone}>
                      {STATUS[r.status].label}
                      {r.exit_code != null && r.status !== 'ok' && ` · exit ${r.exit_code}`}
                    </Badge>
                    {r.reason && <div className="mt-1 max-w-[16rem] text-[11px] text-muted">{r.reason}</div>}
                    {r.network === 'on' && <div className="mt-1 text-[11px] text-warn">network was enabled</div>}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono text-xs tabular-nums">{r.seconds != null ? `${fmtNum(r.seconds, 1)}s` : '—'}</td>
                  <td className="py-2.5 pr-3 text-xs">
                    {r.outputs?.length ? (
                      r.outputs.map((o) => (
                        <div key={o.path} className="font-mono">
                          {o.path} <span className="text-faint">{o.skipped ?? fmtBytes(o.size)}</span>
                        </div>
                      ))
                    ) : (
                      <span className="text-faint">stdout only</span>
                    )}
                  </td>
                  <td className="py-2.5 text-xs">
                    {r.log_path ? (
                      <a href={`/api/jobs/${jobId}/runs/${r.id}/log`} target="_blank" rel="noreferrer" className="font-medium text-accent hover:underline">
                        full log ↗
                      </a>
                    ) : (
                      <span className="text-faint">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data.deviations.length > 0 && (
        <div className="mt-4 rounded-lg border border-warn/30 bg-warn-soft/50 px-4 py-3">
          <div className="text-sm font-semibold text-warn">Deviations made by ReproScope</div>
          <ul className="mt-1 space-y-0.5 text-sm text-warn">
            {data.deviations.map((d) => (
              <li key={d.id}>
                <Mono className="font-semibold">{d.id}</Mono> {d.run_id && <Mono>({d.run_id})</Mono>} {d.detail}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  )
}
