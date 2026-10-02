import { useQuery } from '@tanstack/react-query'
import { api } from '../api'
import type { ParamDiff, PlanItem } from '../types'
import { Badge, Card, Empty, ErrorBox, Mono, SectionTitle, Spinner } from './ui'

const SOURCE_LABEL: Record<string, string> = {
  readme: 'README',
  entrypoint_defaults: 'script defaults',
  repo_config: 'repo config',
}

function DiffRow({ d }: { d: ParamDiff }) {
  const tone = d.status === 'mismatch' ? 'bad' : d.status === 'match' ? 'ok' : 'neutral'
  return (
    <tr className="border-t border-rule/70 align-top">
      <td className="py-1.5 pr-3 font-mono">{d.param}</td>
      <td className="py-1.5 pr-3">
        <Mono>{d.paper_value}</Mono>
        <div className="font-serif text-[11px] italic text-faint">“{d.paper_quote}”</div>
      </td>
      <td className="py-1.5 pr-3">
        {d.repo_value != null ? <Mono>{d.repo_value}</Mono> : <span className="text-faint">not found</span>}
        {d.repo_file && (
          <div className="font-mono text-[11px] text-faint">
            {d.repo_source} · {d.repo_file}
            {d.repo_line ? `:${d.repo_line}` : ''}
          </div>
        )}
      </td>
      <td className="py-1.5">
        <Badge tone={tone}>{d.status === 'not_in_repo' ? 'not in repo' : d.status}</Badge>
        {d.testable && <div className="mt-0.5 text-[11px] text-info">rerun candidate ({d.flag})</div>}
      </td>
    </tr>
  )
}

function PlanCard({ p }: { p: PlanItem }) {
  return (
    <div className={`rounded-lg border p-3.5 ${p.runnable ? 'border-rule' : 'border-warn/40 bg-warn-soft/30'}`}>
      <div className="flex flex-wrap items-center gap-2">
        <Mono className="font-semibold">{p.claim_id}</Mono>
        {p.runnable ? <Badge tone="ok">runnable</Badge> : <Badge tone="warn">will not run</Badge>}
        {p.run_group != null && <Badge tone="neutral" title="Claims in the same group share one execution">run #{p.run_group}</Badge>}
        {p.source && (
          <Badge tone="accent" title={p.source_ref ?? ''}>
            {SOURCE_LABEL[p.source] ?? p.source}
            {p.source_ref && ` · ${p.source_ref}`}
          </Badge>
        )}
        {p.confidence && <span className="text-[11px] text-faint">{p.confidence} confidence</span>}
        {p.retried && <Badge tone="info" title="The first plan failed validation and was corrected">retried</Badge>}
        {p.origin === 'dev_sample' && <Badge tone="warn">dev-mode heuristic</Badge>}
      </div>
      {p.command && <pre className="mt-2 overflow-x-auto rounded bg-neutral-soft px-2.5 py-1.5 font-mono text-[12.5px]">{p.command}</pre>}
      {p.extra_args.length > 0 && (
        <div className="mt-1 text-[11px] text-info">added to select the experiment: {p.extra_args.join(' ')}</div>
      )}
      {!p.runnable && p.reason && <p className="mt-2 text-sm text-warn">{p.reason}</p>}
      {p.validation.warnings.map((w) => (
        <p key={w} className="mt-1 text-xs text-warn">⚠ {w}</p>
      ))}
      {p.overrides.length > 0 && (
        <table className="mt-3 w-full text-xs">
          <thead>
            <tr className="text-left text-faint">
              <th className="pb-1 font-medium">Parameter</th>
              <th className="pb-1 font-medium">Paper says</th>
              <th className="pb-1 font-medium">Repository uses</th>
              <th className="pb-1 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {p.overrides.map((d) => (
              <DiffRow key={d.param} d={d} />
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function RunPlans({ jobId }: { jobId: string }) {
  const { data, isError, error } = useQuery({ queryKey: ['stage', jobId, 'plan'], queryFn: () => api.planned(jobId) })
  if (isError) return <ErrorBox title="Could not load run plans" message={(error as Error).message} />
  if (!data) return <Spinner label="Loading run plans…" />
  const runnable = data.plans.filter((p) => p.runnable).length
  const mismatches = data.plans.flatMap((p) => p.overrides).filter((d) => d.status === 'mismatch').length
  return (
    <Card className="p-5">
      <SectionTitle kicker="Stage 4 output" title="Run plans">
        <span className="text-xs text-muted">
          {runnable} of {data.plans.length} runnable · {data.groups?.length ?? 0} distinct command(s) ·{' '}
          {data.candidates.length} candidate(s) from the repo
          {mismatches > 0 && ` · ${mismatches} paper/repo difference(s), kept out of the primary run`}
        </span>
      </SectionTitle>
      {data.plans.length === 0 ? (
        <Empty title="Nothing to plan">No grounded claims, or no runnable command in the repository.</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {data.plans.map((p) => (
            <PlanCard key={p.id} p={p} />
          ))}
        </div>
      )}
      {data.skipped_unverified.length > 0 && (
        <p className="mt-3 text-xs text-info">
          Not planned (unverified claims, excluded from scoring): {data.skipped_unverified.join(', ')}
        </p>
      )}
    </Card>
  )
}
