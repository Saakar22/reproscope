import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api'
import type { ScanFact } from '../types'
import { Badge, Card, Empty, ErrorBox, Mono, SectionTitle, Spinner } from './ui'

const KIND_LABEL: Record<string, string> = {
  entrypoint: 'Entry points',
  arg: 'CLI arguments',
  readme_cmd: 'README commands',
  config_value: 'Config values',
  requirement: 'Dependencies',
  python_version: 'Python version hints',
  seed_call: 'Random-seed calls',
  split_call: 'Data-split calls',
  metric_call: 'Metric calls',
  data_loader: 'Dataset loading',
  checkpoint: 'Checkpoint loads',
  output_write: 'Outputs & metric logging',
  gpu_only: 'GPU-only code',
  framework: 'Frameworks',
}

function where(f: { file: string | null; line: number | null }) {
  return f.file ? `${f.file}${f.line ? `:${f.line}` : ''}` : ''
}

function show(v: unknown): string {
  if (v === null || v === undefined) return 'None'
  if (typeof v === 'string') return JSON.stringify(v)
  return String(v)
}

export default function RepoCard({ jobId }: { jobId: string }) {
  const { data, isError, error } = useQuery({ queryKey: ['stage', jobId, 'scan'], queryFn: () => api.scanned(jobId) })
  const [openKind, setOpenKind] = useState<string | null>(null)
  if (isError) return <ErrorBox title="Could not load the repository scan" message={(error as Error).message} />
  if (!data) return <Spinner label="Loading repository facts…" />
  const s = data.summary
  const byKind = data.facts.reduce<Record<string, ScanFact[]>>((acc, f) => {
    ;(acc[f.kind] ??= []).push(f)
    return acc
  }, {})

  const yesNo = (v: boolean | null, yes: string, no: string) =>
    v === null ? <Badge>unknown</Badge> : v ? <Badge tone="ok">✓ {yes}</Badge> : <Badge tone="bad">✕ {no}</Badge>

  return (
    <Card className="p-5">
      <SectionTitle kicker="Stage 3 output" title="Repository card">
        <span className="text-xs text-muted">
          {data.facts.length} facts · {s.python_files} Python files · deterministic scan, no code executed
        </span>
      </SectionTitle>

      <div className="flex flex-wrap gap-2 text-xs">
        {s.frameworks.map((f) => (
          <Badge key={f} tone="accent">{f}</Badge>
        ))}
        {s.python_version && <Badge>Python {s.python_version}</Badge>}
        {yesNo(
          s.seed_set_in_entrypoints,
          'every entry point seeded',
          `unseeded: ${s.unseeded_entrypoints.join(', ')}`,
        )}
        {s.deps_total > 0 ? (
          yesNo(s.deps_all_pinned, 'all deps pinned', `${s.deps_pinned}/${s.deps_total} deps pinned`)
        ) : (
          <Badge tone="warn">no dependency file</Badge>
        )}
        {s.gpu_only_files.length > 0 && <Badge tone="warn">GPU-only: {s.gpu_only_files.join(', ')}</Badge>}
        {s.notebooks.length > 0 && <Badge tone="neutral">{s.notebooks.length} notebook(s), not executed</Badge>}
      </div>

      <div className="mt-5 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Entry points and CLI defaults</h3>
          {data.entrypoints.length === 0 ? (
            <Empty title="No entry point found" />
          ) : (
            <div className="space-y-3">
              {data.entrypoints.map((e) => (
                <div key={e.file} className="overflow-hidden rounded-lg border border-rule">
                  <div className="flex items-center justify-between bg-neutral-soft px-3 py-1.5">
                    <Mono className="font-semibold">{e.file}</Mono>
                    <span className="flex items-center gap-2 text-[11px] text-faint">
                      {e.seeded ? (
                        <Badge tone="ok" title={e.seed_calls.map((c) => `${c.call} ${c.file}:${c.line}`).join('\n')}>seeded</Badge>
                      ) : (
                        <Badge tone="bad" title="No random-seed call reachable from this script">no seed</Badge>
                      )}
                      {e.has_main_guard ? '__main__ guard' : 'name hint'}
                    </span>
                  </div>
                  {e.args.length === 0 ? (
                    <div className="px-3 py-2 text-xs text-faint">No CLI arguments</div>
                  ) : (
                    <table className="w-full text-xs">
                      <tbody>
                        {e.args.map((a) => (
                          <tr key={a.name} className="border-t border-rule/70">
                            <td className="px-3 py-1.5 font-mono">{a.name}</td>
                            <td className="px-3 py-1.5 font-mono text-accent">{show(a.default)}</td>
                            <td className="px-3 py-1.5 text-faint">{a.help ?? ''}</td>
                            <td className="px-3 py-1.5 text-right font-mono text-faint">:{a.line}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-5">
          <div>
            <h3 className="mb-2 text-sm font-semibold">Documented commands</h3>
            {data.readme_commands.length === 0 ? (
              <Empty title="No commands in the README" />
            ) : (
              <ul className="space-y-1.5">
                {data.readme_commands.map((c) => (
                  <li key={c.command} className="flex flex-wrap items-baseline gap-2">
                    <Mono className="rounded bg-neutral-soft px-1.5 py-0.5">{c.command}</Mono>
                    <span className="text-[11px] text-faint">{where(c)}</span>
                    {!c.script_exists && <Badge tone="warn">script not in repo</Badge>}
                  </li>
                ))}
              </ul>
            )}
          </div>
          {data.warnings.length > 0 && (
            <div>
              <h3 className="mb-1.5 text-sm font-semibold">Scan warnings</h3>
              <ul className="list-disc space-y-0.5 pl-5 text-xs text-warn">
                {data.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <div className="mt-6">
        <h3 className="mb-2 text-sm font-semibold">All facts by kind</h3>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(byKind).map(([kind, list]) => (
            <button
              key={kind}
              type="button"
              onClick={() => setOpenKind(openKind === kind ? null : kind)}
              className={`rounded-full border px-2.5 py-1 text-xs ${openKind === kind ? 'border-accent bg-accent-soft text-accent' : 'border-rule text-muted hover:text-ink'}`}
            >
              {KIND_LABEL[kind] ?? kind} <span className="font-mono">{list.length}</span>
            </button>
          ))}
        </div>
        {openKind && (
          <div className="mt-3 max-h-80 overflow-auto rounded-lg border border-rule">
            <table className="w-full text-xs">
              <tbody>
                {byKind[openKind].map((f) => (
                  <tr key={f.id} className="border-b border-rule/70 align-top last:border-0">
                    <td className="px-3 py-1.5 font-mono text-faint">{f.id}</td>
                    <td className="px-3 py-1.5 font-mono break-all">{f.name}</td>
                    <td className="px-3 py-1.5 font-mono break-all text-muted">{f.value ?? ''}</td>
                    <td className="px-3 py-1.5 text-right font-mono whitespace-nowrap text-faint">{where(f)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </Card>
  )
}
