import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import FileDrop from '../components/FileDrop'
import { Card, Empty, ErrorBox, JobStatusBadge, Mono, SectionTitle, Spinner, fmtWhen } from '../components/ui'
import type { Job, JobOptions } from '../types'

const GIT_URL_RE = /^https:\/\/(github\.com|gitlab\.com|bitbucket\.org)\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+?(\.git)?\/?$/

function validateUrl(url: string): string | null {
  if (!url.trim()) return 'Enter the repository URL.'
  if (!GIT_URL_RE.test(url.trim())) return 'Use https://github.com/<owner>/<repo> (GitHub, GitLab or Bitbucket).'
  return null
}

export default function NewAnalysis() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const limits = health.data?.limits ?? { max_pdf_mb: 30, max_zip_mb: 100 }

  const [pdf, setPdf] = useState<File | null>(null)
  const [repoMode, setRepoMode] = useState<'git' | 'zip'>('git')
  const [repoUrl, setRepoUrl] = useState('')
  const [repoZip, setRepoZip] = useState<File | null>(null)
  const [options, setOptions] = useState<JobOptions>({
    main_results_only: false,
    max_claims: 6,
    enable_hypothesis_reruns: true,
    timeout_s: 600,
  })
  const [touched, setTouched] = useState(false)

  const pdfError = touched && !pdf ? 'Add the paper PDF.' : null
  const urlError = touched && repoMode === 'git' ? validateUrl(repoUrl) : null
  const zipError = touched && repoMode === 'zip' && !repoZip ? 'Add the repository ZIP.' : null

  const create = useMutation({
    mutationFn: () =>
      api.createJob({
        pdf: pdf!,
        repoUrl: repoMode === 'git' ? repoUrl.trim() : undefined,
        repoZip: repoMode === 'zip' ? repoZip! : undefined,
        options,
      }),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      navigate(`/jobs/${job.id}`)
    },
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    setTouched(true)
    const invalid = !pdf || (repoMode === 'git' ? validateUrl(repoUrl) : !repoZip)
    if (!invalid) create.mutate()
  }

  return (
    <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <section className="animate-fade-in">
        <h1 className="font-serif text-3xl font-semibold tracking-tight sm:text-4xl">Does this paper reproduce?</h1>
        <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-muted">
          Upload a machine-learning paper and its code. ReproScope extracts the reported results, runs the authors'
          experiments in an isolated CPU-only container, and shows where the numbers agree, where they don't, and the
          evidence for why.
        </p>

        <Card className="mt-6 p-5 sm:p-6">
          <form onSubmit={submit} noValidate className="space-y-5">
            <FileDrop
              label="Paper"
              hint={`PDF with a text layer · up to ${limits.max_pdf_mb} MB`}
              accept=".pdf"
              file={pdf}
              onFile={setPdf}
              maxMb={limits.max_pdf_mb}
              error={pdfError}
            />

            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <span className="text-sm font-medium">Code repository</span>
                <div className="inline-flex rounded-md border border-rule p-0.5 text-xs" role="tablist">
                  {(['git', 'zip'] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      role="tab"
                      aria-selected={repoMode === m}
                      onClick={() => setRepoMode(m)}
                      className={`rounded px-2.5 py-1 font-medium ${repoMode === m ? 'bg-accent text-white' : 'text-muted hover:text-ink'}`}
                    >
                      {m === 'git' ? 'Git URL' : 'ZIP upload'}
                    </button>
                  ))}
                </div>
              </div>
              {repoMode === 'git' ? (
                <div>
                  <input
                    type="url"
                    inputMode="url"
                    placeholder="https://github.com/owner/repo"
                    value={repoUrl}
                    onChange={(e) => setRepoUrl(e.target.value)}
                    aria-invalid={!!urlError}
                    className={`w-full rounded-lg border bg-surface px-3 py-2.5 font-mono text-sm outline-none focus:ring-2 focus:ring-accent/40 ${urlError ? 'border-bad/50' : 'border-rule'}`}
                  />
                  {urlError ? (
                    <div className="mt-1.5 text-xs text-bad">{urlError}</div>
                  ) : (
                    <div className="mt-1.5 text-xs text-faint">Shallow clone of the default branch; the commit hash is recorded.</div>
                  )}
                </div>
              ) : (
                <FileDrop
                  label=""
                  hint={`ZIP of the repository · up to ${limits.max_zip_mb} MB`}
                  accept=".zip"
                  file={repoZip}
                  onFile={setRepoZip}
                  maxMb={limits.max_zip_mb}
                  error={zipError}
                />
              )}
            </div>

            <details className="group rounded-lg border border-rule px-4 py-3">
              <summary className="cursor-pointer text-sm font-medium text-muted select-none group-open:text-ink">
                Analysis settings
              </summary>
              <div className="mt-4 grid gap-4 sm:grid-cols-2">
                <label className="text-sm">
                  <span className="mb-1 block text-muted">Max claims to analyse</span>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={options.max_claims}
                    onChange={(e) =>
                      setOptions({ ...options, max_claims: Math.min(20, Math.max(1, Number(e.target.value) || 1)) })
                    }
                    className="w-full rounded-md border border-rule px-2.5 py-1.5"
                  />
                </label>
                <label className="text-sm">
                  <span className="mb-1 block text-muted">Per-run time limit</span>
                  <select
                    value={options.timeout_s}
                    onChange={(e) => setOptions({ ...options, timeout_s: Number(e.target.value) })}
                    className="w-full rounded-md border border-rule bg-surface px-2.5 py-1.5"
                  >
                    {[120, 300, 600, 1200, 1800].map((s) => (
                      <option key={s} value={s}>
                        {s / 60} min
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={options.main_results_only}
                    onChange={(e) => setOptions({ ...options, main_results_only: e.target.checked })}
                  />
                  Main results only
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={options.enable_hypothesis_reruns}
                    onChange={(e) => setOptions({ ...options, enable_hypothesis_reruns: e.target.checked })}
                  />
                  Hypothesis-test reruns
                </label>
              </div>
            </details>

            {create.isError && <ErrorBox title="Could not start the analysis" message={(create.error as Error).message} />}

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={create.isPending}
                className="rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-accent/90 disabled:opacity-60"
              >
                {create.isPending ? 'Uploading…' : 'Start analysis'}
              </button>
              {health.data?.llm_mode === 'dev' && (
                <span className="text-xs text-warn">
                  No LLM key configured: claim extraction will use labelled development samples.
                </span>
              )}
            </div>
          </form>
        </Card>
      </section>

      <section className="animate-fade-in">
        <SectionTitle kicker="History" title="Previous analyses" />
        <JobHistory />
      </section>
    </div>
  )
}

function JobHistory() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['jobs'],
    queryFn: api.jobs,
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === 'running' || j.status === 'queued') ? 3000 : false),
  })
  if (isLoading) return <Spinner label="Loading analyses…" />
  if (isError) return <ErrorBox message={(error as Error).message} />
  if (!data?.length)
    return (
      <Empty title="No analyses yet">
        Start one on the left, or open the <Link to="/sample-report" className="text-accent underline">sample report</Link>{' '}
        to see what a finished report looks like.
      </Empty>
    )
  return (
    <ul className="space-y-3">
      {data.map((job) => (
        <JobRow key={job.id} job={job} />
      ))}
    </ul>
  )
}

function JobRow({ job }: { job: Job }) {
  const finished = ['done', 'failed', 'incomplete'].includes(job.status)
  return (
    <li>
      <Card className="p-4 transition hover:border-faint">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <Link to={`/jobs/${job.id}`} className="block truncate font-medium hover:text-accent">
              {job.paper_title ?? job.pdf_filename ?? job.id}
            </Link>
            <div className="mt-0.5 truncate text-xs text-faint">
              {job.repo_url ?? job.repo_filename}
              {job.repo_commit && <Mono className="ml-1.5">@{job.repo_commit.slice(0, 7)}</Mono>}
            </div>
          </div>
          <JobStatusBadge status={job.status} />
        </div>
        <div className="mt-3 flex items-center justify-between text-xs">
          <span className="text-faint">{fmtWhen(job.created_at)}</span>
          <div className="flex gap-3 font-medium">
            <Link to={`/jobs/${job.id}`} className="text-accent hover:underline">
              Open
            </Link>
            {finished && (
              <Link to={`/jobs/${job.id}?replay=1`} className="text-accent hover:underline">
                Replay
              </Link>
            )}
            {job.status === 'done' && (
              <Link to={`/jobs/${job.id}/report`} className="text-accent hover:underline">
                Report
              </Link>
            )}
          </div>
        </div>
      </Card>
    </li>
  )
}
