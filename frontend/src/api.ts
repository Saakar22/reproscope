import type { ExecuteOutput, ExtractOutput, Health, IngestOutput, Job, ParseOutput, PlanOutput, Rerun, ScanOutput, JobOptions, PipelineEvent, Report, StageId } from './types'

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError(0, 'Cannot reach the ReproScope backend. Is it running on port 8000?')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail || `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/api/health'),
  jobs: () => request<Job[]>('/api/jobs'),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  eventHistory: (id: string) => request<PipelineEvent[]>(`/api/jobs/${id}/events/history`),
  stage: <T = unknown>(id: string, stage: StageId) => request<T>(`/api/jobs/${id}/stages/${stage}`),
  ingest: (id: string) => request<IngestOutput>(`/api/jobs/${id}/stages/ingest`),
  extracted: (id: string) => request<ExtractOutput>(`/api/jobs/${id}/stages/extract`),
  scanned: (id: string) => request<ScanOutput>(`/api/jobs/${id}/stages/scan`),
  planned: (id: string) => request<PlanOutput>(`/api/jobs/${id}/stages/plan`),
  executed: (id: string) => request<ExecuteOutput>(`/api/jobs/${id}/stages/execute`),
  parsed: (id: string) => request<ParseOutput>(`/api/jobs/${id}/stages/parse`),
  report: (id: string) => request<Report>(`/api/jobs/${id}/report`),
  repoFile: (id: string, path: string) =>
    request<{ path: string; language: string; lines: string[]; truncated: boolean }>(
      `/api/jobs/${id}/file?path=${encodeURIComponent(path)}`,
    ),
  reruns: (id: string) => request<Rerun[]>(`/api/jobs/${id}/reruns`),
  startRerun: (id: string, body: { claim_id: string; overrides: Record<string, string>; note?: string }) =>
    request<Rerun>(`/api/jobs/${id}/rerun`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  sampleReport: () => request<Report>('/api/fixtures/sample-report'),
  replay: (id: string, speed: number) =>
    request<{ events_url: string; event_count: number }>(`/api/jobs/${id}/replay?speed=${speed}`, {
      method: 'POST',
    }),

  createJob: (input: { pdf: File; repoUrl?: string; repoZip?: File; options: JobOptions }) => {
    const form = new FormData()
    form.append('pdf', input.pdf)
    if (input.repoUrl) form.append('repo_url', input.repoUrl)
    if (input.repoZip) form.append('repo_zip', input.repoZip)
    form.append('options', JSON.stringify(input.options))
    return request<Job>('/api/jobs', { method: 'POST', body: form })
  },
}

export function exportUrl(source: { fixture: true } | { jobId: string }, format: 'markdown' | 'json') {
  return 'fixture' in source
    ? `/api/fixtures/sample-report/export?format=${format}`
    : `/api/jobs/${source.jobId}/export?format=${format}`
}
