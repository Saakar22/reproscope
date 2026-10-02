import { useQuery } from '@tanstack/react-query'
import { Suspense, lazy, useCallback, useState, type ReactNode } from 'react'
import CodeViewer from '../components/CodeViewer'
import Drawer from '../components/Drawer'
import { Link, useParams } from 'react-router-dom'
import { ApiError, api, exportUrl } from '../api'
import DeltaChart from '../components/report/DeltaChart'
import {
  ClaimsTable,
  ExecutiveSummary,
  Explanations,
  FindingsList,
  ReportMeta,
  RunsTable,
  type EvidenceLinks,
} from '../components/report/sections'
import { Card, ErrorBox, SectionTitle, Spinner } from '../components/ui'

// pdf.js is large; only load it when someone opens the paper.
const PdfViewer = lazy(() => import('../components/PdfViewer'))

function Section({ id, kicker, title, children, action }: { id: string; kicker: string; title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20">
      <SectionTitle kicker={kicker} title={title}>{action}</SectionTitle>
      {children}
    </section>
  )
}

const NAV = [
  ['summary', 'Summary'],
  ['results', 'Results'],
  ['claims', 'Claims & evidence'],
  ['findings', 'Findings'],
  ['runs', 'Runs & logs'],
  ['explanations', 'Explanations'],
] as const

type Viewer = { kind: 'pdf'; page: number; quote: string | null } | { kind: 'file'; path: string; line: number | null }

export default function ReportPage({ source }: { source: 'job' | 'fixture' }) {
  const { jobId } = useParams<{ jobId: string }>()
  const isFixture = source === 'fixture'
  const [viewer, setViewer] = useState<Viewer | null>(null)
  const closeViewer = useCallback(() => setViewer(null), [])
  const query = useQuery({
    queryKey: ['report', isFixture ? 'fixture' : jobId],
    queryFn: () => (isFixture ? api.sampleReport() : api.report(jobId!)),
    retry: (count, err) => !(err instanceof ApiError && err.status === 409) && count < 1,
  })

  if (query.isLoading) return <Spinner label="Loading report…" />
  if (query.isError) {
    const err = query.error as Error
    if (err instanceof ApiError && err.status === 409)
      return (
        <Card className="mx-auto max-w-xl p-6 text-center">
          <h1 className="font-serif text-2xl font-semibold">No report yet</h1>
          <p className="mt-2 text-sm text-muted">{err.message}</p>
          <Link to={`/jobs/${jobId}`} className="mt-4 inline-block text-sm font-medium text-accent underline underline-offset-4">
            Back to the pipeline
          </Link>
        </Card>
      )
    return <ErrorBox title="Could not load the report" message={err.message} />
  }
  const report = query.data!
  const links: EvidenceLinks = {
    pdfAt: (page) => (isFixture ? null : `/api/jobs/${report.job.id}/pdf#page=${page}`),
    openPaper: isFixture ? null : (page, quote) => setViewer({ kind: 'pdf', page, quote: quote ?? null }),
    openFile: isFixture ? null : (path, line) => setViewer({ kind: 'file', path, line: line ?? null }),
    jobId: isFixture ? null : report.job.id,
  }
  const exportSource = isFixture ? ({ fixture: true } as const) : { jobId: report.job.id }

  const exportButtons = (
    <div className="flex gap-2">
      {(['markdown', 'json'] as const).map((f) => (
        <a
          key={f}
          href={exportUrl(exportSource, f)}
          download
          className="rounded-lg border border-rule bg-surface px-3 py-1.5 text-sm font-medium hover:border-faint"
        >
          Export {f === 'markdown' ? 'Markdown' : 'JSON'}
        </a>
      ))}
    </div>
  )

  return (
    <div className="animate-fade-in">
      <Drawer
        open={viewer !== null}
        onClose={closeViewer}
        title={viewer?.kind === 'pdf' ? report.job.paper_title ?? 'Paper' : viewer?.kind === 'file' ? viewer.path : ''}
        subtitle={
          viewer?.kind === 'pdf'
            ? viewer.quote ? `Page ${viewer.page} · looking for “${viewer.quote.slice(0, 80)}”` : `Page ${viewer.page}`
            : viewer?.kind === 'file'
              ? `Repository file${viewer.line ? ` · line ${viewer.line} highlighted` : ''}${report.job.repo_commit ? ` · @${report.job.repo_commit.slice(0, 10)}` : ''}`
              : undefined
        }
      >
        {viewer?.kind === 'pdf' && (
          <Suspense fallback={<div className="p-4"><Spinner label="Loading the PDF viewer…" /></div>}>
            <PdfViewer url={`/api/jobs/${report.job.id}/pdf`} page={viewer.page} quote={viewer.quote} />
          </Suspense>
        )}
        {viewer?.kind === 'file' && <CodeViewer jobId={report.job.id} path={viewer.path} line={viewer.line} />}
      </Drawer>

      {report.is_fixture && (
        <div className="mb-6 rounded-lg border-2 border-dashed border-warn/60 bg-warn-soft px-4 py-3 text-sm text-warn" role="note">
          <strong className="tracking-wide">SAMPLE FIXTURE — NOT A REAL ANALYSIS.</strong> {report.fixture_note}
        </div>
      )}

      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-[11px] font-semibold tracking-[0.14em] text-faint uppercase">Reproducibility report</div>
          <h1 className="mt-1 font-serif text-2xl font-semibold tracking-tight sm:text-3xl">
            {report.job.paper_title ?? report.job.pdf_filename}
          </h1>
          <div className="mt-2"><ReportMeta report={report} /></div>
        </div>
        {exportButtons}
      </div>

      <nav className="sticky top-[57px] z-10 -mx-4 mt-6 overflow-x-auto border-b border-rule bg-paper/90 px-4 backdrop-blur sm:-mx-6 sm:px-6">
        <div className="flex gap-5 text-sm">
          {NAV.map(([id, label]) => (
            <a key={id} href={`#${id}`} className="border-b-2 border-transparent py-2.5 whitespace-nowrap text-muted hover:border-accent hover:text-ink">
              {label}
            </a>
          ))}
        </div>
      </nav>

      <div className="mt-8 space-y-12">
        <Section id="summary" kicker="A" title="Executive summary">
          <ExecutiveSummary report={report} />
        </Section>

        <Section id="results" kicker="B" title="Reported vs obtained">
          <Card className="p-5">
            <DeltaChart claims={report.claims} />
          </Card>
        </Section>

        <Section id="claims" kicker="C" title="Claims and evidence">
          <Card className="p-4">
            <ClaimsTable report={report} links={links} />
          </Card>
        </Section>

        <Section id="findings" kicker="D" title="Repository findings">
          <FindingsList findings={report.findings} links={links} />
        </Section>

        <Section id="runs" kicker="E" title="Experiment runs and logs">
          <Card className="p-4">
            <RunsTable runs={report.runs} logUrl={(rid) => (isFixture ? null : `/api/jobs/${report.job.id}/runs/${rid}/log`)} />
          </Card>
          {report.deviations.length > 0 && (
            <div className="mt-4 text-sm">
              <div className="mb-1 font-medium">Deviations made by ReproScope</div>
              <ul className="list-disc space-y-0.5 pl-5 text-muted">
                {report.deviations.map((d) => (
                  <li key={d.id}>
                    <span className="font-mono text-xs">{d.id}</span> ({d.kind}) {d.detail}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Section>

        <Section id="explanations" kicker="F" title="Discrepancy explanations">
          <Explanations report={report} />
        </Section>

        <section className="border-t border-rule pt-6">
          <div className="flex flex-wrap items-start justify-between gap-6">
            <div>
              <h3 className="text-sm font-semibold">Tolerance policy</h3>
              <ul className="mt-1.5 space-y-0.5 text-xs text-muted">
                {Object.entries(report.tolerance_policy).map(([k, v]) => (
                  <li key={k}><span className="font-mono">{k}</span>: {v}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="mb-1.5 text-sm font-semibold">G. Export</h3>
              {exportButtons}
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
