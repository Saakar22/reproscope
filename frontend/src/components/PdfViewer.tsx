import { useCallback, useMemo, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import { ErrorBox, Spinner } from './ui'

// Must be set in the same module that renders <Document> (react-pdf docs).
pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()

function norm(s: string): string {
  return s.normalize('NFKC').replace(/[−‐-—]/g, '-').replace(/\s+/g, ' ').trim().toLowerCase()
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!)
}

/** Shows one page of the paper with the text items that make up `quote` highlighted. */
export default function PdfViewer({ url, page, quote }: { url: string; page: number; quote?: string | null }) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const [current, setCurrent] = useState(page)
  const [width, setWidth] = useState(720)
  const [error, setError] = useState<string | null>(null)
  const target = useMemo(() => (quote ? norm(quote) : ''), [quote])

  const measure = useCallback((el: HTMLDivElement | null) => {
    if (el) setWidth(Math.min(900, el.clientWidth - 32))
  }, [])

  const renderText = useCallback(
    ({ str }: { str: string }) => {
      const t = norm(str)
      const hit = target && current === page && t.length >= 3 && target.includes(t)
      return hit ? `<mark class="rs-hl">${escapeHtml(str)}</mark>` : escapeHtml(str)
    },
    [target, current, page],
  )

  return (
    <div ref={measure} className="flex flex-col items-center gap-3 p-4">
      <style>{`.rs-hl{background:rgba(250,204,21,.55);color:transparent;border-radius:2px;box-shadow:0 0 0 1px rgba(202,138,4,.6)}`}</style>
      {numPages && (
        <div className="flex items-center gap-3 text-sm">
          <button type="button" disabled={current <= 1} onClick={() => setCurrent(current - 1)}
            className="rounded border border-rule px-2 py-0.5 disabled:opacity-40">‹ Prev</button>
          <span className="text-muted">
            Page {current} of {numPages}
            {current === page && quote && <span className="ml-2 rounded bg-warn-soft px-1.5 py-0.5 text-xs text-warn">quoted text highlighted</span>}
          </span>
          <button type="button" disabled={current >= numPages} onClick={() => setCurrent(current + 1)}
            className="rounded border border-rule px-2 py-0.5 disabled:opacity-40">Next ›</button>
        </div>
      )}
      {error ? (
        <ErrorBox title="Could not display the PDF" message={error} />
      ) : (
        <Document
          file={url}
          onLoadSuccess={(doc) => setNumPages(doc.numPages)}
          onLoadError={(e) => setError(e.message)}
          loading={<Spinner label="Loading paper…" />}
        >
          <div className="overflow-hidden rounded border border-rule shadow-sm">
            <Page pageNumber={current} width={width} customTextRenderer={renderText} loading={<Spinner label="Rendering page…" />} />
          </div>
        </Document>
      )}
    </div>
  )
}
