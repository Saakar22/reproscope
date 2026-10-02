import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'
import { api } from '../api'
import { ErrorBox, Spinner } from './ui'

/** Read-only view of one repository file with the cited line highlighted and scrolled into view. */
export default function CodeViewer({ jobId, path, line }: { jobId: string; path: string; line?: number | null }) {
  const { data, isError, error } = useQuery({ queryKey: ['file', jobId, path], queryFn: () => api.repoFile(jobId, path) })
  const target = useRef<HTMLTableRowElement>(null)
  useEffect(() => {
    target.current?.scrollIntoView({ block: 'center' })
  }, [data])

  if (isError) return <div className="p-4"><ErrorBox title="Could not open the file" message={(error as Error).message} /></div>
  if (!data) return <div className="p-4"><Spinner label="Loading file…" /></div>
  return (
    <div>
      {data.truncated && <div className="bg-warn-soft px-4 py-1.5 text-xs text-warn">Large file: only the first 400 KB are shown.</div>}
      <table className="w-full border-collapse font-mono text-[12.5px] leading-5">
        <tbody>
          {data.lines.map((text, i) => {
            const no = i + 1
            const hit = no === line
            return (
              <tr key={no} ref={hit ? target : undefined} className={hit ? 'bg-warn-soft' : undefined}>
                <td className={`w-12 select-none border-r border-rule px-3 text-right align-top ${hit ? 'font-semibold text-warn' : 'text-faint'}`}>
                  {no}
                </td>
                <td className="px-3 whitespace-pre">{text || ' '}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
