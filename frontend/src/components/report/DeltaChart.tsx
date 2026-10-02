import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ClaimReport, Verdict } from '../../types'
import { VERDICT_META, fmtNum, fmtSigned } from '../ui'

/**
 * Claims are measured in different units (accuracy %, RMSE, …), so plotting raw values on one
 * axis would mislead. Instead each bar is the gap expressed in that claim's own tolerance:
 * |x| ≤ 1 is within tolerance, 1–3 is "partial", beyond 3 is "not reproduced".
 */
interface Row {
  id: string
  units: number
  claim: ClaimReport
}

const MARK: Partial<Record<Verdict, string>> = {
  REPRODUCED: 'var(--color-mark-ok)',
  PARTIAL: 'var(--color-mark-warn)',
  NOT_REPRODUCED: 'var(--color-mark-bad)',
}

function markFor(claim: ClaimReport): string {
  return MARK[claim.comparison!.verdict] ?? 'var(--color-neutral)'
}

/** Horizontal bar with a 4px rounded data-end and a square baseline end, for either sign. */
function DivergingBar(props: { x?: number; y?: number; width?: number; height?: number; fill?: string; payload?: Row }) {
  const { x = 0, y = 0, width = 0, height = 0, fill, payload } = props
  const left = Math.min(x, x + width)
  const w = Math.abs(width)
  if (w < 0.5) return <rect x={left - 1} y={y} width={2} height={height} fill={fill} />
  const r = Math.min(4, w / 2, height / 2)
  const negative = (payload?.units ?? 0) < 0
  const right = left + w
  const d = negative
    ? `M${right},${y} H${left + r} Q${left},${y} ${left},${y + r} V${y + height - r} Q${left},${y + height} ${left + r},${y + height} H${right} Z`
    : `M${left},${y} H${right - r} Q${right},${y} ${right},${y + r} V${y + height - r} Q${right},${y + height} ${right - r},${y + height} H${left} Z`
  return <path d={d} fill={fill} />
}

function TooltipBody({ active, payload }: { active?: boolean; payload?: { payload: Row }[] }) {
  if (!active || !payload?.length) return null
  const { claim, units } = payload[0].payload
  const c = claim.comparison!
  const pct = claim.unit === 'percent' ? ' %' : ''
  return (
    <div className="rounded-lg border border-rule bg-surface px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-semibold text-ink">
        {claim.id} · {claim.experiment}
      </div>
      <table className="text-muted">
        <tbody>
          <tr><td className="pr-3">Metric</td><td className="text-ink">{claim.metric}{pct}</td></tr>
          <tr><td className="pr-3">Reported</td><td className="font-mono text-ink">{fmtNum(c.reported)}</td></tr>
          <tr><td className="pr-3">Obtained</td><td className="font-mono text-ink">{fmtNum(c.obtained)}</td></tr>
          <tr><td className="pr-3">Δ</td><td className="font-mono text-ink">{fmtSigned(c.abs_delta)} ({fmtSigned(units, 1)} × tol)</td></tr>
          <tr><td className="pr-3">Tolerance</td><td className="font-mono text-ink">±{fmtNum(c.tolerance)}</td></tr>
          <tr><td className="pr-3">Verdict</td><td className="text-ink">{VERDICT_META[c.verdict].label}</td></tr>
        </tbody>
      </table>
    </div>
  )
}

export default function DeltaChart({ claims }: { claims: ClaimReport[] }) {
  const rows: Row[] = claims
    .filter((c) => c.comparison?.obtained != null && c.comparison.abs_delta != null && c.comparison.tolerance)
    .map((c) => ({ id: c.id, claim: c, units: c.comparison!.abs_delta! / c.comparison!.tolerance! }))

  if (!rows.length)
    return <p className="text-sm text-muted">No claim has both a reported and an obtained value yet, so there is nothing to plot.</p>

  const extent = Math.max(4, Math.ceil(Math.max(...rows.map((r) => Math.abs(r.units))) + 0.5))
  const ticks = [-extent, -3, -1, 0, 1, 3, extent].filter((v, i, a) => a.indexOf(v) === i)

  return (
    <figure>
      <div style={{ height: 56 + rows.length * 44 }} role="img" aria-label="Gap between obtained and reported value for each claim, in tolerance units">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 8 }} barCategoryGap={12}>
            <ReferenceArea x1={-1} x2={1} fill="var(--color-ok-soft)" fillOpacity={0.9} ifOverflow="hidden" />
            <ReferenceArea x1={-3} x2={-1} fill="var(--color-warn-soft)" fillOpacity={0.55} ifOverflow="hidden" />
            <ReferenceArea x1={1} x2={3} fill="var(--color-warn-soft)" fillOpacity={0.55} ifOverflow="hidden" />
            <CartesianGrid horizontal={false} stroke="var(--color-mark-grid)" />
            <XAxis
              type="number"
              domain={[-extent, extent]}
              ticks={ticks}
              tickFormatter={(v: number) => (v === 0 ? '0' : `${v > 0 ? '+' : ''}${v}×`)}
              tick={{ fontSize: 11, fill: 'var(--color-faint)' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="id"
              width={36}
              tick={{ fontSize: 12, fill: 'var(--color-muted)', fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
            />
            <ReferenceLine x={0} stroke="var(--color-faint)" />
            <Tooltip content={<TooltipBody />} cursor={{ fill: 'rgba(29,53,87,0.05)' }} />
            <Bar dataKey="units" barSize={18} shape={<DivergingBar />} isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.id} fill={markFor(r.claim)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <span>Gap = (obtained − reported) ÷ tolerance.</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-ok-soft ring-1 ring-ok/20" /> within tolerance (±1×)</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm bg-warn-soft ring-1 ring-warn/20" /> partial (1–3×)</span>
        <span>Bar colour = verdict. Exact values are in the table below.</span>
      </figcaption>
    </figure>
  )
}
