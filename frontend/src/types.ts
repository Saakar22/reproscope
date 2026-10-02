// Mirrors backend/reproscope/schemas.py — change both together.

export type StageId = 'ingest' | 'extract' | 'scan' | 'plan' | 'execute' | 'parse' | 'compare'
export type StageStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped' | 'not_implemented'
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'incomplete'

export type Verdict =
  | 'REPRODUCED'
  | 'PARTIAL'
  | 'NOT_REPRODUCED'
  | 'INCONCLUSIVE'
  | 'NOT_RUN'
  | 'NO_METRIC'
  | 'UNVERIFIED_CLAIM'

export interface JobOptions {
  main_results_only: boolean
  max_claims: number
  enable_hypothesis_reruns: boolean
  timeout_s: number
}

export interface Job {
  id: string
  created_at: string
  updated_at: string
  paper_title: string | null
  pdf_filename: string | null
  pdf_pages: number | null
  repo_source: 'git' | 'zip'
  repo_url: string | null
  repo_filename: string | null
  repo_commit: string | null
  status: JobStatus
  current_stage: StageId | null
  stage_detail: string | null
  stages: Record<StageId, StageStatus>
  options: JobOptions
  llm_mode: 'live' | 'dev'
  error: string | null
}

export interface PipelineEvent {
  job_id: string
  seq: number
  ts: string
  stage: StageId | null
  level: 'info' | 'warning' | 'error' | 'stage' | 'log'
  message: string
  data: unknown
}

export interface Health {
  status: string
  llm_mode: 'live' | 'dev'
  llm_provider: string
  llm_model: string | null
  docker: { available: boolean; version?: string; error?: string }
  stages: { id: StageId; label: string }[]
  limits: { max_pdf_mb: number; max_zip_mb: number }
}

export interface IngestOutput {
  pdf: { pages: number; title: string | null; chars_per_page: number[]; empty_pages: number[] }
  repo: {
    source: 'git' | 'zip'
    url: string | null
    commit: string | null
    files: number
    python_files: number
    size_bytes: number
  }
}

export interface ExtractedClaim {
  id: string
  experiment: string
  dataset: string | null
  split: string | null
  metric: string
  higher_is_better: boolean
  reported_value: number
  reported_std: number | null
  unit: string
  source: string | null
  page: number
  quote: string
  hyperparameters: { name: string; value: string; quote: string; verified: boolean }[]
  is_main_result: boolean
  grounded: boolean
  grounding: { quote_score: number; number_found: boolean; page_found: number | null; llm_page: number }
  origin: 'llm' | 'dev_sample' | string
}

export interface ExtractOutput {
  origin: 'llm' | 'dev_sample'
  model: string | null
  cached: boolean
  usage: { total_tokens?: number }
  paper_title: string | null
  skipped_pages: number[]
  rejected_malformed: number
  claims: ExtractedClaim[]
}

export interface ScanFact {
  id: string
  kind: string
  name: string
  value: string | null
  file: string | null
  line: number | null
  extra: Record<string, unknown>
}

export interface ScanArg {
  name: string
  flags: string[]
  default: unknown
  type: string | null
  action: string | null
  required: boolean
  help: string | null
  line: number
}

export interface ScanOutput {
  summary: {
    python_files: number
    files: number
    notebooks: string[]
    frameworks: string[]
    cli_frameworks: string[]
    entrypoints: string[]
    readme_commands: number
    seed_set_in_entrypoints: boolean | null
    unseeded_entrypoints: string[]
    any_seed_call: boolean
    deps_total: number
    deps_pinned: number
    deps_all_pinned: boolean | null
    gpu_only_files: string[]
    python_version: string | null
  }
  entrypoints: {
    file: string
    has_main_guard: boolean
    score: number
    args: ScanArg[]
    seeded: boolean
    seed_calls: { file: string; line: number; call: string }[]
  }[]
  readme_commands: { command: string; file: string; line: number; script: string | null; script_exists: boolean }[]
  warnings: string[]
  facts: ScanFact[]
  fact_counts: Record<string, number>
}

export interface ParamDiff {
  param: string
  paper_name: string
  paper_value: string
  paper_quote: string
  flag: string | null
  repo_value: string | null
  repo_source: 'default' | 'config' | 'command' | null
  repo_file: string | null
  repo_line: number | null
  status: 'match' | 'mismatch' | 'not_in_repo'
  testable: boolean
}

export interface PlanItem {
  id: string
  claim_id: string
  origin: 'llm' | 'dev_sample' | 'none'
  runnable: boolean
  reason: string
  validation: { ok: boolean; errors: string[]; warnings: string[] }
  retried: boolean
  argv: string[] | null
  command: string | null
  script: string | null
  source: 'readme' | 'entrypoint_defaults' | 'repo_config' | null
  source_ref: string | null
  extra_args: string[]
  metric_location: string | null
  metric_key: string | null
  estimated_minutes: number | null
  confidence: 'high' | 'medium' | 'low' | null
  overrides: ParamDiff[]
  run_group: number | null
}

export interface PlanOutput {
  origin: string
  candidates: { id: string; command: string; source: string; source_ref: string | null; note: string }[]
  plans: PlanItem[]
  skipped_unverified: string[]
  groups?: { command: string; plans: string[] }[]
}

export interface ExecRun {
  id: string
  command: string
  argv: string[]
  plan_ids: string[]
  claim_ids: string[]
  status: 'ok' | 'error' | 'timeout' | 'not_runnable' | 'not_executed'
  exit_code?: number | null
  seconds?: number
  network?: string
  outputs?: { path: string; size: number; new?: boolean; skipped?: string }[]
  failure?: string | null
  attempts?: { attempt: number; status: string; exit_code: number | null; seconds: number; network: string; failure: string | null }[]
  log_path?: string
  reason?: string | null
}

export interface ExecuteOutput {
  docker_available: boolean | null
  image: string | null
  python: string | null
  packages: string[]
  runs: ExecRun[]
  plan_runs: Record<string, string>
  deviations: DeviationOut[]
}

export interface ParsedResult {
  claim_id: string
  plan_id: string
  metric: string
  unit: string
  status: 'found' | 'no_metric' | 'not_run'
  run_id: string | null
  run_status?: string
  value?: number
  raw_value?: number
  source?: 'file' | 'regex' | 'llm'
  evidence?: { file?: string; key?: string; line_no?: number; line_text?: string; llm_reason?: string }
  normalisation?: string | null
  variant_note?: string | null
  attempts?: string[]
  reason?: string
}

export interface ParseOutput {
  results: ParsedResult[]
}

export interface RerunResult {
  values: number[]
  obtained_after?: number
  obtained_before?: number
  gap_before?: number
  gap_after?: number
  tolerance?: number
  outcome: HypothesisOutcome
  statement: string
}

export interface Rerun {
  id: string
  claim_id: string
  claim_ids: string[]
  overrides: Record<string, string>
  note: string | null
  argv: string[]
  repeats: number
  status: 'queued' | 'running' | 'done' | 'failed'
  run_ids: string[]
  results: Record<string, RerunResult>
  error: string | null
  created_at: string
  finished_at: string | null
}

// ----------------------------------------------------------------- report

export interface Grounding {
  quote_score: number | null
  number_found: boolean | null
  page_found: number | null
}

export interface ComparisonOut {
  reported: number
  obtained: number | null
  abs_delta: number | null
  rel_delta: number | null
  tolerance: number | null
  tolerance_basis: string | null
  seed_values: number[]
  verdict: Verdict
  better_than_reported: boolean
  run_status: string | null
  note: string | null
}

export interface Hypothesis {
  text: string
  evidence_ids: string[]
  likelihood: 'high' | 'medium' | 'low'
  suggested_check: string | null
}

export interface ClaimReport {
  id: string
  experiment: string
  dataset: string | null
  split: string | null
  metric: string
  unit: 'percent' | 'fraction' | 'raw' | string
  higher_is_better: boolean
  reported_value: number
  reported_std: number | null
  source: string | null
  page: number
  quote: string
  hyperparameters: Record<string, unknown>
  is_main_result: boolean
  grounded: boolean
  grounding: Grounding
  origin: string
  comparison: ComparisonOut | null
  run_ids: string[]
  finding_ids: string[]
  deviation_ids: string[]
  hypotheses: Hypothesis[]
  diagnosis_origin: string | null
}

export interface RepoEvidence {
  file: string
  line: number | null
  snippet: string | null
}

export type HypothesisOutcome = 'supports' | 'partially_supports' | 'does_not_support' | 'inconclusive' | 'not_applicable'

export interface HypothesisTest {
  run_id?: string
  run_ids?: string[]
  override?: Record<string, unknown>
  values?: number[]
  obtained_after?: number
  gap_before?: number
  gap_after?: number
  tolerance?: number
  supports_hypothesis?: HypothesisOutcome
  statement?: string
}

export interface FindingOut {
  id: string
  rule: string
  severity: 'high' | 'medium' | 'low'
  claim_id: string | null
  claim_ids?: string[]
  summary: string
  impact: string | null
  paper_evidence: { page?: number; quote?: string } | null
  repo_evidence: RepoEvidence[]
  run_evidence: { run_id?: string; line_no?: number; line_text?: string } | null
  test: HypothesisTest | null
}

export interface DeviationOut {
  id: string
  run_id: string | null
  kind: string
  detail: string
}

export interface RunOut {
  id: string
  plan_id: string
  claim_id: string
  kind: string
  command: string
  overrides: Record<string, unknown>
  status: string
  exit_code: number | null
  seconds: number | null
  image_tag: string | null
  environment: Record<string, unknown>
  log_available: boolean
  metric_evidence: Record<string, unknown> | null
}

export interface PlanOut {
  id: string
  claim_id: string
  script: string
  command: string
  source_of_command: string | null
  expected_metric_location: string | null
  estimated_minutes: number | null
  confidence: string | null
  paper_overrides: Record<string, unknown>
  validation: { ok?: boolean; errors?: string[] }
}

export interface RepoFactOut {
  id: string
  kind: string
  name: string
  value: string | null
  file: string | null
  line: number | null
}

export interface ReportSummary {
  total_claims: number
  grounded_claims: number
  reproduced: number
  partial: number
  not_reproduced: number
  inconclusive: number
  not_run: number
  no_metric: number
  unverified: number
  better_than_reported: number
}

export interface Report {
  is_fixture: boolean
  fixture_note: string | null
  generated_at: string
  job: {
    id: string
    paper_title: string | null
    pdf_filename: string | null
    repo_source: string
    repo_url: string | null
    repo_filename: string | null
    repo_commit: string | null
    created_at: string
    status: string
    llm_mode: string
  }
  summary: ReportSummary
  checklist: {
    seed_set: boolean | null
    deps_pinned: boolean | null
    all_hparams_stated: boolean | null
    split_stated: boolean | null
  }
  claims: ClaimReport[]
  findings: FindingOut[]
  deviations: DeviationOut[]
  runs: RunOut[]
  plans: PlanOut[]
  repo_facts: RepoFactOut[]
  tolerance_policy: Record<string, string>
  reruns?: Rerun[]
}
