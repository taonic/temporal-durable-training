export interface GpuSlot {
  gpu_id: number;
  holder: string | null;
}

export interface GpuUtilization {
  total: number;
  busy: number;
  free: number;
  queue_depth: number;
  slots: GpuSlot[];
  waiting: string[];
}

export interface EpochMetrics {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_accuracy: number;
  checkpoint_path: string;
}

export interface JobStep {
  name: string;
  label: string;
  kind: string; // workflow | activity | history
  status: string; // pending | running | done
}

export interface RunProgress {
  run_id: string;
  status: string;
  current_epoch: number;
  max_epochs: number;
  best_metric: number | null;
  best_epoch: number;
  latest_checkpoint: string | null;
  gpu_id: number | null;
  history: EpochMetrics[];
  steps?: JobStep[];
  needs_attention: { epoch: number; reason: string; val_loss: number } | null;
  retrying?: boolean;
  retry_attempt?: number;
  last_failure?: string;
  worker?: string | null;
  temporal_url?: string;
}

export interface LeaderboardEntry {
  run_id: string;
  hyperparams: { learning_rate: number; batch_size: number };
  best_metric: number;
  status: string;
}

export interface SweepStatus {
  name: string;
  status: string;
  total: number;
  completed: number;
  leaderboard: LeaderboardEntry[];
  pending_approval: {
    candidate_run_id: string;
    model_name: string;
    metric: number;
    hyperparams: { learning_rate: number; batch_size: number };
    checkpoint_path: string | null;
  } | null;
  decision: { approved: boolean; reviewer: string; note: string } | null;
}

export interface WorkerInfo {
  identity: string;
  workflow: boolean;
  activity: boolean;
  last_access: string | null;
  pid: number | null;
  host?: string;
  spawned: boolean;
  alive: boolean | null; // OS (ps) liveness; null = remote host, can't check
}

export interface DashboardState {
  gpu: GpuUtilization | null;
  runs: RunProgress[];
  sweeps: SweepStatus[];
  workers: WorkerInfo[];
  temporal_ui: string;
  temporal_ui_proxy: string;
}
