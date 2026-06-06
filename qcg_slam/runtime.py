"""Runtime accounting helpers."""

from dataclasses import dataclass
import os


@dataclass
class RuntimeStats:
    tracking_iter_time_sum: float = 0.0
    tracking_iter_time_count: int = 0
    mapping_iter_time_sum: float = 0.0
    mapping_iter_time_count: int = 0
    tracking_frame_time_sum: float = 0.0
    tracking_frame_time_count: int = 0
    mapping_frame_time_sum: float = 0.0
    mapping_frame_time_count: int = 0

    def add_tracking_iter(self, duration):
        self.tracking_iter_time_sum += duration
        self.tracking_iter_time_count += 1

    def add_mapping_iter(self, duration):
        self.mapping_iter_time_sum += duration
        self.mapping_iter_time_count += 1

    def add_tracking_frame(self, duration):
        self.tracking_frame_time_sum += duration
        self.tracking_frame_time_count += 1

    def add_mapping_frame(self, duration):
        self.mapping_frame_time_sum += duration
        self.mapping_frame_time_count += 1

    def averages(self):
        tracking_iter_count = self.tracking_iter_time_count or 1
        tracking_frame_count = self.tracking_frame_time_count or 1
        mapping_iter_count = self.mapping_iter_time_count or 1
        mapping_frame_count = self.mapping_frame_time_count or 1
        return {
            'tracking_iter':
                self.tracking_iter_time_sum / tracking_iter_count,
            'tracking_frame':
                self.tracking_frame_time_sum / tracking_frame_count,
            'mapping_iter':
                self.mapping_iter_time_sum / mapping_iter_count,
            'mapping_frame':
                self.mapping_frame_time_sum / mapping_frame_count,
        }


def report_runtime_stats(config, eval_dir, wandb_run, stats,
                         total_global_optimization_time):
    """Print and persist final runtime statistics."""
    averages = stats.averages()
    print("\nAverage Tracking/Iteration Time: "
          f"{averages['tracking_iter'] * 1000} ms")
    print(f"Average Tracking/Frame Time: {averages['tracking_frame']} s")
    print(f"Average Mapping/Iteration Time: {averages['mapping_iter']*1000} ms")
    print(f"Average Mapping/Frame Time: {averages['mapping_frame']} s")

    if config['use_wandb']:
        wandb_run.log({
            "Final Stats/Average Tracking Iteration Time (ms)":
                averages['tracking_iter'] * 1000,
            "Final Stats/Average Tracking Frame Time (s)":
                averages['tracking_frame'],
            "Final Stats/Average Mapping Iteration Time (ms)":
                averages['mapping_iter'] * 1000,
            "Final Stats/Average Mapping Frame Time (s)":
                averages['mapping_frame'],
            "Final Stats/step":
                1
        })
        return

    with open(os.path.join(eval_dir, "runtime.txt"), "w") as f:
        f.write("Average Tracking Iteration Time: %.6f ms \n" %
                (averages['tracking_iter'] * 1000))
        f.write("Average Tracking Frame Time: %.6f s \n" %
                (averages['tracking_frame']))
        f.write("Average Mapping Iteration Time: %.6f ms \n" %
                (averages['mapping_iter'] * 1000))
        f.write("Average Mapping Frame Time: %.6f s \n" %
                (averages['mapping_frame']))
        f.write("Total Global Optimization Time: %.6f s \n" %
                (total_global_optimization_time))
