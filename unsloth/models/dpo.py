# Unsloth Studio
# Copyright (C) 2023-present the Unsloth AI team. All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from transformers.utils.notebook import (
    IntervalStrategy,
    NotebookTrainingTracker,
    NotebookProgressCallback,
)

DPOTrainer_metrics = [
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
    "logps/rejected",
    "logps/chosen",
    "logits/rejected",
    "logits/chosen",
]
set_DPOTrainer_metrics = frozenset(DPOTrainer_metrics)


def NotebookProgressCallback_on_train_begin(self, args, state, control, **kwargs):
    self.first_column = "Epoch" if args.evaluation_strategy == IntervalStrategy.EPOCH else "Step"
    self.training_loss = 0
    self.last_log = 0
    column_names = [self.first_column] + ["Training Loss"]
    if args.evaluation_strategy != IntervalStrategy.NO:
        column_names.append("Validation Loss")
    column_names += [x.replace("/", " / ") for x in DPOTrainer_metrics]
    self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)
pass


def NotebookProgressCallback_on_log(self, args, state, control, logs=None, **kwargs):
    # Only for when there is no evaluation
    if args.evaluation_strategy == IntervalStrategy.NO and "loss" in logs:
        values = {"Training Loss": logs["loss"]}
        for metric in DPOTrainer_metrics:
            values[metric.replace("/", " / ")] = logs[metric]
        pass
        # First column is necessarily Step since we're not in epoch eval strategy
        values["Step"] = state.global_step
        self.training_tracker.write_line(values)
    pass
pass


def NotebookTrainingTracker_write_line(self, values):
    """
    Write the values in the inner table.

    Args:
        values (`Dict[str, float]`): The values to display.
    """
    if self.inner_table is None:
        self.inner_table = [list(values.keys()), list(values.values())]
    else:
        columns = self.inner_table[0]
        new_values = {}
        for key, value in values.items():
            lowered = key.lower()
            if lowered in set_DPOTrainer_metrics:
                new_values[lowered.replace("/", " / ")] = value
            else:
                new_values[key] = value
        pass
        values = new_values

        self.inner_table[0] = columns
        if len(self.inner_table) > 1:
            last_values = self.inner_table[-1]
            first_column = self.inner_table[0][0]
            if last_values[0] != values[first_column]:
                # write new line
                self.inner_table.append([values[c] if c in values else "No Log" for c in columns])
            else:
                # update last line
                new_values = values
                for c in columns:
                    if c not in new_values.keys():
                        new_values[c] = last_values[columns.index(c)]
                self.inner_table[-1] = [new_values[c] for c in columns]
        else:
            # Edit for evaluation purposes
            self.inner_table.append([values[c] if c in values else 0 for c in columns])
        pass
    pass
pass


def PatchDPOTrainer():
    # Patch DPO notebook printing
    NotebookTrainingTracker.write_line = NotebookTrainingTracker_write_line
    from transformers.trainer import DEFAULT_PROGRESS_CALLBACK
    DEFAULT_PROGRESS_CALLBACK.on_train_begin = NotebookProgressCallback_on_train_begin
    DEFAULT_PROGRESS_CALLBACK.on_log         = NotebookProgressCallback_on_log
pass

