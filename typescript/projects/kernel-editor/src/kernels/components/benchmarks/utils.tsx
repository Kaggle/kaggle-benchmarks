import { IBmTaskInfo } from "./tree-sitter/benchmarkTaskParser";
import { Body3, Code, TextLink } from "@kaggle/material";
import * as React from "react";
import styled from "styled-components";

const WarningBody = styled(Body3)`
  color: ${p => p.theme.km.color.status.error};
`;

const WarningCode = styled(Code)`
  font-size: 12px;
`;

/**
 * Returns a warning element if there are issues with the selected benchmark tasks.
 *
 * Checks for:
 * 1. No tasks detected.
 * 2. No task runs.
 * 3. Multiple task runs without a %choose command.
 * 4. Multiple %choose commands.
 */
export const getBenchmarkTaskWarning = (
  localTasks: IBmTaskInfo[] | null
): React.ReactElement | null => {
  if (!localTasks) return null;

  const runTasks = localTasks.filter(t => t.isRun);
  const chosenTasks = localTasks.filter(t => t.isChosen);

  if (localTasks.length === 0) {
    return (
      <WarningBody emphasis="low">
        No tasks detected. Please create a task using the{" "}
        <TextLink
          href={
            "https://github.com/Kaggle/kaggle-benchmarks/blob/ci/user_guide.md#the-kbenchtask-decorator"
          }
        >
          <WarningCode emphasis="low">@kbench.task</WarningCode>
        </TextLink>{" "}
        decorator.
      </WarningBody>
    );
  }

  if (runTasks.length === 0) {
    return (
      <WarningBody emphasis="low">
        Building a task requires benchmark results. Please generate results
        first by executing a task using{" "}
        <TextLink
          href={
            "https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md#basic-task"
          }
        >
          <WarningCode emphasis="low">.run()</WarningCode>
        </TextLink>{" "}
        or{" "}
        <TextLink
          href={
            "https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md#basic-task"
          }
        >
          <WarningCode emphasis="low">.evaluate()</WarningCode>
        </TextLink>
        .
      </WarningBody>
    );
  }

  if (runTasks.length > 1 && chosenTasks.length === 0) {
    return (
      <WarningBody emphasis="low">
        Can only save one task. Use{" "}
        <TextLink
          href={
            "https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md#recipe-publishing-your-task-to-the-leaderboard"
          }
        >
          <WarningCode emphasis="low">%choose &lt;task_name&gt;</WarningCode>
        </TextLink>{" "}
        to select your active task.
      </WarningBody>
    );
  }

  if (chosenTasks.length > 1) {
    return (
      <WarningBody emphasis="low">
        Can only save one task. Multiple tasks selected with{" "}
        <Code emphasis="low">%choose</Code>. Please select only one task.
      </WarningBody>
    );
  }

  return null;
};

/**
 * Validates if the current benchmark task selection is valid for saving.
 *
 * Returns true if valid, false otherwise.
 */
export const validateBenchmarkTaskSelection = (
  localTasks: IBmTaskInfo[] | null
): boolean => {
  if (!localTasks) return true; // Assume valid if no tasks detected yet (or let backend handle it)

  const runTasks = localTasks.filter(t => t.isRun);
  const chosenTasks = localTasks.filter(t => t.isChosen);

  // Invalid if no tasks defined
  if (localTasks.length === 0) {
    return false;
  }

  // Invalid if no runs
  if (runTasks.length === 0) {
    return false;
  }

  // Invalid if multiple runs and no choice
  if (runTasks.length > 1 && chosenTasks.length === 0) {
    return false;
  }

  // Invalid if multiple choices
  if (chosenTasks.length > 1) {
    return false;
  }

  return true;
};
