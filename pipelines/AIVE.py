
import copy
import datetime
import json
import math
import os
import re
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from AIVE_baseline import PipelineBase
from utils.metrics import classify_answer, compute_accuracy


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def resize_to_short_side(img: np.ndarray, target_short: int = 512) -> np.ndarray:
    """Resize an image so its shorter side equals *target_short* pixels."""
    h, w = img.shape[:2]
    if min(h, w) == target_short:
        return img
    scale = target_short / float(min(h, w))
    new_w, new_h = int(math.ceil(w * scale)), int(math.ceil(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SpatialVQAPipelineAIVE(PipelineBase):

    # ------------------------------------------------------------------
    # System prompts
    # ------------------------------------------------------------------

    EXPLORATION_CHECK_PROMPT = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D scene.\n"
        "Your task is to analyze the provided image(s) and a spatial reasoning question,\n"
        "then determine if you need to move the camera to explore the scene further.\n\n"
        "Guidelines:\n"
        "1. Carefully observe what is visible in the current image(s).\n"
        "2. Consider if the question can be answered based solely on what you can see now.\n"
        "3. If you need to see something that is not currently visible (e.g., behind objects,\n"
        "around corners, different angles, objects that are partially occluded), then\n"
        "exploration IS needed.\n"
        "4. If all relevant information is already visible or can be inferred from the "
        "image(s), exploration is NOT needed.\n\n"
        "CRITICAL: Output EXACTLY one word — either EXPLORE or ANSWER.\n"
        "No dashes, no bullets, no punctuation, no explanation. Just the single word."
    )

    ACTION_PLANNING_PROMPT = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D scene. "
        "Your task is to plan the next sequence of camera actions to explore the scene and "
        "gather information needed to answer a spatial reasoning question.\n\n"
        "Available actions:\n"
        "- move-forward {distance} meters (ONLY use 0.25m increments: 0.25, 0.5 etc.)\n"
        "- turn-left {angle} degrees (ONLY use 9° increments: 9, 18 etc.)\n"
        "- turn-right {angle} degrees (ONLY use 9° increments: 9, 18 etc.)\n\n"
        "CRITICAL RULES:\n"
        "1. Plan 2-4 actions.\n"
        "2. Each action on its own line, format exactly: 'action_type value'\n"
        "3. DO NOT output any explanation, reasoning, or commentary — actions ONLY.\n"
        "4. Even if uncertain, always output a plausible exploration sequence.\n\n"
        "Example output:\n"
        "turn-right 18\n"
        "move-forward 0.50\n"
        "turn-left 9\n"
    )

    CONTINUE_EXPLORATION_PROMPT = (
        "You are an AI assistant helping with spatial reasoning. You have been exploring "
        "a 3D scene by taking actions and observing the results. Based on the sequence of "
        "images you've seen and the original question, decide whether you need to continue "
        "exploring.\n\n"
        "Decision Rules:\n"
        "1. OUTPUT 'STOP' IF: The information currently available (from all images seen so "
        "far) is sufficient to answer the question with confidence.\n"
        "2. OUTPUT 'CONTINUE' IF: Critical information is still missing, obscured, or "
        "requires a different viewpoint.\n\n"
        "CRITICAL: Output EXACTLY one word — either STOP or CONTINUE.\n"
        "No dashes, no bullets, no punctuation, no explanation. Just the single word."
    )

    ANSWER_QUESTION_PROMPT = (
        "You are an AI assistant designed to help with spatial reasoning in a 3D scene. "
        "You must analyze any provided images or observations and answer the question.\n\n"
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        model_args = self.model_args
        model_args.seed = 42
        os.makedirs(model_args.output_dir, exist_ok=True)
        self._log_file_handle: Optional[Any] = None
        self._log_file_path: Optional[str] = None

    # ------------------------------------------------------------------
    # World-model interface (override in subclass)
    # ------------------------------------------------------------------

    @abstractmethod
    def _execute_action_sequence(
        self,
        image_path: str,
        step_idx: int,
        action_sequence: List[str],
        save_dir: str,
    ) -> Optional[str]:
        """Execute planned actions and return a new viewpoint image path.

        Subclasses must implement this to integrate a specific world model,
        simulator (e.g. AI2-THOR), or renderer.

        Returns:
            Path to the result image, or ``None`` on failure.
        """
        ...

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_print(self, message: str, level: str = "INFO") -> None:
        """Print a timestamped log message and persist it as JSONL."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {"timestamp": timestamp, "level": level, "message": message}
        print(f"[{timestamp}] [{level}] {message}")

        if self._log_file_handle is None:
            log_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file_path = os.path.join(
                self.model_args.output_dir, f"run_log_{log_start_time}.jsonl"
            )
            os.makedirs(os.path.dirname(self._log_file_path), exist_ok=True)
            self._log_file_handle = open(self._log_file_path, "a", encoding="utf-8")
        elif self._log_file_handle.closed:
            self._log_file_handle = open(self._log_file_path, "a", encoding="utf-8")

        try:
            self._log_file_handle.write(json.dumps(log_entry) + "\n")
            self._log_file_handle.flush()
        except ValueError:
            self._log_file_handle = open(self._log_file_path, "a", encoding="utf-8")
            self._log_file_handle.write(json.dumps(log_entry) + "\n")
            self._log_file_handle.flush()

    def _close_log(self) -> None:
        """Close the log file handle if open."""
        if self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.close()
            self._log_file_handle = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        start_time = datetime.datetime.now()
        os.makedirs(self.model_args.output_dir, exist_ok=True)
        self._log_print(f"Output directory: {self.model_args.output_dir}")
        self._log_print("Starting pipeline")

        for question in self.questions:
            qid = question["database_idx"]

            # --- filtering ---
            if question["question_type"] == "other":
                self._log_print(f"Skipping QID {qid} — question type 'other'.")
                self.results["skip_indices"].append(qid)
                self.save_results()
                continue

            if (self.model_args.question_type != "None"
                    and question["question_type"] != self.model_args.question_type):
                self._log_print(f"Skipping QID {qid} — type mismatch.")
                self.results["skip_indices"].append(qid)
                self.save_results()
                continue

            if len(question["img_paths"]) > self.model_args.max_images:
                self._log_print(f"Skipping QID {qid} — too many images.")
                self.results["skip_indices"].append(qid)
                self.save_results()
                continue

            if self._already_processed(qid):
                self._log_print(f"Skipping already-processed QID {qid}.")
                continue

            # --- load & resize images ---
            save_dir = os.path.join(self.model_args.output_dir, str(qid))
            os.makedirs(os.path.join(save_dir, "step_0"), exist_ok=True)

            primary_img_path, helper_img_path = self._load_and_prepare_images(
                question, save_dir
            )
            if primary_img_path is None:
                self.results["skip_indices"].append(qid)
                self.save_results()
                continue

            # --- exploration gate ---
            self._log_print(
                f"\n===== QID {qid} (type: {question['question_type']}) ====="
            )
            needs_exploration = self._check_if_exploration_needed(
                question=question["question"],
                answer_choices=question["answer_choices"],
                current_image_path=primary_img_path,
                helper_img_path=helper_img_path,
                save_dir=save_dir,
            )

            if getattr(self.model_args, "force_explore", False):
                self._log_print("force_explore=True — overriding gate to EXPLORE.")
                needs_exploration = True

            if not needs_exploration:
                self._handle_direct_answer(question, primary_img_path, helper_img_path, save_dir)
                continue

            # --- exploration loop ---
            self._log_print("Exploration needed — starting exploration loop.")
            result = self._run_exploration_loop(
                question, primary_img_path, helper_img_path, save_dir
            )

            if result == "out of control":
                self.results["parsing_err_stats"]["answer"] += 1
                self.results["parsing_err_stats"]["answer_qid"].append(qid)
                result = "wrong"
            if result in ("correct", "wrong"):
                self.results["progress"][question["question_type"]][result].append(qid)
            self._update_accuracy_stats()

        # --- final log ---
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        self._log_print(f"Pipeline completed. Total time: {duration}")
        with open(os.path.join(self.model_args.output_dir, "final_log.json"), "w", encoding="utf-8") as f:
            json.dump({
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_formatted": str(duration),
                "output_directory": self.model_args.output_dir,
                "final_accuracy": self.results["accuracy"],
            }, f, indent=2)

        self._close_log()

    # ------------------------------------------------------------------
    # Per-question helpers
    # ------------------------------------------------------------------

    def _already_processed(self, qid: str) -> bool:
        """Check whether *qid* has already been scored."""
        if qid in self.results["skip_indices"]:
            return True
        for per_type in self.results["progress"].values():
            if qid in per_type["correct"] or qid in per_type["wrong"]:
                return True
        return False

    def _load_and_prepare_images(
        self, question: Dict[str, Any], save_dir: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Load the primary (and optional helper) image, resize, and save.

        Returns:
            (primary_img_path, helper_img_path) — primary_img_path is None on failure.
        """
        full_path = os.path.join(self.model_args.input_dir, question["img_paths"][0])
        img = cv2.imread(full_path)
        if img is None:
            self._log_print(f"[ERROR] Cannot load image: {full_path}", level="ERROR")
            return None, None

        primary_img_path = os.path.join(save_dir, "step_0", "img_0.png")
        img = self._resize_for_vlm(img)
        cv2.imwrite(primary_img_path, img)

        helper_img_path = None
        if len(question["img_paths"]) > 1:
            full_helper = os.path.join(self.model_args.input_dir, question["img_paths"][1])
            helper = cv2.imread(full_helper)
            if helper is None:
                self._log_print(f"[WARN] Cannot load helper image: {full_helper}", level="WARNING")
            else:
                helper = self._resize_for_vlm(helper)
                helper_img_path = os.path.join(save_dir, "step_0", "helper_img.png")
                cv2.imwrite(helper_img_path, helper)

        return primary_img_path, helper_img_path

    def _resize_for_vlm(self, img: np.ndarray) -> np.ndarray:
        """Resize image based on which VLM is in use."""
        vlm_name = self.model_args.vlm_qa_model_name.lower()
        if "internvl3" in vlm_name:
            return cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        return resize_to_short_side(img, target_short=512)

    def _image_list(
        self, primary: str, helper: Optional[str], num_original: int
    ) -> List[str]:
        """Build the image list for a VLM call."""
        if num_original > 1 and helper is not None:
            return [primary, helper]
        return [primary]

    # ------------------------------------------------------------------
    # Direct-answer path (no exploration)
    # ------------------------------------------------------------------

    def _handle_direct_answer(
        self,
        question: Dict[str, Any],
        primary_img_path: str,
        helper_img_path: Optional[str],
        save_dir: str,
    ) -> None:
        """Answer a question directly (no exploration needed)."""
        qid = question["database_idx"]
        images = self._image_list(
            primary_img_path, helper_img_path, len(question["img_paths"])
        )
        self._log_print("No exploration needed — answering directly.")

        response, result = None, "out of control"
        for _ in range(self.model_args.max_tries_gpt):
            response = self.vlm.run_qa_model(
                prompt_type="answer_question",
                question=question["question"],
                answer_choices=question["answer_choices"],
                images=images,
                sys_prompt=self.ANSWER_QUESTION_PROMPT,
            )
            self._log_print(f"[LLM] {response}")
            result = classify_answer(response, question)
            if result != "out of control":
                break

        self._dump_llm_interaction(
            save_dir, None, None, question, response, result, [], None
        )
        if result == "out of control":
            self.results["parsing_err_stats"]["answer"] += 1
            self.results["parsing_err_stats"]["answer_qid"].append(qid)
            result = "wrong"
        if result in ("correct", "wrong"):
            self.results["progress"][question["question_type"]][result].append(qid)
        self._update_accuracy_stats()

    # ------------------------------------------------------------------
    # Exploration loop
    # ------------------------------------------------------------------

    def _run_exploration_loop(
        self,
        question: Dict[str, Any],
        primary_img_path: str,
        helper_img_path: Optional[str],
        save_dir: str,
    ) -> str:
        """Run the exploration loop and return final answer result."""
        images = self._image_list(
            primary_img_path, helper_img_path, len(question["img_paths"])
        )
        exploration_history: List[Dict[str, Any]] = []
        current_image_path = primary_img_path
        max_steps = self.model_args.max_steps_per_question

        for step in range(max_steps):
            self._log_print(f"--- Step {step} ---")

            action_sequence = self._plan_next_actions(
                question=question["question"],
                answer_choices=question["answer_choices"],
                current_image_path=current_image_path,
                helper_img_path=helper_img_path,
                exploration_history=exploration_history,
                step_idx=step,
                save_dir=save_dir,
            )
            if not action_sequence:
                self._log_print("No actions planned — moving to Q&A.")
                break

            self._log_print(f"Planned actions: {action_sequence}")

            result_image_path = self._execute_action_sequence(
                image_path=current_image_path,
                step_idx=step,
                action_sequence=action_sequence,
                save_dir=save_dir,
            )
            if result_image_path is None:
                self._log_print("Action execution failed — moving to Q&A.", level="WARNING")
                break

            exploration_history.append({
                "step": step,
                "action_sequence": action_sequence,
                "result_image_path": result_image_path,
            })
            current_image_path = result_image_path

            if step < max_steps - 1:
                should_continue = self._decide_continue_exploration(
                    question=question["question"],
                    answer_choices=question["answer_choices"],
                    current_image_path=current_image_path,
                    helper_img_path=helper_img_path,
                    exploration_history=exploration_history,
                    save_dir=save_dir,
                    step_idx=step,
                )
                if not should_continue:
                    self._log_print("VLM decided to stop exploration.")
                    break

        # --- final answer ---
        self._log_print(
            f"Exploration done — {len(exploration_history)} step(s). Generating final answer."
        )
        response, result = None, "out of control"
        for _ in range(self.model_args.max_tries_gpt):
            response = self.vlm.run_qa_model(
                prompt_type="answer_question_exploration",
                question=question["question"],
                answer_choices=question["answer_choices"],
                images=images,
                exploration_history=exploration_history,
                sys_prompt=self.ANSWER_QUESTION_PROMPT,
            )
            self._log_print(f"[LLM] {response}")
            result = classify_answer(response, question)
            if result != "out of control":
                break

        self._dump_llm_interaction(
            save_dir, None, None, question, response, result, [], None
        )
        return result

    # ------------------------------------------------------------------
    # Accuracy tracking
    # ------------------------------------------------------------------

    def _update_accuracy_stats(self) -> None:
        """Recompute per-type and overall accuracy, then persist results."""
        self.results["accuracy"] = compute_accuracy(
            self.results["progress"],
            skip_indices=self.results.get("skip_indices", []),
            total_questions=len(self.questions),
        )
        self.save_results()

    # ------------------------------------------------------------------
    # VLM helpers
    # ------------------------------------------------------------------

    def _check_if_exploration_needed(
        self,
        question: str,
        answer_choices: List[str],
        current_image_path: str,
        helper_img_path: Optional[str],
        save_dir: str,
    ) -> bool:
        """Run the discriminator VLM to decide whether to explore or answer directly."""
        img_paths: List[Optional[str]] = [current_image_path, helper_img_path]
        active_imgs = [p for p in img_paths if p is not None]

        for _ in range(self.model_args.max_tries_gpt):
            response = self.vlm.run_d_model(
                prompt_type="discriminator",
                question=question,
                answer_choices=answer_choices,
                images=active_imgs,
                sys_prompt=self.EXPLORATION_CHECK_PROMPT,
            )
            self._log_print(f"[Exploration Check] {response}")
            decision = response.strip().upper()
            if "EXPLORE" in decision:
                self._dump_exploration_check(save_dir, question, current_image_path, response, True)
                return True
            elif "ANSWER" in decision:
                self._dump_exploration_check(save_dir, question, current_image_path, response, False)
                return False
            time.sleep(1)

        self._log_print("[WARN] VLM decision unclear — defaulting to EXPLORE.", level="WARNING")
        self._dump_exploration_check(
            save_dir, question, current_image_path,
            "VLM decision unclear, defaulting to EXPLORE", True,
        )
        return True

    def _plan_next_actions(
        self,
        question: str,
        answer_choices: List[str],
        current_image_path: str,
        helper_img_path: Optional[str],
        exploration_history: List[Dict[str, Any]],
        step_idx: int,
        save_dir: str,
    ) -> Optional[List[str]]:
        """Run the action-planning VLM and return a validated action sequence."""
        img_paths: List[Optional[str]] = [current_image_path, helper_img_path]
        active_imgs = [p for p in img_paths if p is not None]

        for attempt in range(self.model_args.max_tries_gpt):
            response = self.vlm.run_ap_model(
                prompt_type="action_planing",
                question=question,
                answer_choices=answer_choices,
                images=active_imgs,
                step_idx=step_idx,
                exploration_history=exploration_history,
                sys_prompt=self.ACTION_PLANNING_PROMPT,
            )
            self._log_print(f"[Planning attempt {attempt}] {response}")
            action_sequence = self._parse_action_sequence(response)
            if action_sequence is not None:
                corrected = self._correct_action_values(action_sequence)
                if self._validate_action_sequence(corrected):
                    self._dump_planning_interaction(save_dir, step_idx, question, response, corrected)
                    return corrected
            time.sleep(1)

        self._log_print("[ERROR] Could not get valid action sequence.", level="ERROR")
        return None

    def _decide_continue_exploration(
        self,
        question: str,
        answer_choices: List[str],
        current_image_path: str,
        helper_img_path: Optional[str],
        exploration_history: List[Dict[str, Any]],
        save_dir: str,
        step_idx: int,
    ) -> bool:
        """Run the discriminator to decide whether to continue exploring."""
        img_paths: List[Optional[str]] = [current_image_path, helper_img_path]
        active_imgs = [p for p in img_paths if p is not None]

        for _ in range(self.model_args.max_tries_gpt):
            response = self.vlm.run_d_model(
                prompt_type="discriminator",
                question=question,
                answer_choices=answer_choices,
                images=active_imgs,
                exploration_history=exploration_history,
                sys_prompt=self.CONTINUE_EXPLORATION_PROMPT,
                step_idx=step_idx,
            )
            self._log_print(f"[Continue Decision] {response}")
            decision = response.strip().upper()
            if "STOP" in decision:
                return False
            elif "CONTINUE" in decision:
                return True
            time.sleep(1)

        self._log_print("[WARN] Cannot decide — defaulting to STOP.", level="WARNING")
        return False

    # ------------------------------------------------------------------
    # Action parsing / validation / correction
    # ------------------------------------------------------------------

    def _parse_action_sequence(self, response: str) -> Optional[List[str]]:
        """Parse a VLM response into a list of action strings.

        Handles two patterns: actions with explicit values ("turn-left 18")
        and bare actions without values ("turn left" → defaults to min value).
        """
        patterns_with_value = {
            r'(?:turn|rotat)[-_\s]?left\D*(\d+(?:\.\d+)?)':  "turn-left",
            r'(?:turn|rotat)[-_\s]?right\D*(\d+(?:\.\d+)?)': "turn-right",
            r'move[-_\s]?forward\D*(\d+(?:\.\d+)?)':         "move-forward",
        }
        patterns_without_value = {
            r'(?:turn|rotat)[-_\s]?left':  "turn-left",
            r'(?:turn|rotat)[-_\s]?right': "turn-right",
            r'move[-_\s]?forward':        "move-forward",
        }

        actions: List[str] = []
        for line in response.lower().split("\n"):
            matched = False
            for pattern, action_type in patterns_with_value.items():
                match = re.search(pattern, line)
                if match:
                    try:
                        value = float(match.group(1))
                        if value > 0:
                            if action_type in ("turn-left", "turn-right"):
                                max_turn = int(getattr(self.model_args, "max_turn_angle", 90))
                                value = max(9.0, min(round(value / 9) * 9, float(max_turn)))
                                actions.append(f"{action_type} {int(value)}")
                            else:
                                value = max(0.25, min(round(value / 0.25) * 0.25, 1.0))
                                actions.append(f"{action_type} {value:.2f}")
                            matched = True
                            break
                    except ValueError:
                        continue
            if not matched:
                for pattern, action_type in patterns_without_value.items():
                    if re.search(pattern, line):
                        if action_type in ("turn-left", "turn-right"):
                            actions.append(f"{action_type} 9")
                        else:
                            actions.append(f"{action_type} 0.25")
                        break
            if len(actions) >= 5:
                break

        if not actions:
            self._log_print("[WARN] No actions parsed.", level="WARNING")
            return None
        return actions[:5]

    def _correct_action_values(self, action_sequence: List[str]) -> List[str]:
        """Snap action values to valid increments (9deg / 0.25m)."""
        corrected: List[str] = []
        for action in action_sequence:
            action_type, value_str = action.split()
            value = float(value_str)
            if action_type in ("turn-left", "turn-right"):
                max_turn = int(getattr(self.model_args, "max_turn_angle", 90))
                v = round(value / 9) * 9
                v = max(9, min(v if v > 0 else 9, max_turn))
                corrected.append(f"{action_type} {v}")
            elif action_type == "move-forward":
                v = round(value / 0.25) * 0.25
                v = max(0.25, min(v if v > 0 else 0.25, 1.0))
                corrected.append(f"{action_type} {v:.2f}")
            else:
                corrected.append(action)
        return corrected

    def _validate_action_sequence(self, action_sequence: List[str]) -> bool:
        """Check that an action sequence respects distance/angle budgets."""
        if not action_sequence or len(action_sequence) > 5:
            return False
        total_fwd = 0.0
        total_rot = 0.0
        for action in action_sequence:
            try:
                action_type, value_str = action.split()
                value = float(value_str)
            except (ValueError, AttributeError):
                return False
            if value <= 0:
                return False
            if action_type == "move-forward":
                if value > self.model_args.max_forward_distance + 0.1:
                    return False
                total_fwd += value
            else:
                if value > self.model_args.max_turn_angle:
                    return False
                total_rot += value
        if total_fwd > self.model_args.max_forward_distance + 0.1:
            return False
        if total_rot > self.model_args.max_turn_angle + 5:
            return False
        return True

    # ------------------------------------------------------------------
    # Dump / logging helpers
    # ------------------------------------------------------------------

    def _dump_exploration_check(
        self, save_dir: str, question: str, image_path: str,
        response: str, needs_exploration: bool,
    ) -> None:
        """Persist an exploration-gate decision."""
        check_dir = os.path.join(save_dir, "exploration_checks")
        os.makedirs(check_dir, exist_ok=True)
        with open(os.path.join(check_dir, "exploration_check.json"), "w") as f:
            json.dump({
                "question": question,
                "image_path": image_path,
                "vlm_response": response,
                "needs_exploration": needs_exploration,
                "timestamp": time.time(),
            }, f, indent=2)

    def _dump_llm_interaction(
        self,
        save_dir: str,
        step: Optional[int],
        double_search_step: Optional[int],
        question: Dict[str, Any],
        response: Optional[str],
        result: str,
        actions: List[str],
        magnitude: Optional[float],
    ) -> None:
        """Persist a complete VLM interaction."""
        prompt = copy.deepcopy(self.vlm.curr_prompt)
        log: Dict[str, Any] = {
            "question": question,
            "result": result,
            "action_list": actions,
            "magnitude": magnitude,
            "llm_response": response,
            "prompt": prompt,
        }
        if step is not None and double_search_step is not None:
            step_dir = os.path.join(save_dir, f"step_{step}")
            os.makedirs(step_dir, exist_ok=True)
            out_path = os.path.join(step_dir, f"gpt_{double_search_step}.json")
        else:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, "gpt.json")

        with open(out_path, "w") as f:
            json.dump(log, f, indent=2)

    def _dump_planning_interaction(
        self, save_dir: str, step_idx: int, question: str,
        response: str, action_sequence: List[str],
    ) -> None:
        """Persist an action-planning interaction."""
        planning_dir = os.path.join(save_dir, "planning_logs")
        os.makedirs(planning_dir, exist_ok=True)
        with open(os.path.join(planning_dir, f"step_{step_idx}_planning.json"), "w") as f:
            json.dump({
                "step": step_idx,
                "question": question,
                "planned_actions": action_sequence,
                "llm_response": response,
                "timestamp": time.time(),
            }, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = SpatialVQAPipelineAIVE()
    try:
        pipeline.run()
    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline._close_log()
