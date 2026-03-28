from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from evals.contracts import OfflineEvalRun, SkippedEvalCase, summarize_offline_results
from evals.scorers import score_offline_case
from model.contracts import ModelBackend, PolicyDeskAssistantRequest, ScenarioName
from services.qa_service import PolicyDeskService
from services.storage_service import StorageService


OPENAI_SUPPORTED_SCENARIOS = {
    ScenarioName.NORMAL,
    ScenarioName.RETRIEVAL_MISS,
    ScenarioName.CONFLICTING_EVIDENCE,
    ScenarioName.SLOW_RESPONSE,
}


def load_eval_cases(dataset_path: str = "data/eval_dataset.json"):
    with Path(dataset_path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return payload


class OfflineEvalRunner:
    def __init__(self, policy_service: PolicyDeskService, storage: StorageService, dataset_path: str = "data/eval_dataset.json") -> None:
        self.policy_service = policy_service
        self.storage = storage
        self.dataset_path = dataset_path

    def run(self, *, model_backend, prompt_version: str) -> OfflineEvalRun:
        eval_run_id = str(uuid4())
        raw_cases = load_eval_cases(self.dataset_path)
        case_results = []
        skipped_cases: list[SkippedEvalCase] = []
        model_name = "mock-policy-model"
        for raw_case in raw_cases:
            scenario = ScenarioName(raw_case["scenario"])
            if model_backend == ModelBackend.OPENAI and scenario not in OPENAI_SUPPORTED_SCENARIOS:
                skipped_cases.append(
                    SkippedEvalCase(
                        case_id=raw_case["case_id"],
                        scenario=scenario,
                        reason="Scenario is supported only on the mock backend.",
                    )
                )
                continue
            request = PolicyDeskAssistantRequest(
                question=raw_case["question"],
                scenario=scenario,
                model_backend=model_backend,
                prompt_version=prompt_version,
            )
            response = self.policy_service.respond(request)
            run_record = self.storage.get_run(response.run_id)
            assert run_record is not None
            model_name = run_record.model_name
            result = score_offline_case(eval_run_id=eval_run_id, raw_case=raw_case, response=response, run_record=run_record)
            case_results.append(result)
        summary = summarize_offline_results(
            eval_run_id=eval_run_id,
            model_backend=model_backend,
            model_name=model_name,
            prompt_version=prompt_version,
            case_results=case_results,
            skipped_cases=skipped_cases,
        )
        self.storage.write_offline_eval(summary, case_results)
        return OfflineEvalRun(summary=summary, case_results=case_results)
