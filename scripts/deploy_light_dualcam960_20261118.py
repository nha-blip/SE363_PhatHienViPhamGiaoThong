"""Upload the light DualCam@960 fine-tune package to Kaggle2 (v5nhdf, private) and submit the GPU notebook.

Without --execute only preflight checks run (identity, quota, input datasets private, payload).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_yolo26_fullscene_resume_20261028 import canonical_notebook, read_call  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPLOAD = ROOT / "exports/yolo26_light_dualcam960_20261118/upload"
CODE = ROOT / "config/kaggle/yolo26_light_dualcam960_20261118/train.py"
REPORT = ROOT / "runs/project363/yolo26_light_dualcam960_20261118"
OWNER = "v5nhdf"
DATASET = "yolo363-light-dualcam-20261118"
NOTEBOOK = "yolo26-light-dualcam960-20261118"
INPUTS = [f"{OWNER}/yolo363-fullscene-resume-20261028", f"{OWNER}/yolo363-light-yellow-20261116", f"{OWNER}/yolo363-light-vnpool-20261117", f"{OWNER}/{DATASET}"]
SESSION_TIMEOUT_SECONDS = 16200  # 3.5 h training budget + extraction and before/after evaluation on 3 vals
EXPECTED_FILES = {"dataset-metadata.json", "package.json", "dualcam.zip.bin"}


def main():
    import winreg
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiGetDatasetStatusRequest
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest, ApiGetKernelRequest, ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    from requests import HTTPError

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if {p.name for p in UPLOAD.iterdir()} != EXPECTED_FILES:
        raise RuntimeError("Unexpected upload payload")
    code = CODE.read_text(encoding="utf-8")
    compile(code, "train.py", "exec")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "submitted_train.py").write_text(code, encoding="utf-8")
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        token = winreg.QueryValueEx(key, "KAGGLE_API_TOKEN_2")[0].strip()
    os.environ.update(KAGGLE_API_TOKEN=token, KAGGLE_API_ENVIRONMENT="PROD", KAGGLE_CONFIG_DIR=str(ROOT / ".tools/kaggle2_profile"))
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        os.environ.pop(name, None)
    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        query = IntrospectTokenRequest()
        query.token = token
        identity = read_call(client.security.oauth_client.introspect_token, query)
        if not identity.active or identity.username != OWNER:
            raise RuntimeError("Wrong Kaggle2 identity")
        quota = read_call(client.kernels.kernels_api_client.get_accelerator_quota_statistics, ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        remaining = quota.total_time_allowed.total_seconds() - quota.time_used.total_seconds()
        print(json.dumps({"account": OWNER, "remaining_gpu_hours": round(remaining / 3600, 2), "session_timeout_hours": SESSION_TIMEOUT_SECONDS / 3600}), flush=True)
        if quota.time_reserved.total_seconds() != 0 or remaining < SESSION_TIMEOUT_SECONDS:
            raise RuntimeError("GPU quota is reserved or too low for the planned session")
        source = ApiGetDatasetRequest()
        source.owner_slug, source.dataset_slug = INPUTS[0].split("/")
        if not read_call(client.datasets.dataset_api_client.get_dataset, source).is_private:
            raise RuntimeError("Input dataset is not private")
        if not arguments.execute:
            print("PREFLIGHT_OK_NO_UPLOAD_NO_SUBMIT", flush=True)
            return
        dataset = ApiGetDatasetRequest()
        dataset.owner_slug, dataset.dataset_slug = OWNER, DATASET
        try:
            read_call(client.datasets.dataset_api_client.get_dataset, dataset)
            raise RuntimeError("Dataset already exists; refusing to overwrite")
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
        from kaggle import api

        result = api.dataset_create_new(str(UPLOAD), public=False, quiet=True, dir_mode="skip")
        if result.error:
            raise RuntimeError("Dataset rejected")
        status = ApiGetDatasetStatusRequest()
        status.owner_slug, status.dataset_slug = OWNER, DATASET
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                name = read_call(client.datasets.dataset_api_client.get_dataset_status, status).status.name
            except HTTPError as error:
                if error.response.status_code not in (403, 404):
                    raise
                name = "PENDING"
            if name in {"READY", "COMPLETE"}:
                break
            if "ERROR" in name or "FAIL" in name:
                raise RuntimeError("Dataset processing failed")
            time.sleep(30)
        else:
            raise RuntimeError("Dataset not ready in time")
        kernel = ApiGetKernelRequest()
        kernel.user_name, kernel.kernel_slug = OWNER, NOTEBOOK
        try:
            read_call(client.kernels.kernels_api_client.get_kernel, kernel)
            raise RuntimeError("Notebook already exists; refusing to submit twice")
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
        save = ApiSaveKernelRequest()
        save.slug, save.new_title = f"{OWNER}/{NOTEBOOK}", "YOLO26 Light DualCam960 20261118"
        save.text, save.language, save.kernel_type = code, "python", "script"
        save.is_private, save.enable_gpu, save.enable_tpu, save.enable_internet = True, True, False, True
        save.machine_shape, save.session_timeout_seconds = "NvidiaTeslaT4", SESSION_TIMEOUT_SECONDS
        save.dataset_data_sources = INPUTS
        save.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
        saved = client.kernels.kernels_api_client.save_kernel(save)
        if saved.error or saved.invalid_dataset_sources:
            raise RuntimeError("Notebook rejected")
        reference = canonical_notebook(saved.url)
        (REPORT / "submission.json").write_text(json.dumps({"url": saved.url, "reference": reference, "version": saved.version_number}, indent=2))
        print(json.dumps({"SUBMITTED": reference, "version": saved.version_number}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"STOPPED_SAFELY": type(error).__name__, "reason": str(error) if isinstance(error, (ValueError, RuntimeError)) else None}), flush=True)
        sys.exit(1)
