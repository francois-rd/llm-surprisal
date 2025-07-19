from vec_inf.client.api import VecInfClient, ModelStatus
from retry import retry

from ..io import get_logger


def await_server(
    slurm_job_id: int, delay: int = 60, logger_name: str | None = None
) -> str:
    if logger_name:
        get_logger(logger_name).info("Starting server monitoring...")

    @retry(exceptions=AssertionError, delay=delay)
    def assert_ready():
        client = VecInfClient()
        assert client.get_status(slurm_job_id).server_status == ModelStatus.READY

    if logger_name:
        get_logger(logger_name).info("Done. Server is running.")

    status = VecInfClient().get_status(slurm_job_id)
    assert status.server_status == ModelStatus.READY
    if status.base_url is None:
        raise ValueError(f"Server has ready status but no base URL: {status}")
    return status.base_url
