"""Smoke checks for the mlx-openai-server console entrypoint."""

import importlib
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from app.cli import cli


def test_cli_help_lists_launch():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "launch" in result.output


def test_launch_help():
    result = CliRunner().invoke(cli, ["launch", "--help"])
    assert result.exit_code == 0
    assert "--model-path" in result.output
    assert "--model-type" in result.output


def test_launch_starts_server_through_create_app():
    """launch must use the current FastAPI app factory, not the removed setup_server."""
    runner = CliRunner()
    fake_app = MagicMock(name="app")
    with (
        patch("app.cli.create_app", return_value=fake_app) as create_app,
        patch("app.cli.uvicorn.run") as run,
    ):
        result = runner.invoke(
            cli,
            [
                "launch",
                "--model-path",
                "mlx-community/Qwen3.5-4B-4bit",
                "--model-type",
                "lm",
                "--host",
                "127.0.0.1",
                "--port",
                "9001",
                "--context-length",
                "4096",
                "--no-log-file",
                "--log-level",
                "ERROR",
            ],
        )

    assert result.exit_code == 0, result.output
    create_app.assert_called_once()
    args = create_app.call_args.args[0]
    assert args.model_path == "mlx-community/Qwen3.5-4B-4bit"
    assert args.model_type == "lm"
    assert args.host == "127.0.0.1"
    assert args.port == 9001
    assert args.context_length == 4096
    assert args.no_log_file is True
    assert args.log_level == "ERROR"
    assert args.mlx_warmup is True

    run.assert_called_once_with(
        fake_app,
        host="127.0.0.1",
        port=9001,
        log_level="info",
        access_log=True,
    )


def test_launch_reports_startup_failure():
    runner = CliRunner()
    with (
        patch("app.cli.create_app", side_effect=RuntimeError("boom")),
        patch("app.cli.uvicorn.run") as run,
        patch("app.cli.logger.error") as log_error,
    ):
        result = runner.invoke(
            cli,
            ["launch", "--model-path", "some/model", "--no-log-file"],
        )

    assert result.exit_code == 1
    assert any("boom" in str(call) for call in log_error.call_args_list)
    run.assert_not_called()


def test_mlx_lm_supports_qwen3_5_architectures():
    """mlx-lm must import the architectures that Qwen3.5 configs declare."""
    from mlx_lm.utils import _get_classes

    for model_type in ("qwen3_5", "qwen3_5_moe"):
        model_cls, model_args_cls = _get_classes({"model_type": model_type})
        assert model_cls.__module__ == f"mlx_lm.models.{model_type}"
        assert model_args_cls is not None
        # The loader used to raise "Model type qwen3_5 not supported."
        assert importlib.import_module(f"mlx_lm.models.{model_type}").Model is model_cls
