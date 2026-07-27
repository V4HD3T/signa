"""Loading a checkpoint must not be able to execute code.

A .pt file is a pickle. `torch.load` with the default (or explicit
weights_only=False) will happily run whatever a crafted file tells it to, and
this project's checkpoints are exactly the kind of artifact people share --
"try my trained model" is a normal sentence. These tests fail if that door is
reopened, which is the only way anyone would notice: a permissive load looks
identical to a safe one until the day it doesn't.
"""

import pytest
import torch

from signa.demo import load_checkpoint


class _Exploit:
    """Stands in for a malicious payload: unpickling this calls os.system.

    __reduce__ is the pickle hook an attacker uses -- it names a callable and
    its arguments, and the unpickler obligingly calls it. Here the callable is
    harmless (`print`), because the point is to prove the *mechanism* is blocked,
    not to run anything."""

    def __reduce__(self):
        return (print, ("PICKLE PAYLOAD EXECUTED",))


def _write_malicious(path):
    """A well-formed torch checkpoint that carries a pickle payload."""
    torch.save({"model_state": {}, "labels": ["a"], "config": _Exploit()}, path)
    return path


def test_a_checkpoint_with_a_payload_is_refused(tmp_path, capsys):
    path = _write_malicious(tmp_path / "malicious.pt")

    # The restricted unpickler must refuse it rather than calling the payload.
    with pytest.raises(Exception):
        load_checkpoint(path, "cpu")

    assert "PICKLE PAYLOAD EXECUTED" not in capsys.readouterr().out


def test_the_same_file_does_run_its_payload_when_unpickling_is_unrestricted(tmp_path, capsys):
    # The counterexample, so the test above is demonstrably testing something.
    # (torch may still reject the file afterwards -- note the payload runs
    # *during* unpickling, i.e. before any validation could save you.)
    path = _write_malicious(tmp_path / "malicious.pt")

    try:
        torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        pass
    assert "PICKLE PAYLOAD EXECUTED" in capsys.readouterr().out


def test_a_normal_checkpoint_still_loads(tmp_path):
    from signa import models
    from signa.config import Config

    cfg = Config(model="tcn")
    model = models.build(cfg, num_classes=3)
    path = tmp_path / "best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "labels": ["a", "b", "c"],
        "config": {"model": "tcn", "hidden": 128, "layers": 2, "dropout": 0.3,
                   "heads": 4, "frames": 48, "normalize": True, "use_pose": True},
    }, path)

    loaded, labels, restored = load_checkpoint(path, "cpu")
    assert labels == ["a", "b", "c"]
    assert restored.model == "tcn" and restored.frames == 48
    assert loaded(torch.randn(1, 48, 152)).shape == (1, 3)
