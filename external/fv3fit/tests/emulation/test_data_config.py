import pytest
from fv3fit._shared import SliceConfig
import dataclasses
from fv3fit.emulation.data.config import TransformConfig


@pytest.mark.parametrize("start", [None, 1])
@pytest.mark.parametrize("stop", [None, 5])
@pytest.mark.parametrize("step", [None, 2])
def test_SliceConfig(start, stop, step):
    expected = slice(start, stop, step)
    config = SliceConfig(start=start, stop=stop, step=step)
    assert config.slice == expected


def _get_config() -> TransformConfig:
    return TransformConfig(
        antarctic_only=False, vertical_subselections={"a": SliceConfig(start=5)},
    )


def test_TransformConfig():
    transform = _get_config()
    assert transform.vert_sel_as_slices["a"] == slice(5, None)


def test_TransformConfig_from_dict():
    transform_in = _get_config()
    transform_from_dict = TransformConfig.from_dict(dataclasses.asdict(transform_in))
    assert transform_from_dict == transform_in


def test_TransformConfig_get_dataset_names():
    class Mock:
        def forward(self, x):
            return x

        def backward_names(self, required):
            return required | {"z"}

    config = TransformConfig(tensor_transforms=[Mock()])
    assert config.get_dataset_names(set()) == {"z"}
