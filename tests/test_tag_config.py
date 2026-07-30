import pytest

from pca_model_builder.tag_config import engineering_ranges, normalize_tag_configs


def test_tag_config_normalizes_metadata_and_engineering_ranges():
    configs = normalize_tag_configs(
        ["TI001", "PI001"],
        {
            "TI001": {
                "description": "反应温度",
                "unit": "℃",
                "type": "continuous",
                "engineering_min": 0,
                "engineering_max": 300,
                "normal_min": 120,
                "normal_max": 180,
                "alarm_min": 80,
                "alarm_max": 220,
            },
            "PI001": {},
        },
    )

    assert configs["TI001"]["description"] == "反应温度"
    assert configs["PI001"]["engineering_min"] is None
    assert engineering_ranges(configs) == {"TI001": (0.0, 300.0)}


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"TI001": {"engineering_min": 0}}, "requires both"),
        (
            {"TI001": {"normal_min": 180, "normal_max": 120}},
            "lower value must be less",
        ),
        ({"TI001": {"type": "discrete"}}, "type continuous"),
        ({"UNKNOWN": {}}, "unselected Tags"),
    ],
)
def test_tag_config_rejects_invalid_configuration(raw, message):
    with pytest.raises(ValueError, match=message):
        normalize_tag_configs(["TI001"], raw)
