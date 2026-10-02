import numpy as np
import pandas as pd
import pytest

from pca_model_builder.screening import screen_performance_states


def test_performance_conditions_use_inclusive_and_logic():
    frame = pd.DataFrame(
        {
            "yield": [90.0, 92.0, 94.0, 96.0, 98.0],
            "consumption": [12.0, 11.0, 10.0, 9.0, 8.0],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="5min"),
    )

    result = screen_performance_states(
        frame,
        [
            {"column": "yield", "minimum": 94},
            {"column": "consumption", "maximum": 10},
        ],
        sample_interval_minutes=5,
    )

    assert result["matched_rows"] == 3
    assert result["match_share"] == pytest.approx(0.6)
    assert [item["matched_rows"] for item in result["conditions"]] == [3, 3]
    assert result["representative_windows"] == [
        {
            "start": "2026-01-01T00:10:00",
            "end": "2026-01-01T00:20:00",
            "count": 3,
        }
    ]
    assert result["engineer_decision_required"] is True


def test_performance_windows_do_not_cross_physical_time_gaps():
    index = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:30",
            "2026-01-01 00:35",
        ]
    )
    frame = pd.DataFrame({"quality": [1.0, 1.0, 1.0, 1.0]}, index=index)

    result = screen_performance_states(
        frame,
        [{"column": "quality", "minimum": 1.0, "maximum": 1.0}],
        sample_interval_minutes=5,
    )

    assert [window["count"] for window in result["representative_windows"]] == [2, 2]


def test_performance_screen_rejects_empty_data():
    frame = pd.DataFrame(
        {"yield": pd.Series(dtype=float)},
        index=pd.DatetimeIndex([]),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        screen_performance_states(
            frame,
            [{"column": "yield", "minimum": 1}],
            sample_interval_minutes=5,
        )


@pytest.mark.parametrize(
    "conditions, message",
    [
        ([], "at least one"),
        ([{"column": "yield"}], "requires a bound"),
        ([{"column": "yield", "minimum": 2, "maximum": 1}], "reversed"),
        (
            [
                {"column": "yield", "minimum": 1},
                {"column": "yield", "maximum": 2},
            ],
            "cannot be repeated",
        ),
    ],
)
def test_performance_screen_rejects_invalid_conditions(conditions, message):
    frame = pd.DataFrame(
        {"yield": [1.0, 2.0]},
        index=pd.date_range("2026-01-01", periods=2, freq="5min"),
    )

    with pytest.raises(ValueError, match=message):
        screen_performance_states(frame, conditions, sample_interval_minutes=5)


def _parent(frame, identifier="parent-1", start=0, end=-1, source="manual", source_ref=None):
    return {"id": identifier, "start": frame.index[start].isoformat(),
            "end": frame.index[end].isoformat(), "source": source, "source_ref": source_ref}


def _refinement(frame, parents, **kwargs):
    return screen_performance_states(frame, [{"column": "quality", "minimum": 1}], 5,
                                     scope={"type": "candidate_windows"}, parent_windows=parents, **kwargs)


def test_default_scope_preserves_old_result_and_adds_stable_candidate_identity():
    frame = pd.DataFrame({"quality": [1, 0, 1]}, index=pd.date_range("2026-01-01", periods=3, freq="5min"))
    result = screen_performance_states(frame, [{"column": "quality", "minimum": 1}], 5)
    explicit = screen_performance_states(frame, [{"column": "quality", "minimum": 1}], 5, scope={"type": "all"})
    assert result == explicit
    assert result["total_rows"] == 3 and result["matched_rows"] == 2
    assert all(set(window) == {"start", "end", "count"} for window in result["representative_windows"])
    assert len({window["source_ref"] for window in result["candidate_windows"]}) == 2
    assert result["parent_count"] == 0


def test_single_parent_intersection_and_provenance_are_stable():
    frame = pd.DataFrame({"quality": [1, 1, 0, 1, 1, 1]}, index=pd.date_range("2026-01-01", periods=6, freq="5min"))
    parent = _parent(frame, start=1, end=4, source="cluster", source_ref="state-exploration-run-abc-cluster_2-candidate-003")
    result = _refinement(frame, [parent])
    assert result["total_rows"] == 4 and result["matched_rows"] == 3
    assert [window["count"] for window in result["candidate_windows"]] == [2, 1]
    assert result["original_rows"] == 4 and result["derived_candidate_count"] == 2
    provenance = result["candidate_windows"][0]["provenance"]
    assert provenance == {"scope_type": "candidate_windows", "parent_candidate_id": "parent-1",
                          "parent_source": "cluster", "parent_source_ref": parent["source_ref"],
                          "conditions": [{"column": "quality", "minimum": 1.0, "maximum": None}],
                          "ordinal_in_parent": 1, "cluster_id": "cluster_2", "exploration_run_id": "run-abc",
                          "origin_candidate_id": "cluster_2-candidate-003", "origin_source": "cluster",
                          "origin_source_ref": parent["source_ref"]}
    assert result == _refinement(frame, [parent])
    assert parent["source"] == "cluster"  # Input parents remain unchanged.


def test_multiple_parents_never_merge_across_parent_boundaries():
    frame = pd.DataFrame({"quality": [1] * 6}, index=pd.date_range("2026-01-01", periods=6, freq="5min"))
    result = _refinement(frame, [_parent(frame, "first", 0, 2), _parent(frame, "second", 3, 5)])
    assert [window["count"] for window in result["candidate_windows"]] == [3, 3]
    assert [window["provenance"]["parent_candidate_id"] for window in result["candidate_windows"]] == ["first", "second"]
    assert result["parent_count"] == 2 and result["matched_parent_count"] == 2
    assert len({window["source_ref"] for window in result["candidate_windows"]}) == 2


def test_overlapping_parents_are_processed_separately_with_distinct_identity():
    frame = pd.DataFrame({"quality": [1] * 3}, index=pd.date_range("2026-01-01", periods=3, freq="5min"))
    result = _refinement(frame, [_parent(frame, "first"), _parent(frame, "second")])
    assert len(result["candidate_windows"]) == 2
    assert len({window["source_ref"] for window in result["candidate_windows"]}) == 2
    assert result["total_rows"] == 6 and result["matched_rows"] == 6


def test_cluster_scope_uses_only_traceable_current_parent_ranges():
    frame = pd.DataFrame({"quality": [1] * 6}, index=pd.date_range("2026-01-01", periods=6, freq="5min"))
    parents = [_parent(frame, "cluster2", 0, 1, "cluster", "cluster-2-2026-01-01T00:00:00-2026-01-01T00:05:00"),
               _parent(frame, "cluster3", 2, 3, "cluster", "state-exploration-run-cluster_3-candidate-001"),
               {**_parent(frame, "untraceable", 4, 5), "cluster_id": "cluster_2"}]
    result = screen_performance_states(frame, [{"column": "quality", "minimum": 1}], 5,
                                     scope={"type": "cluster", "cluster_ids": ["cluster_2"]}, parent_windows=parents)
    assert result["parent_count"] == 1 and result["matched_rows"] == 2
    assert result["candidate_windows"][0]["provenance"]["parent_candidate_id"] == "cluster2"
    with pytest.raises(ValueError, match="no traceable"):
        screen_performance_states(frame, [{"column": "quality", "minimum": 1}], 5,
                                  scope={"type": "cluster", "cluster_ids": ["cluster_9"]}, parent_windows=parents)


def test_eligibility_exclusion_and_engineering_conditions_split_parent():
    frame = pd.DataFrame({"quality": [1, 1, 1, 1, 0, 1, 1], "gate": [1, 1, 0, 1, 1, 1, 1]},
                         index=pd.date_range("2026-01-01", periods=7, freq="5min"))
    result = _refinement(frame, [_parent(frame)], modeling_eligibility={"exclude_rule_groups": [[{"column": "gate", "maximum": 0}]]})
    assert result["original_rows"] == 7 and result["total_rows"] == 6 and result["matched_rows"] == 5
    assert [window["count"] for window in result["candidate_windows"]] == [2, 2, 1]
    assert result["derived_candidate_count"] == 3


@pytest.mark.parametrize("scoped", [False, True])
def test_deleted_source_rows_do_not_reconnect_on_target_sampling_grid(scoped):
    frame = pd.DataFrame({"quality": [1] * 6, "gate": [1, 0, 0, 0, 0, 1]},
                         index=pd.date_range("2026-01-01", periods=6, freq="1min"))
    options = {"scope": {"type": "candidate_windows"}, "parent_windows": [_parent(frame)]} if scoped else {}
    result = screen_performance_states(frame, [{"column": "quality", "minimum": 1}], 5,
                                      modeling_eligibility={"keep_conditions": [{"column": "gate", "minimum": 1}]}, **options)
    assert [window["count"] for window in result["representative_windows"]] == [1, 1]


def test_refinement_respects_physical_gaps_and_returns_all_segments():
    index = pd.date_range("2026-01-01", periods=24, freq="5min").delete(2)
    frame = pd.DataFrame({"quality": [1 if n % 2 == 0 else 0 for n in range(len(index))]}, index=index)
    result = _refinement(frame, [_parent(frame)])
    assert result["derived_candidate_count"] == 12  # Refinement must not silently keep only the top ten.
    assert all(window["count"] == 1 for window in result["candidate_windows"])
    frame["quality"] = 1
    result = _refinement(frame, [_parent(frame)])
    assert sorted(window["count"] for window in result["candidate_windows"]) == [2, 21]


def test_empty_and_partial_parent_results_have_clear_counts():
    frame = pd.DataFrame({"quality": [1, 1, 0, 0]}, index=pd.date_range("2026-01-01", periods=4, freq="5min"))
    parents = [_parent(frame, "yes", 0, 1), _parent(frame, "no", 2, 3)]
    result = _refinement(frame, parents)
    assert result["parent_count"] == 2 and result["matched_parent_count"] == 1 and result["unmatched_parent_count"] == 1
    assert len(result["candidate_windows"]) == 1
    result = _refinement(frame, [parents[1]])
    assert result["candidate_windows"] == [] and result["match_share"] == 0
    result = _refinement(frame, parents, modeling_eligibility={"keep_conditions": [{"column": "quality", "minimum": 2}]})
    assert result["total_rows"] == 0 and result["candidate_windows"] == []


def test_refinement_validates_engineering_numbers_only_inside_selected_scope():
    frame = pd.DataFrame({"quality": [np.nan, 1, 1]}, index=pd.date_range("2026-01-01", periods=3, freq="5min"))
    result = _refinement(frame, [_parent(frame, start=1)])
    assert result["matched_rows"] == 2
    with pytest.raises(ValueError, match="finite"):
        _refinement(frame, [_parent(frame)])


def test_refined_cluster_parent_retains_original_exploration_source():
    frame = pd.DataFrame({"quality": [1] * 3}, index=pd.date_range("2026-01-01", periods=3, freq="5min"))
    first = _refinement(frame, [_parent(frame, source="cluster", source_ref="state-exploration-run-cluster_2-candidate-001")])["candidate_windows"][0]
    parent = {**first, "id": first["candidate_id"]}
    result = screen_performance_states(frame, [{"column": "quality", "maximum": 1}], 5,
                                      scope={"type": "cluster", "cluster_ids": ["cluster_2"]}, parent_windows=[parent])
    provenance = result["candidate_windows"][0]["provenance"]
    assert provenance["parent_candidate_id"] == first["candidate_id"]
    assert provenance["parent_source"] == "performance"
    assert provenance["cluster_id"] == "cluster_2" and provenance["exploration_run_id"] == "run"
    assert provenance["parent_provenance"] == first["provenance"]


@pytest.mark.parametrize("scope,parents", [({"type": "unknown"}, []), ({"type": "candidate_windows"}, []),
                                           ({"type": "cluster", "cluster_ids": []}, [])])
def test_refinement_rejects_invalid_scope(scope, parents):
    frame = pd.DataFrame({"quality": [1] * 3}, index=pd.date_range("2026-01-01", periods=3, freq="5min"))
    with pytest.raises(ValueError):
        screen_performance_states(frame, [{"column": "quality", "minimum": 1}], 5, scope=scope, parent_windows=parents)
