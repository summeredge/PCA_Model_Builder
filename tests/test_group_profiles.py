import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from pca_model_builder.cluster_quality import analyze_cluster_quality
from pca_model_builder.state_exploration import _attach_group_time_profiles
from pca_model_builder.web import _cluster_result_payload
from pca_model_builder.clustering import OperatingStateClusters
from pca_model_builder import state_exploration, web_model_results
from pca_model_builder.preprocessing import PreprocessingConfig


def test_weighted_signed_profiles_ties_constants_missing_and_middle_group():
    labels = ['a', 'a', 'a', 'b', 'c']
    raw = pd.DataFrame({'x': [0, 0, 0, 5, 10], 'tie': [1, 1, 1, 1, 2],
                        'constant': [7] * 5, 'missing': [np.nan] * 5})
    scores = pd.DataFrame({'pc1': [0, 0, 0, 1, 2], 'pc2': [0] * 5})
    quality = analyze_cluster_quality(scores, labels, raw_data=raw, feature_names=raw.columns)
    profiles = {p['cluster_id']: p for p in quality['group_profiles']}
    x = profiles['a']['variables'][0]
    assert x['overall_mean'] == 3  # All samples, not equally weighted group means.
    assert x['deviation'] == pytest.approx(-3 / raw.x.std(ddof=0))
    assert profiles['c']['variables'][0]['rank'] == 1
    assert profiles['a']['variables'][1]['tied']
    assert profiles['a']['variables'][2]['deviation'] is None
    assert profiles['a']['variables'][3]['mean'] is None
    assert all(v['deviation'] not in (None, 0) for v in profiles['a']['top_variables'])
    json.dumps(quality, allow_nan=False)
    middle_raw = pd.DataFrame({'x': [0, 5, 10]})
    middle = analyze_cluster_quality(scores.iloc[:3], ['a', 'b', 'c'], raw_data=middle_raw, feature_names=['x'])
    p = middle['group_profiles'][1]
    assert p['top_variables'] == []
    assert p['comparison_variables'][0]['mean'] == 5
    assert p['comparison_variables'][0]['rank'] == 2


def test_episodes_segments_gaps_transitions_coverage_and_candidates():
    index = pd.to_datetime(['2026-01-01 00:00', '2026-01-01 00:05', '2026-01-01 00:10',
                            '2026-01-01 00:15', '2026-01-01 01:00', '2026-01-01 01:05'])
    points = pd.DataFrame({'cluster_id': ['a', 'a', 'b', 'a', 'b', 'a'],
                           'segment_id': [1, 1, 2, 2, 3, 3]}, index=index)
    quality = {'group_profiles': [{'cluster_id': 'a'}, {'cluster_id': 'b'}]}
    candidates = [{'cluster_id': 'a', 'candidate_id': 'existing-id', 'source': 'cluster'}]
    original = json.dumps(candidates)
    _attach_group_time_profiles(quality, points, candidates, 5)
    a, b = quality['group_profiles']
    assert a['episode_count'] == 3
    assert a['total_duration_minutes'] == 20
    assert a['median_duration_minutes'] == 5
    assert a['longest_duration_minutes'] == 10
    assert a['candidate_count'] == 1
    assert b['total_duration_minutes'] == 10
    assert quality['transitions'] == [{'from': 'b', 'to': 'a', 'count': 2}]
    assert sum(sum(e['duration_minutes'] for e in p['episodes']) for p in quality['group_profiles']) == len(points) * 5
    assert json.dumps(candidates) == original


def test_assistance_and_shared_profiles_use_full_aligned_engineering_values():
    index = pd.date_range('2026-01-01', periods=6, freq='5min')
    points = pd.DataFrame({'pc1': [-2, -1, -1, 1, 1, 2], 'pc2': [1, 2, 3, 2, 3, 1],
                           'cluster': [1, 1, 1, 2, 2, 2]}, index=index)
    raw = pd.DataFrame({'tag': [10, 20, 30, 40, 50, 60], 'performance': [999] * 6}, index=index).iloc[::-1]
    centers = {1: np.array([-1.3, 2]), 2: np.array([1.3, 2])}
    result = OperatingStateClusters(points, (), 2, .95, centers, ('pc1', 'pc2'), (.6, .35))
    assistance = _cluster_result_payload(result, 5, raw, ['tag'])['cluster_quality']
    shared = analyze_cluster_quality(points[['pc1', 'pc2']], points.cluster, index, raw, ['tag'], [.6, .35], 5, centers)
    assert assistance['group_profiles'] == shared['group_profiles']
    assert assistance['group_profiles'][0]['variables'][0]['mean'] == 20
    assert assistance['group_profiles'][1]['variables'][0]['mean'] == 50
    assert all(len(profile['variables']) == 1 for profile in assistance['group_profiles'])


@pytest.mark.parametrize('direction', [None, 'target_range', 'higher_is_better'])
def test_profile_reuses_summary_performance_only_for_target_range(direction):
    rng = np.random.default_rng(17)
    index = pd.date_range('2026-01-01', periods=90, freq='5min')
    frame = pd.DataFrame(rng.normal(size=(90, 3)), index=index, columns=['A', 'B', 'C'])
    frame['B'] = 2 * frame['A'] + rng.normal(scale=.01, size=90)
    frame['P'] = np.arange(90, dtype=float)
    config = None if direction is None else state_exploration.PerformanceConfig(
        'P', direction, target_min=20 if direction == 'target_range' else None,
        target_max=60 if direction == 'target_range' else None,
    )
    result = state_exploration.run_state_exploration(
        frame, ['A', 'B', 'C'], PreprocessingConfig(5, 0, 0, 5, filter_method='none'),
        state_exploration.ExplorationConfig(cluster_count=2), performance_config=config,
    )
    summaries = {item['cluster_id']: item for item in result['cluster_summaries']}
    keys = ('performance_valid_count', 'performance_target_count', 'performance_target_ratio', 'performance_median')
    for profile in result['cluster_quality']['group_profiles']:
        if direction == 'target_range':
            assert {key: profile[key] for key in keys} == {key: summaries[profile['cluster_id']][key] for key in keys}
        else:
            assert not any(key in profile for key in keys)
        assert profile['episodes']
        assert 'incoming' in profile and 'outgoing' in profile


def test_profile_renderer_directions_performance_visibility_and_time_scope():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js unavailable')
    html = web_model_results.INDEX_HTML
    renderer = html[html.index('function groupProfileNumber'):html.index('function applyGroupProfileFocus')]
    source = r"""
const assert=require('node:assert/strict');
const state={groupProfileSelection:{state_exploration:'a',cluster_assistance:'a'}};
const escapeHtml=value=>String(value).replaceAll('<','&lt;');
const clusterUiLabel=value=>value;
const explorationPercent=value=>`${(value*100).toFixed(1)}%`;
const metric=(label,value)=>`${label}：${value}`;
const variables=[{tag:'FIC001',mean:1,deviation:1,rank:1,rank_count:2},{tag:'TIC003',mean:2,deviation:-1,rank:2,rank_count:2}];
const profile={cluster_id:'a',sample_count:5,share:.5,variables,top_variables:variables,comparison_variables:[],episode_count:2,median_duration_minutes:5,longest_duration_minutes:10,total_duration_minutes:15,average_duration_minutes:7.5,candidate_count:0,incoming:[],outgoing:[],performance_valid_count:5,performance_target_count:3,performance_target_ratio:.6,performance_median:42};
function render(perspective,profile) {
 const detail={innerHTML:'',querySelector:()=>null};
 const section={innerHTML:'',querySelectorAll:()=>[],querySelector:()=>detail};
 globalThis.document={createElement:()=>section};
 const container={children:[],querySelector:()=>null,append(node){this.children.push(node);}};
 renderGroupProfiles(container,{group_profiles:[profile],transitions:[]},perspective);
 return {list:section.innerHTML,detail:detail.innerHTML};
}
""" + renderer + r"""
const result=render('state_exploration',profile);
assert(result.list.includes('FIC001 ↑、TIC003 ↓'));
assert(result.detail.includes('工程量均值')&&result.detail.includes('相对总体偏离 z'));
assert(result.detail.includes('性能后验评价'));
assert(result.detail.includes('有效样本：5达标样本：3达标率：60.0%中位数：42'));
assert(result.detail.includes('不参与 PCA 聚类')&&result.detail.includes('不代表该工况自动属于正常/异常'));
assert(result.detail.includes('总覆盖 15 分钟')&&result.detail.includes('主要进入')&&result.detail.includes('定位时间轴'));
assert(result.detail.includes('完整变量对比')&&result.detail.includes('完整转换计数'));
assert(result.detail.indexOf('定位PC图')<result.detail.indexOf('定位时间轴')&&result.detail.indexOf('定位时间轴')<result.detail.indexOf('查看关联候选'));
const noPerformance={...profile};for(const key of Object.keys(noPerformance)) if(key.startsWith('performance_')) delete noPerformance[key];
assert(!render('state_exploration',noPerformance).detail.includes('性能后验评价'));
assert(!render('state_exploration',{...profile,performance_valid_count:0,performance_target_ratio:null}).detail.includes('性能后验评价'));
const assistance=render('cluster_assistance',profile);
assert(!assistance.detail.includes('性能后验评价'));
assert(!assistance.detail.includes('总覆盖')&&!assistance.detail.includes('主要进入')&&!assistance.detail.includes('定位时间轴'));
assert(!assistance.detail.includes('定位PC图')&&!assistance.detail.includes('查看关联候选'));
"""
    result = subprocess.run([node, '-'], input=source, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stderr
