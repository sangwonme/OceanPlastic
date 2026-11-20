"""outputs/features.csv → '쓰레기 많음(주의 요함)' 이진 분류 (기본) | `python train.py reg`: 회귀

평가 원칙 (CLAUDE.md 참고): 주 지표는 '새 지역' — 10km 안 해변을 한 묶음으로 GroupKFold (spatial_groups).
해변 단위 분리는 이웃 해변 정보가 새어 AUC를 약 0.1 과대평가함.

검증된 레시피 (outputs/analysis/A4·A5·A10·A13):
  - 타깃 = log1p(무게 / 세션 수)  ← 노력량 표준화. 세션 수 자체는 X에 안 넣음
  - 피처 = USE 블록만 (도움이 된 블록. 전체 블록을 넣으면 94개 해변에서 과적합)
  - 모델 = PARAMS_FINAL (강한 규제). fold 안 자동 피처 선택(top-k, shadow)은 오히려 나빴음 (A13)
  - KOEM은 학습에 합치지 않음 (analysis.py에서 검증용으로만 사용)
분류 기준 (학술 근거):
  - EU MSFD 해안 쓰레기 기준치 20개/100m (Van Loon et al. 2020, JRC121707 — EU 2015–16 기준선의 15분위)
  - EU 2015–16 기준선 수준 287개/100m (EEA coastline macro litter indicator)
  - 반려해변은 100m당 개수가 없어 '분위 대응': KOEM 조사 중 287개를 넘는 비율 = 반려해변 '많음' 비율.
    분위는 EU처럼 기준 기간(2022–2024)에서 정하고 전체 기간에 적용.
X는 BLOCKS에 적힌 컬럼만 쓴다. 키(loc_id, date), 좌표(lat, lng), 노력량(n_sessions, distance_km),
당일 카테고리 무게는 절대 X에 안 들어간다 (cv_xgb의 assert).
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBRegressor, XGBClassifier

FEATURES = 'outputs/features.csv'
TARGETS = {  # 반려해변 타깃 그룹 (합산 → 세션 수로 나눔 → log1p)
    'total': ['total_weight_g'],
    'plastic': ['plastic_weight_g'],
    'fishing': ['fishing_net_rope_g', 'styrofoam_buoy_g', 'packaging_band_g'],
    'household': ['plastic_container_g', 'cap_straw_spoon_g', 'vinyl_bag_g', 'product_packaging_g',
                  'cigarette_butt_g', 'styrofoam_container_g'],
    'natural': ['wood_g', 'rubber_g'],
    'items': ['item_count'],
}
W_GAP = ['rain_sum', 'rain_max_1h', 'rain_days_30', 'onshore_mean', 'onshore_max', 'wind_fetch', 'strong_h',
         'swh_mean', 'swh_max', 'storm_h', 'wave_cos']
W_SHORT = ['rain_sum', 'onshore_mean', 'wind_fetch', 'swh_max', 'storm_h']
BLOCKS = {
    'base': ['gap_days', 'is_first', 'month_sin', 'month_cos', 'year'],
    'port': ['port_near_km', 'port_n_10km', 'port_national_near_km', 'river_near_km', 'river_basin_w'],
    'aqua': ['buoy_farm_ha_5km', 'buoy_farm_ha_10km', 'buoy_farm_ha_20km', 'farm_ha_10km', 'buoy_farm_near_km'],
    'coast': ['fetch_mean_km', 'open_sectors'],
    'tourism': ['visitors_log', 'is_designated', 'summer_days'],
    'tide': ['tide_spring_day', 'tide_spring_win'],
    'koem': ['koem_prior_kg'],
    'w_gap': [f'{c}_gap' for c in W_GAP],   # 직전 수거 이후 구간
    'w_7d': [f'{c}_7d' for c in W_SHORT],   # 수거 직전 7일
    'w_30d': [f'{c}_30d' for c in W_SHORT],  # 수거 직전 30일
    'history': ['prev1_y', 'prev2_y', 'prev3_y', 'prev1_rate', 'beach_mean_y', 'beach_n_prev'],
    'eb': ['beach_eb_y'],  # 경험적 베이즈로 줄인 위치 평균
    'terrain': ['t_rock', 't_tidal_flat', 't_enclosure', 't_beach_len', 't_harbor', 't_armored', 't_urban',
                't_stream', 't_aqua_visible', 't_sub_gravel', 't_sub_rock', 't_sub_mud'],  # 위성 판독 지형 라벨
}
FORBIDDEN = {'loc_id', 'beach_id', 'station_name', 'date', 'lat', 'lng', 'n_sessions', 'distance_km',
             'participants', 'people', 'weight_kg', 'count_ea', 'volume_l'} | set(sum(TARGETS.values(), []))
PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=5.0, n_jobs=4)
# 최종 학습용 강한 규제 — 얕은 트리·큰 잎·분할 비용. 94개 해변에서 노이즈 분할을 막음 (A13: R² 0.145 → 0.197)
PARAMS_FINAL = dict(PARAMS, max_depth=2, min_child_weight=10, gamma=1.0, colsample_bytree=0.5, reg_lambda=20.0)
SEEDS = [0, 1, 2]  # 반복 GroupKFold — 행 수가 작아 fold 편차가 큼
USE = ['base', 'history', 'eb', 'tourism', 'aqua', 'tide', 'terrain']  # 학습에 쓰는 블록


def load_features(path=FEATURES, per_session=True):
    """per_session=True: 노력량 표준화 타깃 log1p(무게/세션 수). False: 원값 log1p(무게) (analysis.py 비교용)."""
    df = pd.read_csv(path, parse_dates=['date']).sort_values(['loc_id', 'date']).reset_index(drop=True)
    effort = df.n_sessions if per_session else 1
    return df, {k: np.log1p(df[v].sum(1) / effort) for k, v in TARGETS.items()}


def add_history(df, y):
    """위치별 과거 기록 (이번 행 제외 — shift). 타깃마다 다시 계산."""
    g = pd.DataFrame({'loc': df.loc_id, 'y': y.values, 'gap': df.gap_days}).groupby('loc')
    X = df.copy()
    for k in (1, 2, 3):
        X[f'prev{k}_y'] = g.y.shift(k).values
    X['prev1_rate'] = np.expm1(X.prev1_y) / g.gap.shift(1).values
    X['beach_mean_y'] = g.y.transform(lambda v: v.shift().expanding().mean()).values
    X['beach_n_prev'] = g.cumcount().values
    # 경험적 베이즈: 과거 기록이 적은 위치는 전체 평균 쪽으로 당김. k = 위치 내 분산 / 위치 간 분산
    # ponytail: k를 전체 y로 한 번 추정 (스칼라 하나라 누수 미미) — 엄밀히 하려면 fold 안에서 추정
    yy = pd.Series(y.values)
    within = yy.groupby(df.loc_id.values).var().mean()
    k = within / max(yy.groupby(df.loc_id.values).mean().var() - within / yy.groupby(df.loc_id.values).size().mean(), 1e-6)
    X['beach_eb_y'] = (X.beach_mean_y.fillna(0) * X.beach_n_prev + yy.mean() * k) / (X.beach_n_prev + k)
    return X


def cols_of(blocks):
    return [c for b in blocks for c in BLOCKS[b]]


def cv_xgb(X, y, groups, cols, seeds=SEEDS, return_pred=False, params=PARAMS):
    """반복 GroupKFold(위치 단위). seed마다 out-of-fold 예측 전체의 R²."""
    assert not set(cols) & FORBIDDEN, set(cols) & FORBIDDEN
    y, r2, preds = np.asarray(y), [], []
    for seed in seeds:
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            assert not set(groups[tr]) & set(groups[te])
            m = XGBRegressor(**params, random_state=seed).fit(X.iloc[tr][cols], y[tr])
            pred[te] = m.predict(X.iloc[te][cols])
        r2.append(r2_score(y, pred)); preds.append(pred)
    return (np.mean(r2), np.std(r2), np.mean(preds, 0)) if return_pred else (np.mean(r2), np.std(r2))


def time_split(X, y, cols, test_year=2026, params=PARAMS):
    tr, te = (X.year < test_year).values, (X.year >= test_year).values
    m = XGBRegressor(**params, random_state=0).fit(X[tr][cols], np.asarray(y)[tr])
    return r2_score(np.asarray(y)[te], m.predict(X[te][cols]))


def importance(X, y, cols, top=15, params=PARAMS):
    m = XGBRegressor(**params, random_state=0).fit(X[cols], y)
    imp = pd.Series(m.get_booster().get_score(importance_type='gain')).reindex(cols).fillna(0)
    return (imp / imp.sum()).sort_values(ascending=False).head(top)


# ── 분류: 쓰레기 많음(주의 요함) ─────────────────────────────────────────────
EU_TV, EU_BASELINE = 20, 287          # 100m당 개수 (위 docstring 참고)
BASELINE_YEARS = (2022, 2024)         # 반려해변 분위 기준 기간
PARAMS_CLS = dict(PARAMS_FINAL, min_child_weight=2)  # 분류 헤시안 p(1-p)≤0.25 → 잎 최소량도 그만큼 낮춤
# 분류는 과거 기록 없이도 AUC가 같음 (0.795 → 0.799) → history·eb를 빼서 처음 보는 해변에도 바로 적용되는 모델로.
# aqua는 세 타깃 모두 AUC를 낮춤 (−aqua: +0.03~0.04, A15)
USE_CLS = [b for b in USE if b not in ('aqua', 'history', 'eb')]
CLUSTER_KM = 10  # 공간 블록 검증: 이 거리 안의 해변은 같은 묶음으로 train/test에 함께 이동


def spatial_groups(df, km=CLUSTER_KM):
    """km 안에 있는 해변끼리 연결(single linkage)한 클러스터 id. 이웃 해변을 통한 정보 누설을 막는 검증용."""
    from data_load import haversine
    loc = df.groupby('loc_id')[['lat', 'lng']].first()
    D = haversine(loc.lat.values[:, None], loc.lng.values[:, None], loc.lat.values[None], loc.lng.values[None])
    lab = np.arange(len(loc))
    changed = True
    while changed:  # 연결된 해변끼리 가장 작은 id로 수렴
        new = np.array([lab[D[i] < km].min() for i in range(len(loc))])
        changed = (new != lab).any(); lab = new
    return df.loc_id.map(pd.Series(lab, index=loc.index)).values


def high_quantile(koem_path='outputs/koem_features.csv'):
    """반려해변 '많음' 분위 = 1 − (KOEM 조사 중 EU 기준선 287개/100m 초과 비율)."""
    c = pd.read_csv(koem_path).count_ea.dropna()
    return 1 - (c > EU_BASELINE).mean()


def label_high(df, y, q):
    """기준 기간(2022–2024) 분포의 q분위를 넘으면 1(많음)."""
    base = np.asarray(y)[df.year.between(*BASELINE_YEARS).values]
    thr = np.quantile(base, q)
    return (np.asarray(y) > thr).astype(int), thr


def cv_cls(X, yb, groups, cols, seeds=SEEDS, params=PARAMS_CLS):
    """반복 GroupKFold 이진 분류. 클래스 불균형은 fold마다 scale_pos_weight로 보정 → 0.5 컷이 균형점."""
    assert not set(cols) & FORBIDDEN, set(cols) & FORBIDDEN
    yb, res, probs = np.asarray(yb), [], []
    for seed in seeds:
        prob = np.zeros(len(yb))
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, yb, groups):
            spw = (yb[tr] == 0).sum() / max((yb[tr] == 1).sum(), 1)
            m = XGBClassifier(**params, scale_pos_weight=spw, random_state=seed).fit(X.iloc[tr][cols], yb[tr])
            prob[te] = m.predict_proba(X.iloc[te][cols])[:, 1]
        pred = (prob > 0.5).astype(int)
        res.append(dict(AUC=roc_auc_score(yb, prob), accuracy=np.mean(pred == yb),
                        balanced_acc=balanced_accuracy_score(yb, pred), precision=precision_score(yb, pred, zero_division=0),
                        recall=recall_score(yb, pred), F1=f1_score(yb, pred)))
        probs.append(prob)
    out = {k: round(np.mean([r[k] for r in res]), 3) for k in res[0]}
    out['AUC_sd'] = round(np.std([r['AUC'] for r in res]), 3)
    out['majority_acc'] = round(max(yb.mean(), 1 - yb.mean()), 3)  # 전부 다수 클래스로 찍을 때
    return out, np.mean(probs, 0)


def main():
    """반려해변: 세션당 무게가 '많음'(EU 기준선 대응 분위 초과)인지 분류.
    주 검증 = 새 지역(10km 묶음 단위 GroupKFold). 보조 = 새 해변(해변 단위). 둘 다 테스트 해변의 과거 기록은 안 씀.
    ablation도 주 검증(새 지역) 기준."""
    df, targets = load_features()
    groups, sgroups = spatial_groups(df), df.loc_id.values  # groups = 주 검증(새 지역), sgroups = 보조(새 해변)
    q = high_quantile()
    all_blocks = [b for b in USE_CLS if set(BLOCKS[b]) <= set(df.columns) | set(BLOCKS['history']) | {'beach_eb_y'}]
    print(f'rows {len(df)}, beaches {df.loc_id.nunique()} | 많음 = 기준기간 {BASELINE_YEARS} 상위 {1 - q:.1%} '
          f'(KOEM에서 EU 기준선 {EU_BASELINE}개/100m 초과 비율) | 블록 {all_blocks}')
    rows = []
    for name in ['total', 'plastic', 'fishing']:
        y = targets[name]
        yb, thr = label_high(df, y, q)
        X = add_history(df, y)  # history는 연속값(세션당 무게)의 과거 기록
        full = cols_of(all_blocks)
        m, _ = cv_cls(X, yb, groups, full)
        ms, _ = cv_cls(X, yb, sgroups, full)
        r = {'target': name, 'threshold_g_per_session': round(float(np.expm1(thr))), 'positive_rate': round(yb.mean(), 3), **m,
             'AUC_new_beach': ms['AUC'], 'balanced_acc_new_beach': ms['balanced_acc']}
        for b in all_blocks:  # ablation: 블록을 뺐을 때 AUC 변화 (음수 = 그 블록이 도움)
            r[f'-{b}'] = round(cv_cls(X, yb, groups, cols_of([x for x in all_blocks if x != b]))[0]['AUC'] - m['AUC'], 3)
        rows.append(r)
        print(f"[{name}] 많음 기준 {r['threshold_g_per_session']:,} g/세션 | 양성 {r['positive_rate']:.1%}\n"
              f"   새 지역({CLUSTER_KM}km 묶음, 주 지표) AUC {m['AUC']:.3f} ± {m['AUC_sd']:.3f} | 균형정확도 {m['balanced_acc']:.3f} | "
              f"정밀도 {m['precision']:.3f} 재현율 {m['recall']:.3f} | 정확도 {m['accuracy']:.3f} (다수 클래스 {m['majority_acc']:.3f})\n"
              f"   새 해변(해변 단위, 참고)   AUC {ms['AUC']:.3f} | 균형정확도 {ms['balanced_acc']:.3f}")
    res = pd.DataFrame(rows)
    print('\n=== 타깃 × 블록 ablation (새 지역 AUC 변화, 음수일수록 중요) ===')
    print(res[['target'] + [f'-{b}' for b in all_blocks]].to_string(index=False))
    res.to_csv('outputs/results_cls.csv', index=False)


def main_regression():
    df, targets = load_features()
    groups = spatial_groups(df)  # 주 검증 = 새 지역
    print(f'rows {len(df)}, beaches {df.loc_id.nunique()}, 2026 test rows {(df.year == 2026).sum()}')
    all_blocks = [b for b in USE if set(BLOCKS[b]) <= set(df.columns) | set(BLOCKS['history']) | {'beach_eb_y'}]
    print('타깃: 세션당 무게(노력량 표준화) | 블록:', all_blocks,
          '| 빠진 블록(피처 없음):', [b for b in USE if b not in all_blocks])
    rows = []
    for name, y in targets.items():
        X = add_history(df, y)
        full = cols_of(all_blocks)
        r = {'target': name}
        r['all'], r['all_sd'] = cv_xgb(X, y, groups, full, params=PARAMS_FINAL)
        r['time_2026'] = time_split(X, y, full, params=PARAMS_FINAL)
        for b in all_blocks:  # ablation: 블록 하나씩 뺐을 때 R² 변화 (음수 = 그 블록이 도움)
            r[f'-{b}'] = cv_xgb(X, y, groups, cols_of([x for x in all_blocks if x != b]), params=PARAMS_FINAL)[0] - r['all']
        rows.append(r)
        print(f"\n[{name}] GroupKFold R² {r['all']:.3f} ± {r['all_sd']:.3f} | 2026 split R² {r['time_2026']:.3f}")
        print(importance(X, y, full, params=PARAMS_FINAL).round(3).to_string())
    res = pd.DataFrame(rows).round(3)
    print('\n=== 타깃 × ablation (−블록 = 그 블록을 뺐을 때 R² 변화, 음수일수록 중요) ===')
    print(res.to_string(index=False))
    res.to_csv('outputs/results.csv', index=False)


if __name__ == '__main__':
    import sys
    main_regression() if sys.argv[1:] == ['reg'] else main()
