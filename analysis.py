"""왜 R²가 안 나오는가 — 진단 실험. 결과는 콘솔 표 + outputs/analysis/A*.csv

A1 분산분해: y 중 '위치에 고정된' 몫 vs '시간에 따라 변하는' 몫
A2 측정 신뢰도(노이즈 천장): 같은 해변·같은 날 세션을 반으로 나눠 서로 얼마나 일치하나
A3 노력량: 세션 수·이동거리만으로 총무게가 얼마나 설명되나
A4 타깃 정의: 총무게 vs 세션당 무게 vs km당 무게 vs 카테고리 (그리고 노력 표준화된 KOEM)
A5 블록 ablation: 어떤 요인이 실제로 도움이 되나 (반려해변 vs KOEM)
A6 합성 신호 복원: 진짜 R²가 x일 때 이 표본 크기로 얼마나 복원되나 (검정력)
A7 학습곡선: 위치 수를 늘리면 R²가 오르나
A8 rollout 변형: history·시간창(gap/7일/30일) 조합 + walk-forward(연도별 미래 예측)
A9 데이터 증강: KOEM을 학습에 합치거나(pooling) KOEM 모델 예측을 피처로(stacking)
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
import data_load as dl
from train import BLOCKS, PARAMS, SEEDS, add_history, cols_of, cv_xgb, load_features

OUT = Path('outputs/analysis')
RNG = np.random.default_rng(0)
STATIC_BLOCKS = ['port', 'aqua', 'coast', 'tourism', 'koem']
WEATHER = ['w_gap', 'w_7d', 'w_30d']
ALL = ['base', 'tide'] + STATIC_BLOCKS + WEATHER + ['history']
KOEM_ALL = [b for b in ALL if b not in ('tourism', 'koem')]  # KOEM: 관광 매칭 없음, koem_prior는 자기 평균이라 누수


def section(t, df=None, name=None):
    print(f'\n{"=" * 100}\n{t}\n{"=" * 100}')
    if df is not None:
        print(df.to_string(index=False))
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')


def load_all():
    C, tC = load_features('outputs/features.csv', per_session=False)
    K = pd.read_csv('outputs/koem_features.csv', parse_dates=['date']).sort_values(['loc_id', 'date']).reset_index(drop=True)
    per = lambda w, d: np.log1p(w / d.where(d > 0))
    T = {  # 반려해변 타깃 후보
        'total': tC['total'],
        'per_session': per(C.total_weight_g, C.n_sessions),
        'per_km': per(C.total_weight_g, C.distance_km),
        'plastic/session': per(C.plastic_weight_g, C.n_sessions),
        'fishing/session': per(np.expm1(tC['fishing']), C.n_sessions),
        'household/session': per(np.expm1(tC['household']), C.n_sessions),
        'natural/session': per(np.expm1(tC['natural']), C.n_sessions),
        'items/session': per(C.item_count, C.n_sessions),
    }
    TK = {'koem_weight': np.log1p(K.weight_kg), 'koem_count': np.log1p(K.count_ea)}
    return C, T, K, TK


def subset(df, y):
    ok = np.isfinite(np.asarray(y, float))
    return df[ok].reset_index(drop=True), pd.Series(np.asarray(y, float)[ok])


# ── A1 ─────────────────────────────────────────────────────────────────────
def var_decomp(y, g):
    y, g = pd.Series(np.asarray(y, float)), pd.Series(np.asarray(g))
    s, n = y.groupby(g).transform('sum'), y.groupby(g).transform('count')
    loo = np.where(n > 1, (s - y) / (n - 1).clip(lower=1), y.mean())  # 같은 위치의 '다른' 행 평균
    m = y.groupby(g).transform('mean')
    eta2 = 1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return eta2, r2_score(y, loo), n.groupby(g).first().median()


def a1(C, T, K, TK):
    rows = []
    for name, y, df in [*[(k, v, C) for k, v in T.items()], *[(k, v, K) for k, v in TK.items()]]:
        d, yy = subset(df, y)
        eta2, loo, nmed = var_decomp(yy, d.loc_id)
        rows.append(dict(target=name, rows=len(yy), locations=d.loc_id.nunique(), rows_per_loc_median=nmed,
                         eta2_in_sample=round(eta2, 3), loo_loc_mean_R2=round(loo, 3)))
    section('A1 분산분해 — loo_loc_mean_R2: 같은 위치의 다른 기록 평균만으로 맞힌 R² (= 위치 고유 성분의 크기)',
            pd.DataFrame(rows), 'A1_variance')


# ── A2 ─────────────────────────────────────────────────────────────────────
def a2():
    s = dl.load_sessions(); b = dl.load_beaches()
    s = dl.match_gps_only(s, b)
    m = s[(s.weight_method == 'measured') & s.beach_id.notna()]
    rows = []
    for col in ['total_weight_g', 'plastic_weight_g', 'fishing_net_rope_g', 'item_count']:
        days = [g[col].values for _, g in m.groupby(['beach_id', 'day']) if len(g) >= 4]
        rs = []
        for _ in range(300):
            a, c = [], []
            for w in days:
                p = RNG.permutation(len(w)); h = len(w) // 2
                a.append(np.log1p(w[p[:h]].mean())); c.append(np.log1p(w[p[h:]].mean()))
            rs.append(np.corrcoef(a, c)[0, 1])
        r = np.mean(rs)
        rows.append(dict(target=f'{col} per session', beach_days=len(days), split_half_r=round(r, 3),
                         reliability_full=round(2 * r / (1 + r), 3)))
    within_cv = m.groupby(['beach_id', 'day']).total_weight_g.agg(lambda v: v.std() / v.mean() if len(v) > 1 else np.nan)
    section('A2 측정 신뢰도 — 같은 해변·같은 날 세션을 무작위 반반으로 나눠 세션당 무게(log)의 상관. '
            'reliability_full ≈ 완벽한 모델이 낼 수 있는 R² 상한\n'
            f'   (같은 날 세션 간 무게 변동계수 중앙값 {within_cv.median():.2f})', pd.DataFrame(rows), 'A2_reliability')


# ── A3 ─────────────────────────────────────────────────────────────────────
def a3(C, T):
    from sklearn.linear_model import LinearRegression
    rows = []
    for name in ['total', 'per_session', 'per_km']:
        d, y = subset(C, T[name])
        for feats in [['n_sessions'], ['n_sessions', 'distance_km']]:
            dd = d.dropna(subset=feats); yy = y[d[feats].notna().all(1)]
            Z = np.log1p(dd[feats].values)
            rows.append(dict(target=name, effort=' + '.join(feats), rows=len(yy),
                             R2_effort_only=round(LinearRegression().fit(Z, yy).score(Z, yy), 3)))
    section('A3 노력량 — 세션 수·이동거리(log)만으로 설명되는 분산 (선형, in-sample)', pd.DataFrame(rows), 'A3_effort')


# ── A4 ─────────────────────────────────────────────────────────────────────
def a4(C, T, K, TK):
    rows = []
    for name, y, df, blocks in [*[(k, v, C, ALL) for k, v in T.items()], *[(k, v, K, KOEM_ALL) for k, v in TK.items()]]:
        d, yy = subset(df, y)
        X = add_history(d, yy)
        r, sd = cv_xgb(X, yy, d.loc_id.values, cols_of(blocks))
        rows.append(dict(target=name, rows=len(yy), R2=round(r, 3), sd=round(sd, 3)))
    d, yy = subset(C, T['total'])
    X = add_history(d, yy).assign(log_sessions=np.log(d.n_sessions))
    from train import FORBIDDEN
    r, sd = cv_xgb(X, yy, d.loc_id.values, cols_of(ALL) + ['log_sessions'])
    rows.append(dict(target='total (+세션 수 피처, 참고용)', rows=len(yy), R2=round(r, 3), sd=round(sd, 3)))
    section('A4 타깃 정의별 GroupKFold R² (전체 블록, 위치 단위 5-fold × 3 seed)', pd.DataFrame(rows), 'A4_targets')


# ── A5 ─────────────────────────────────────────────────────────────────────
def a5(C, T, K, TK):
    out = []
    for dname, df, y, blocks in [('caresea per_session', C, T['per_session'], ALL),
                                 ('KOEM weight', K, TK['koem_weight'], KOEM_ALL)]:
        d, yy = subset(df, y)
        X, g = add_history(d, yy), d.loc_id.values
        full = cv_xgb(X, yy, g, cols_of(blocks))[0]
        base = cv_xgb(X, yy, g, cols_of(['base']))[0]
        for b in blocks:
            out.append(dict(dataset=dname, block=b, full_R2=round(full, 3),
                            drop_delta=round(cv_xgb(X, yy, g, cols_of([x for x in blocks if x != b]))[0] - full, 3),
                            solo_with_base=round(cv_xgb(X, yy, g, cols_of(['base', b]))[0] if b != 'base' else base, 3)))
    section('A5 블록 ablation — drop_delta: 그 블록을 뺐을 때 R² 변화(음수=도움), solo_with_base: base + 그 블록만 썼을 때 R²',
            pd.DataFrame(out), 'A5_ablation')


# ── A6 ─────────────────────────────────────────────────────────────────────
def a6(C, T, K, TK):
    z = lambda v: (lambda a: (a - np.nanmean(a)) / np.nanstd(a))(np.nan_to_num(np.asarray(v, float), nan=np.nanmean(v)))
    out = []
    for dname, df, y, blocks in [('caresea', C, T['per_session'], ALL), ('KOEM', K, TK['koem_weight'], KOEM_ALL)]:
        d, yy = subset(df, y)
        X, g = add_history(d, yy), d.loc_id.values
        static = z(np.log1p(d.buoy_farm_ha_10km)) + z(d.fetch_mean_km)
        temporal = z(d.onshore_mean_gap) + z(d.rain_sum_gap)
        sig = z(static + temporal + 0.5 * np.maximum(z(temporal), 0) ** 2)  # 정적 + 시간 + 비선형
        for true_r2 in [0.1, 0.2, 0.3, 0.5]:
            rec = []
            for seed in range(3):
                e = np.random.default_rng(seed).standard_normal(len(sig))
                ys = pd.Series(np.sqrt(true_r2) * sig + np.sqrt(1 - true_r2) * e)
                rec.append(cv_xgb(X, ys, g, cols_of(blocks), seeds=[seed])[0])
            out.append(dict(dataset=dname, rows=len(yy), locations=len(set(g)), true_R2=true_r2,
                            recovered_R2=round(np.mean(rec), 3), recovery_ratio=round(np.mean(rec) / true_r2, 2)))
    section('A6 합성 신호 복원 — 실제 피처로 만든 가짜 타깃(진짜 R² 알고 있음)을 같은 파이프라인이 얼마나 복원하나',
            pd.DataFrame(out), 'A6_synthetic')


# ── A7 ─────────────────────────────────────────────────────────────────────
def a7(C, T, K, TK):
    out = []
    for dname, df, y, blocks in [('caresea per_session', C, T['per_session'], ALL),
                                 ('KOEM weight', K, TK['koem_weight'], KOEM_ALL)]:
        d, yy = subset(df, y)
        X, g, cols = add_history(d, yy), d.loc_id.values, cols_of(blocks)
        for frac in [0.25, 0.5, 0.75, 1.0]:
            r2s, ntr = [], []
            for seed in SEEDS:
                pred = np.full(len(yy), np.nan)
                for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, yy, g):
                    locs = np.unique(g[tr]); keep = np.random.default_rng(seed).choice(locs, max(3, int(len(locs) * frac)), replace=False)
                    tr = tr[np.isin(g[tr], keep)]; ntr.append(len(keep))
                    pred[te] = XGBRegressor(**PARAMS, random_state=seed).fit(X.iloc[tr][cols], yy.iloc[tr]).predict(X.iloc[te][cols])
                r2s.append(r2_score(yy, pred))
            out.append(dict(dataset=dname, train_locations=int(np.mean(ntr)), R2=round(np.mean(r2s), 3), sd=round(np.std(r2s), 3)))
    section('A7 학습곡선 — 학습에 쓰는 위치 수를 줄였을 때 R²', pd.DataFrame(out), 'A7_learning_curve')


# ── A8 ─────────────────────────────────────────────────────────────────────
def a8(C, T, K, TK):
    sets = {
        'static only': ['base'] + STATIC_BLOCKS,
        '+history': ['base'] + STATIC_BLOCKS + ['history'],
        '+w_gap': ['base'] + STATIC_BLOCKS + ['w_gap'],
        '+w_7d': ['base'] + STATIC_BLOCKS + ['w_7d'],
        '+w_30d': ['base'] + STATIC_BLOCKS + ['w_30d'],
        '+all weather': ['base'] + STATIC_BLOCKS + WEATHER,
        '+history+all weather+tide': ALL,
    }
    out, wf = [], []
    for dname, df, y, drop in [('caresea per_session', C, T['per_session'], []),
                               ('KOEM weight', K, TK['koem_weight'], ['tourism', 'koem'])]:
        d, yy = subset(df, y)
        X, g = add_history(d, yy), d.loc_id.values
        for sname, blocks in sets.items():
            blocks = [b for b in blocks if b not in drop]
            r, sd = cv_xgb(X, yy, g, cols_of(blocks))
            out.append(dict(dataset=dname, features=sname, R2=round(r, 3), sd=round(sd, 3)))
        cols = cols_of([b for b in ALL if b not in drop])
        years = sorted(d.year.unique())[-3:]
        for yr in years:  # walk-forward: 과거 연도로만 학습해서 그 해를 예측
            tr, te = (d.year < yr).values, (d.year == yr).values
            m = XGBRegressor(**PARAMS, random_state=0).fit(X[tr][cols], yy[tr])
            wf.append(dict(dataset=dname, test_year=yr, train_rows=tr.sum(), test_rows=te.sum(),
                           R2=round(r2_score(yy[te], m.predict(X[te][cols])), 3),
                           R2_history_only=round(r2_score(yy[te], X[te].beach_mean_y.fillna(yy[tr].mean())), 3)))
    section('A8 rollout 변형 — 정적 피처에 history / 시간창(gap·7일·30일)을 더했을 때 GroupKFold R²', pd.DataFrame(out), 'A8_rollout')
    section('A8b walk-forward — 과거 연도로 학습 → 다음 해 예측 (R2_history_only: 해변 과거 평균만으로 예측)',
            pd.DataFrame(wf), 'A8b_walkforward')


# ── A9 ─────────────────────────────────────────────────────────────────────
def a9(C, T, K, TK):
    common = [b for b in ALL if b not in ('tourism', 'koem')]
    cols = cols_of(common)
    dC, yC = subset(C, T['per_session']); XC = add_history(dC, yC)
    dK, yK = subset(K, TK['koem_weight']); XK = add_history(dK, yK)
    zC, zK = (yC - yC.mean()) / yC.std(), (yK - yK.mean()) / yK.std()
    g = dC.loc_id.values
    # stacking 피처: KOEM으로만 학습한 정적+날씨 모델의 예측 (history 제외 — 위치가 달라서)
    sc = cols_of([b for b in common if b != 'history'])
    XC['koem_model'] = XGBRegressor(**PARAMS, random_state=0).fit(XK[sc], zK).predict(XC[sc])
    out = []
    for name, extra_cols, pool in [('caresea only', [], False), ('+ koem_model 피처 (stacking)', ['koem_model'], False),
                                   ('+ KOEM 행 합쳐 학습 (pooling)', ['is_koem'], True)]:
        r2s = []
        for seed in SEEDS:
            pred = np.zeros(len(zC))
            for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(XC, zC, g):
                Xtr, ytr = XC.iloc[tr].assign(is_koem=0), zC.iloc[tr]
                if pool:
                    Xtr = pd.concat([Xtr, XK.assign(is_koem=1)], ignore_index=True); ytr = pd.concat([ytr, zK], ignore_index=True)
                m = XGBRegressor(**PARAMS, random_state=seed).fit(Xtr[cols + extra_cols], ytr)
                pred[te] = m.predict(XC.iloc[te].assign(is_koem=0)[cols + extra_cols])
            r2s.append(r2_score(zC, pred))
        out.append(dict(setup=name, R2=round(np.mean(r2s), 3), sd=round(np.std(r2s), 3)))
    section('A9 데이터 증강 — KOEM(노력 표준화 1,700행)을 반려해변 학습에 활용 (타깃: 세션당 무게, 공통 피처)',
            pd.DataFrame(out), 'A9_augmentation')


# ── A10 ────────────────────────────────────────────────────────────────────
def pooled_cv(XC, yC, XK, yK, cols):
    """반려해변 fold로 평가하되 학습에는 KOEM 전체를 더한다. 타깃은 데이터셋별 z-score."""
    zC, zK, g, r2s = (yC - yC.mean()) / yC.std(), (yK - yK.mean()) / yK.std(), XC.loc_id.values, []
    for seed in SEEDS:
        pred = np.zeros(len(zC))
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(XC, zC, g):
            Xtr = pd.concat([XC.iloc[tr].assign(is_koem=0), XK.assign(is_koem=1)], ignore_index=True)
            ytr = pd.concat([zC.iloc[tr], zK], ignore_index=True)
            pred[te] = XGBRegressor(**PARAMS, random_state=seed).fit(Xtr[cols + ['is_koem']], ytr).predict(
                XC.iloc[te].assign(is_koem=0)[cols + ['is_koem']])
        r2s.append(r2_score(zC, pred))
    return np.mean(r2s), np.std(r2s)


def a10(C, T, K, TK):
    lean = ['base', 'history', 'eb', 'tourism', 'aqua', 'tide']  # A5에서 도움이 된 블록만
    recipes = {'full (현재)': ALL, 'full + eb': ALL + ['eb'], 'lean': lean,
               'history+eb+tourism': ['history', 'eb', 'tourism'], 'eb only': ['eb']}
    out = []
    for tname in ['per_session', 'plastic/session', 'fishing/session']:
        d, yy = subset(C, T[tname]); X, g = add_history(d, yy), d.loc_id.values
        for rname, blocks in recipes.items():
            r, sd = cv_xgb(X, yy, g, cols_of(blocks))
            out.append(dict(dataset='caresea', target=tname, recipe=rname, R2=round(r, 3), sd=round(sd, 3)))
    dK, yK = subset(K, TK['koem_weight']); XK = add_history(dK, yK)
    for rname, blocks in {'full (현재)': KOEM_ALL, 'full + eb': KOEM_ALL + ['eb'],
                          'lean': ['base', 'history', 'eb', 'aqua', 'coast', 'w_30d'], 'eb only': ['eb']}.items():
        r, sd = cv_xgb(XK, yK, dK.loc_id.values, cols_of(blocks))
        out.append(dict(dataset='KOEM', target='koem_weight', recipe=rname, R2=round(r, 3), sd=round(sd, 3)))
    d, yy = subset(C, T['per_session']); X = add_history(d, yy)
    common = cols_of(['base', 'history', 'eb', 'aqua', 'tide'])
    zC = (yy - yy.mean()) / yy.std()
    r, sd = cv_xgb(X, zC, d.loc_id.values, common)
    out.append(dict(dataset='caresea', target='per_session (z)', recipe='lean 공통 피처, caresea만', R2=round(r, 3), sd=round(sd, 3)))
    r, sd = pooled_cv(X, yy, XK, yK, common)
    out.append(dict(dataset='caresea', target='per_session (z)', recipe='lean 공통 피처 + KOEM pooling', R2=round(r, 3), sd=round(sd, 3)))
    section('A10 개선 레시피 검증 — 피처 축소(lean), 경험적 베이즈 위치평균(eb), KOEM pooling', pd.DataFrame(out), 'A10_recipes')


# ── A11 ────────────────────────────────────────────────────────────────────
def loc_mean_r2(g, cols, n_perm=0):
    """위치 평균 y를 Ridge로 LOO 예측한 R². n_perm>0이면 라벨을 위치 간에 섞은 귀무분포의 p값도."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    def fit(X):
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-1, 3, 15)), X, g.y.values, cv=LeaveOneOut())
        return r2_score(g.y, p)
    X = g[cols].fillna(g[cols].median()).values.astype(float)
    r = fit(X)
    if not n_perm:
        return r, np.nan
    null = [fit(X[np.random.default_rng(i).permutation(len(X))]) for i in range(n_perm)]
    return r, (np.sum(np.array(null) >= r) + 1) / (n_perm + 1)


def a11(C, T, K, TK):
    from scipy.stats import spearmanr
    terr, static = BLOCKS['terrain'], BLOCKS['port'] + BLOCKS['aqua'] + BLOCKS['coast']
    out, cor = [], []
    for dname, df, y, min_n in [('KOEM 정점', K, TK['koem_weight'], 1), ('caresea 해변(기록≥3)', C, T['per_session'], 3)]:
        d = df.assign(y=np.asarray(y))
        g = d.groupby('loc_id').agg(y=('y', 'mean'), n=('y', 'size'), **{c: (c, 'first') for c in terr + static})
        g = g[g.n >= min_n]
        for name, cols in [('정적 피처(어항·양식장·해안)', static), ('지형 라벨', terr), ('둘 다', static + terr)]:
            r, p = loc_mean_r2(g, cols, n_perm=200 if name == '지형 라벨' else 0)
            out.append(dict(test='위치평균 Ridge LOO', dataset=dname, locations=len(g), features=name, R2=round(r, 3),
                            perm_p=round(p, 3) if p == p else ''))
        for c in terr:
            rho, pv = spearmanr(g[c], g.y)
            cor.append(dict(dataset=dname, label=c, rho=round(rho, 2), p=round(pv, 3), positives=int((g[c] > 0).sum())))
    for dname, df, y, base in [('caresea per_session', C, T['per_session'], ['base', 'history', 'eb', 'tourism', 'aqua', 'tide']),
                               ('caresea plastic/session', C, T['plastic/session'], ['base', 'history', 'eb', 'tourism', 'aqua', 'tide']),
                               ('caresea fishing/session', C, T['fishing/session'], ['base', 'history', 'eb', 'tourism', 'aqua', 'tide']),
                               ('KOEM weight', K, TK['koem_weight'], ['base', 'history', 'eb', 'aqua', 'coast', 'w_30d'])]:
        dd, yy = subset(df, y); X, g = add_history(dd, yy), dd.loc_id.values
        for name, blocks in [('lean', base), ('lean + terrain', base + ['terrain']),
                             ('terrain 단독(+base)', ['base', 'terrain']), ('history 없이 lean + terrain',
                                                                         [b for b in base if b not in ('history', 'eb')] + ['terrain'])]:
            r, sd = cv_xgb(X, yy, g, cols_of(blocks))
            out.append(dict(test='GroupKFold XGB', dataset=dname, locations=len(set(g)), features=name, R2=round(r, 3), perm_p=f'sd {sd:.3f}'))
    section('A11 지형 라벨 효과 — 위치평균 예측(라벨 섞기 순열검정 200회) + 전체 모델 CV', pd.DataFrame(out), 'A11_terrain')
    section('A11b 라벨별 Spearman ρ (위치 평균 y와)', pd.DataFrame(cor).sort_values(['dataset', 'p']), 'A11b_terrain_corr')


# ── A12 ────────────────────────────────────────────────────────────────────
def metrics_cv(X, y, g, cols, pool=None):
    """반복 GroupKFold로 여러 지표. pool=(XK, yK)면 KOEM 행을 반려해변 척도로 맞춰 학습에만 추가."""
    from scipy.stats import spearmanr
    y = np.asarray(y, float); med = np.median(y); res = []
    for seed in SEEDS:
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            Xtr, ytr = X.iloc[tr][cols], y[tr]
            if pool is not None:
                XK, yK = pool
                yk = (yK - yK.mean()) / yK.std() * y[tr].std() + y[tr].mean()
                Xtr, ytr = pd.concat([Xtr.assign(is_koem=0), XK[cols].assign(is_koem=1)]), np.r_[ytr, yk]
            m = XGBRegressor(**PARAMS, random_state=seed).fit(Xtr, ytr)
            pred[te] = m.predict(X.iloc[te][cols].assign(is_koem=0) if pool is not None else X.iloc[te][cols])
        err = np.abs(pred - y)
        res.append(dict(R2=r2_score(y, pred), spearman=spearmanr(pred, y)[0], within_2x=np.mean(err < np.log(2)),
                        hilo_acc=np.mean((pred > med) == (y > med)), err_factor=np.exp(np.median(err))))
    return {k: round(np.mean([r[k] for r in res]), 3) for k in res[0]}


def a12(C, T, K, TK):
    lean = ['base', 'history', 'tourism', 'aqua', 'tide']
    final = lean + ['eb', 'terrain']
    d, y_ps = subset(C, T['per_session']); y_raw = T['total']
    X, g = add_history(d, y_ps), d.loc_id.values
    Xr = add_history(d, y_raw)  # 원값 타깃용 history
    rows = []
    ref = dict(R2=0.0, spearman=np.nan, within_2x=round(np.mean(np.abs(y_ps - y_ps.mean()) < np.log(2)), 3),
               hilo_acc=0.5, err_factor=round(np.exp(np.median(np.abs(y_ps - y_ps.mean()))), 3))
    rows.append(dict(group='참고', step='전체 평균으로만 예측', **ref))
    steps = [('① 출발점: 원값 총무게 + 초기 피처', Xr, y_raw, ['base', 'port', 'koem', 'w_gap', 'history']),
             ('② + 노력량 표준화 (세션당)', X, y_ps, ['base', 'port', 'koem', 'w_gap', 'history']),
             ('③ + 신규 피처 전부 (양식장·해안·관광·조석·7/30일창)', X, y_ps, ALL),
             ('④ + 피처 축소 (lean)', X, y_ps, lean),
             ('⑤ + 경험적 베이즈 위치평균', X, y_ps, lean + ['eb']),
             ('⑥ + 지형 라벨 = 최종', X, y_ps, final)]
    for name, XX, yy, blocks in steps:
        rows.append(dict(group='누적', step=name, **metrics_cv(XX, yy, g, cols_of(blocks))))
    rows.append(dict(group='최종에서 하나 빼기', step='− 노력량 표준화 (원값 타깃)', **metrics_cv(Xr, y_raw, g, cols_of(final))))
    for b in final:
        rows.append(dict(group='최종에서 하나 빼기', step=f'− {b}', **metrics_cv(X, y_ps, g, cols_of([x for x in final if x != b]))))
    dK, yK = subset(K, TK['koem_weight']); XK = add_history(dK, yK)
    rows.append(dict(group='대안 기법', step='최종 + KOEM 행 합쳐 학습', **metrics_cv(X, y_ps, g, cols_of(final), pool=(XK, yK))))
    dk, ykm = subset(C, T['per_km'])
    rows.append(dict(group='대안 기법', step='최종, 타깃만 km당 무게', **metrics_cv(add_history(dk, ykm), ykm, dk.loc_id.values, cols_of(final))))
    section('A12 세션당 총무게 — 기법 누적·제거 ablation (해변 단위 5-fold × 3 seed 평균)', pd.DataFrame(rows), 'A12_ablation')

    fin = []
    for tname in ['per_session', 'plastic/session', 'fishing/session', 'household/session', 'natural/session', 'items/session']:
        dd, yy = subset(C, T[tname]); XX = add_history(dd, yy)
        m = metrics_cv(XX, yy, dd.loc_id.values, cols_of(final))
        tr, te = (dd.year < 2026).values, (dd.year == 2026).values
        p = XGBRegressor(**PARAMS, random_state=0).fit(XX[tr][cols_of(final)], yy[tr]).predict(XX[te][cols_of(final)])
        fin.append(dict(target=tname, **m, R2_2026=round(r2_score(yy[te], p), 3)))
    section('A12b 최종 모델 성능 (타깃별) — R2_2026: 2025년까지로 학습해 2026년 예측', pd.DataFrame(fin), 'A12_final')


# ── A13 ────────────────────────────────────────────────────────────────────
def nested_select_cv(X, y, g, cols, method=None, k=None, params=PARAMS):
    """fold 안의 학습 데이터만으로 피처를 고른 뒤 평가 (선택 편향 없음).
    method: None(전체) | 'topk'(gain 상위 k) | 'shadow'(섞은 복제 피처보다 중요한 것만 — Boruta 간이판)."""
    y = np.asarray(y, float); r2s, nsel, picked = [], [], pd.Series(0, index=cols)
    for seed in SEEDS:
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            Xtr, use = X.iloc[tr][cols], cols
            if method:
                rng = np.random.default_rng(seed)
                if method == 'shadow':
                    sh = Xtr.apply(lambda c: rng.permutation(c.values)).add_prefix('sh_')
                    Xtr_ = pd.concat([Xtr, sh], axis=1)
                else:
                    Xtr_ = Xtr
                imp = pd.Series(XGBRegressor(**params, random_state=seed).fit(Xtr_, y[tr])
                                .get_booster().get_score(importance_type='gain')).reindex(Xtr_.columns).fillna(0)
                if method == 'shadow':
                    use = [c for c in cols if imp[c] > imp.filter(like='sh_').max()] or list(imp[cols].nlargest(3).index)
                else:
                    use = list(imp[cols].nlargest(k).index)
            picked[use] += 1; nsel.append(len(use))
            pred[te] = XGBRegressor(**params, random_state=seed).fit(X.iloc[tr][use], y[tr]).predict(X.iloc[te][use])
        r2s.append(r2_score(y, pred))
    return np.mean(r2s), np.std(r2s), np.mean(nsel), picked / (len(SEEDS) * 5)


def a13(C, T, K, TK):
    cand = cols_of(ALL + ['eb', 'terrain'])
    d, y = subset(C, T['per_session']); X, g = add_history(d, y), d.loc_id.values
    strong = dict(PARAMS, max_depth=2, min_child_weight=10, gamma=1.0, colsample_bytree=0.5, reg_lambda=20.0)
    rows, freq = [], None
    for name, kw in [('전체 피처, 선택 없음', {}),
                     ('전체 피처 + 강한 규제', dict(params=strong)),
                     ('fold 내 자동 선택: gain 상위 5', dict(method='topk', k=5)),
                     ('fold 내 자동 선택: gain 상위 10', dict(method='topk', k=10)),
                     ('fold 내 자동 선택: gain 상위 20', dict(method='topk', k=20)),
                     ('fold 내 자동 선택: shadow(Boruta 간이)', dict(method='shadow')),
                     ('fold 내 shadow 선택 + 강한 규제', dict(method='shadow', params=strong))]:
        r, sd, n, f = nested_select_cv(X, y, g, cand, **kw)
        rows.append(dict(setup=name, candidates=len(cand), avg_selected=round(n, 1), R2=round(r, 3), sd=round(sd, 3)))
        if kw.get('method') == 'shadow' and 'params' not in kw:
            freq = f
    r, sd = cv_xgb(X, y, g, cols_of(['base', 'history', 'tourism', 'aqua', 'tide', 'eb', 'terrain']))
    rows.append(dict(setup='참고: 사람이 고른 최종 블록 (같은 데이터로 골라 낙관적)', candidates=len(cand), avg_selected=len(
        cols_of(['base', 'history', 'tourism', 'aqua', 'tide', 'eb', 'terrain'])), R2=round(r, 3), sd=round(sd, 3)))
    section('A13 자동 피처 선택 — 피처 선택을 각 fold 학습 데이터 안에서만 수행 (세션당 총무게)', pd.DataFrame(rows), 'A13_selection')
    top = freq.sort_values(ascending=False).head(15).round(2).rename('selected_frac').reset_index().rename(columns={'index': 'feature'})
    section('A13b shadow 선택에서 자주 뽑힌 피처 (15개 fold 중 선택 비율)', top, 'A13_selected_features')


# ── A14 ────────────────────────────────────────────────────────────────────
GRID_BLOCKS = ['base', 'history', 'tourism', 'aqua', 'tide', 'eb', 'terrain']
MONO = {'prev1_y': 1, 'prev2_y': 1, 'prev3_y': 1, 'beach_mean_y': 1, 'beach_eb_y': 1, 't_rock': 1, 't_urban': -1}
GRID = dict(weight=['none', 'sqrt', 'linear'], objective=['squared', 'huber'], monotone=[False, True],
            max_depth=[2, 3], min_child_weight=[5, 10, 20], reg_lambda=[5, 20])


def grid_configs():
    import itertools
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def make_model(cfg, cols, seed, y_train):
    p = dict(PARAMS, max_depth=cfg['max_depth'], min_child_weight=cfg['min_child_weight'], gamma=1.0,
             colsample_bytree=0.5, reg_lambda=cfg['reg_lambda'], n_jobs=1)
    if cfg['objective'] == 'huber':
        # 큰 오차(큰 그물 등)의 영향 완화. 시작값을 직접 지정 — 자동 추정이 252로 튀어 학습이 안 됨 (y≈9)
        p.update(objective='reg:pseudohubererror', huber_slope=1.0, base_score=float(np.median(y_train)))
    if cfg['monotone']:
        p['monotone_constraints'] = {c: v for c, v in MONO.items() if c in cols}
    return XGBRegressor(**p, random_state=seed)


def weights(cfg, n_sessions):
    w = {'none': np.ones(len(n_sessions)), 'sqrt': np.sqrt(n_sessions), 'linear': n_sessions.astype(float)}[cfg['weight']]
    return w / w.mean()  # 평균 1로 맞춤 (min_child_weight 의미 유지)


def cv_cfg(X, y, g, ns, cols, cfg, seeds, n_splits=5):
    r2 = []
    for seed in seeds:
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, g):
            m = make_model(cfg, cols, seed, y[tr]).fit(X.iloc[tr][cols], y[tr], sample_weight=weights(cfg, ns[tr]))
            pred[te] = m.predict(X.iloc[te][cols])
        r2.append(r2_score(y, pred))
    return np.mean(r2), np.std(r2)


def a14(C, T, K, TK):
    from joblib import Parallel, delayed
    d, yy = subset(C, T['per_session']); X, g = add_history(d, yy), d.loc_id.values
    y, ns, cols, cfgs = yy.values, d.n_sessions.values, cols_of(GRID_BLOCKS), grid_configs()

    # 1) 일반 그리드 서치 (5-fold × 3 seed) — 순위용. 1등 점수는 낙관적
    res = Parallel(n_jobs=-1)(delayed(cv_cfg)(X, y, g, ns, cols, c, SEEDS) for c in cfgs)
    board = pd.DataFrame([dict(**c, R2=round(r, 3), sd=round(s, 3)) for c, (r, s) in zip(cfgs, res)]).sort_values('R2', ascending=False)
    section('A14 그리드 서치 상위 15 (세션당 총무게, 해변 단위 5-fold × 3 seed)', board.head(15), 'A14_grid_top')
    marg = pd.concat([board.groupby(k).R2.agg(['mean', 'max']).round(3).assign(param=k).reset_index().rename(columns={k: 'value'})
                      for k in GRID])[['param', 'value', 'mean', 'max']]
    section('A14b 파라미터별 효과 — 해당 값을 쓴 모든 조합의 평균/최대 R²', marg, 'A14_marginal')

    # 2) 중첩 CV: 바깥 fold마다 안쪽(4-fold)에서 그리드 서치 → 고른 설정으로 바깥 test 평가 = 정직한 성능
    outer = []
    for seed in [0, 1]:
        pred = np.zeros(len(y)); chosen = []
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            inner = Parallel(n_jobs=-1)(delayed(cv_cfg)(X.iloc[tr], y[tr], g[tr], ns[tr], cols, c, [seed], 4) for c in cfgs)
            best = cfgs[int(np.argmax([r for r, _ in inner]))]; chosen.append(best)
            m = make_model(best, cols, seed, y[tr]).fit(X.iloc[tr][cols], y[tr], sample_weight=weights(best, ns[tr]))
            pred[te] = m.predict(X.iloc[te][cols])
        outer.append(dict(outer_seed=seed, nested_R2=round(r2_score(y, pred), 3),
                          chosen=' | '.join(f"{c['weight']},{c['objective']},mono={int(c['monotone'])},d{c['max_depth']},"
                                            f"mcw{c['min_child_weight']},l{c['reg_lambda']}" for c in chosen)))
    section('A14c 중첩 CV — "그리드 서치 후 선택" 절차의 정직한 성능 (바깥 5-fold, fold마다 고른 설정)', pd.DataFrame(outer), 'A14_nested')

    # 3) 1등 설정을 다른 타깃·2026 분할에도 적용
    best = board.iloc[0][list(GRID)].to_dict()
    best = {k: (bool(v) if k == 'monotone' else int(v) if k in ('max_depth', 'min_child_weight', 'reg_lambda') else v) for k, v in best.items()}
    cur = dict(weight='none', objective='squared', monotone=False, max_depth=2, min_child_weight=10, reg_lambda=20)
    rows = []
    for tname in ['per_session', 'plastic/session', 'fishing/session', 'household/session', 'items/session']:
        dd, yt = subset(C, T[tname]); XX = add_history(dd, yt)
        for name, cfg in [('현재 PARAMS_FINAL', cur), ('그리드 1등', best)]:
            r, s = cv_cfg(XX, yt.values, dd.loc_id.values, dd.n_sessions.values, cols, cfg, SEEDS)
            tr, te = (dd.year < 2026).values, (dd.year == 2026).values
            m = make_model(cfg, cols, 0, yt.values[tr]).fit(XX[tr][cols], yt.values[tr], sample_weight=weights(cfg, dd.n_sessions.values[tr]))
            rows.append(dict(target=tname, setup=name, R2=round(r, 3), sd=round(s, 3),
                             R2_2026=round(r2_score(yt.values[te], m.predict(XX[te][cols])), 3)))
    section(f'A14d 그리드 1등 설정을 타깃별로 적용 — 1등: {best}', pd.DataFrame(rows), 'A14_best_by_target')


# ── A15 ────────────────────────────────────────────────────────────────────
def a15(C, T, K, TK):
    """분류 과제: '많음(주의 요함)' 판정. 수거 단위 + 위치×연도 단위, 반려해변과 KOEM."""
    from train import cv_cls, label_high, high_quantile, EU_TV, EU_BASELINE, PARAMS_CLS, USE
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from xgboost import XGBClassifier
    rows = []
    d, y = subset(C, T['per_session']); X, g = add_history(d, y), d.loc_id.values
    q = high_quantile()
    use_noaqua = [b for b in USE if b != 'aqua']
    for lname, qq in [(f'많음 = EU 기준선 대응 상위 {1 - q:.0%}', q), ('많음/적음 = 기준기간 중앙값', 0.5)]:
        yb, thr = label_high(d, y, qq)
        for bname, blocks in [('최종 블록', USE), ('최종 − aqua', use_noaqua)]:
            m, prob = cv_cls(X, yb, g, cols_of(blocks))
            rows.append(dict(dataset='반려해변', unit='해변×수거일', label=lname, features=bname, n=len(yb), pos_rate=round(yb.mean(), 3),
                             **{k: m[k] for k in ['AUC', 'AUC_sd', 'balanced_acc', 'accuracy', 'majority_acc', 'precision', 'recall']}))
            # 해변×연도: 그해 평균 확률 vs 그해 평균 세션당 무게가 기준 초과인지
            by = pd.DataFrame({'loc': g, 'year': d.year.values, 'y': y.values, 'p': prob}).groupby(['loc', 'year']).agg(
                y=('y', 'mean'), p=('p', 'mean'), n=('y', 'size'))
            by = by[by.n >= 3]; lab = (by.y > thr).astype(int)
            if lab.nunique() == 2:
                pred = (by.p > 0.5).astype(int)
                rows.append(dict(dataset='반려해변', unit='해변×연도 (기록≥3)', label=lname, features=bname, n=len(by), pos_rate=round(lab.mean(), 3),
                                 AUC=round(roc_auc_score(lab, by.p), 3), balanced_acc=round(balanced_accuracy_score(lab, pred), 3),
                                 accuracy=round(np.mean(pred == lab), 3), majority_acc=round(max(lab.mean(), 1 - lab.mean()), 3)))
    dK, yK = subset(K, TK['koem_weight']); XK = add_history(dK, yK); gK = dK.loc_id.values
    colsK = cols_of([b for b in KOEM_ALL if b != 'aqua'] + ['eb', 'terrain'])
    for lname, thr in [(f'EU 기준치 초과 (>{EU_TV}개/100m)', EU_TV), (f'EU 기준선 초과 (>{EU_BASELINE}개/100m)', EU_BASELINE)]:
        yb = (dK.count_ea.values > thr).astype(int)
        m, _ = cv_cls(XK, yb, gK, colsK)
        rows.append(dict(dataset='KOEM', unit='정점×조사', label=lname, features='공통 블록', n=len(yb), pos_rate=round(yb.mean(), 3),
                         **{k: m[k] for k in ['AUC', 'AUC_sd', 'balanced_acc', 'accuracy', 'majority_acc', 'precision', 'recall']}))
        # 정점×연도 walk-forward: 과거 연도로만 학습 → 그해 조사들의 평균 확률로 '연평균 개수 > 기준' 판정
        P, L = [], []
        for yr in [2021, 2022, 2023]:
            tr, te = (dK.year < yr).values, (dK.year == yr).values
            spw = (yb[tr] == 0).sum() / max((yb[tr] == 1).sum(), 1)
            mm = XGBClassifier(**PARAMS_CLS, scale_pos_weight=spw, random_state=0).fit(XK[tr][colsK], yb[tr])
            t = pd.DataFrame({'loc': gK[te], 'p': mm.predict_proba(XK[te][colsK])[:, 1], 'c': dK.count_ea.values[te]}).groupby('loc').mean()
            P += list(t.p); L += list((t.c > thr).astype(int))
        P, L = np.array(P), np.array(L)
        rows.append(dict(dataset='KOEM', unit='정점×연도 (다음 해 예측)', label=lname, features='공통 블록', n=len(L), pos_rate=round(L.mean(), 3),
                         AUC=round(roc_auc_score(L, P), 3), balanced_acc=round(balanced_accuracy_score(L, P > 0.5), 3),
                         accuracy=round(np.mean((P > 0.5) == L), 3), majority_acc=round(max(L.mean(), 1 - L.mean()), 3)))
    section('A15 분류: 쓰레기 많음(주의 요함) 판정 — AUC·균형정확도 중심 (정확도는 다수 클래스 정확도와 비교)',
            pd.DataFrame(rows), 'A15_classification')


# ── A16 ────────────────────────────────────────────────────────────────────
def a16(C, T, K, TK):
    """최종 분류기(세션당 총무게 '많음')의 피처 중요도 — 새 지역 검증 기준.
    permutation: 테스트 묶음에서 그 피처만 섞었을 때 AUC 하락 (가장 직접적). SHAP: 기여 크기와 방향. gain: 참고."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    from train import load_features, label_high, high_quantile, spatial_groups, USE_CLS, PARAMS_CLS
    df, Tr = load_features(); y = Tr['total']; yb, _ = label_high(df, y, high_quantile())
    X, g, cols = add_history(df, y), spatial_groups(df), cols_of(USE_CLS)
    blocks = {c: b for b in USE_CLS for c in BLOCKS[b]}
    base_p, perm_p = [], {c: [] for c in cols}
    for seed in SEEDS:
        prob = np.zeros(len(yb)); pp = {c: np.zeros(len(yb)) for c in cols}
        rng = np.random.default_rng(seed)
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, yb, g):
            spw = (yb[tr] == 0).sum() / max((yb[tr] == 1).sum(), 1)
            m = XGBClassifier(**PARAMS_CLS, scale_pos_weight=spw, random_state=seed).fit(X.iloc[tr][cols], yb[tr])
            Xte = X.iloc[te][cols]; prob[te] = m.predict_proba(Xte)[:, 1]
            for c in cols:
                acc = np.zeros(len(te))
                for _ in range(5):
                    Xp = Xte.copy(); Xp[c] = rng.permutation(Xp[c].values); acc += m.predict_proba(Xp)[:, 1] / 5
                pp[c][te] = acc
        base_p.append(roc_auc_score(yb, prob))
        for c in cols:
            perm_p[c].append(roc_auc_score(yb, pp[c]))
    base = np.mean(base_p)
    m = XGBClassifier(**PARAMS_CLS, scale_pos_weight=(yb == 0).sum() / (yb == 1).sum(), random_state=0).fit(X[cols], yb)
    shap = m.get_booster().predict(__import__('xgboost').DMatrix(X[cols]), pred_contribs=True)[:, :-1]
    gain = pd.Series(m.get_booster().get_score(importance_type='gain')).reindex(cols).fillna(0)
    rows = []
    for i, c in enumerate(cols):
        v = X[c].values; ok = ~np.isnan(v.astype(float))
        rho = spearmanr(v[ok], shap[ok, i])[0] if np.nanstd(v) > 0 else np.nan
        rows.append(dict(feature=c, block=blocks[c], perm_AUC_drop=round(base - np.mean(perm_p[c]), 4),
                         perm_sd=round(np.std([base_p[k] - perm_p[c][k] for k in range(len(SEEDS))]), 4),
                         shap_mean_abs=round(np.abs(shap[:, i]).mean(), 4),
                         direction=('값↑ → 많음↑' if rho > 0.2 else '값↑ → 많음↓' if rho < -0.2 else '불분명'),
                         gain_share=round(gain[c] / gain.sum(), 3)))
    out = pd.DataFrame(rows).sort_values('perm_AUC_drop', ascending=False)
    section(f'A16 피처 중요도 — 최종 분류기, 새 지역 검증 (기준 AUC {base:.3f})', out, 'A16_importance')
    blk = out.groupby('block')[['perm_AUC_drop', 'shap_mean_abs', 'gain_share']].sum().sort_values('perm_AUC_drop', ascending=False).round(3)
    section('A16b 블록별 합계', blk.reset_index(), 'A16_importance_blocks')


# ── A17 ────────────────────────────────────────────────────────────────────
def perm_drop(X, yb, g, cols, seed, n_splits=4, reps=3):
    """새 지역(묶음) CV에서 피처별 permutation AUC 하락. 음수 = 섞는 게 오히려 나음 = 새 지역에서 노이즈."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    from train import PARAMS_CLS
    rng = np.random.default_rng(seed)
    prob = np.zeros(len(yb)); pp = {c: np.zeros(len(yb)) for c in cols}
    for tr, te in GroupKFold(n_splits, shuffle=True, random_state=seed).split(X, yb, g):
        spw = (yb[tr] == 0).sum() / max((yb[tr] == 1).sum(), 1)
        m = XGBClassifier(**PARAMS_CLS, scale_pos_weight=spw, random_state=seed).fit(X.iloc[tr][cols], yb[tr])
        Xte = X.iloc[te][cols]; prob[te] = m.predict_proba(Xte)[:, 1]
        for c in cols:
            for _ in range(reps):
                Xp = Xte.copy(); Xp[c] = rng.permutation(Xp[c].values); pp[c][te] += m.predict_proba(Xp)[:, 1] / reps
    base = roc_auc_score(yb, prob)
    return pd.Series({c: base - roc_auc_score(yb, pp[c]) for c in cols})


def a17(C, T, K, TK):
    """중첩 검증: 안쪽 새 지역 CV의 permutation 중요도가 음수인 피처를 빼는 '규칙'이 바깥 새 지역에서도 이득인가."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from xgboost import XGBClassifier
    from train import load_features, label_high, high_quantile, spatial_groups, USE_CLS, PARAMS_CLS
    df, Tr = load_features(); y = Tr['total']; yb, _ = label_high(df, y, high_quantile())
    X, g, cols = add_history(df, y), spatial_groups(df), cols_of(USE_CLS)
    fixed3 = [c for c in cols if c not in ('t_beach_len', 'gap_days', 't_armored')]
    res = {k: [] for k in ['전체 피처', '중첩 규칙으로 제거 (정직한 값)', '고정 3개 제거 (전체 데이터로 고름, 낙관적)']}
    bal = {k: [] for k in res}
    dropped = pd.Series(0, index=cols)
    for seed in SEEDS:
        P = {k: np.zeros(len(yb)) for k in res}
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, yb, g):
            imp = perm_drop(X.iloc[tr], yb[tr], g[tr], cols, seed)
            keep = [c for c in cols if imp[c] >= 0]
            dropped[[c for c in cols if c not in keep]] += 1
            spw = (yb[tr] == 0).sum() / max((yb[tr] == 1).sum(), 1)
            for k, use in [('전체 피처', cols), ('중첩 규칙으로 제거 (정직한 값)', keep), ('고정 3개 제거 (전체 데이터로 고름, 낙관적)', fixed3)]:
                m = XGBClassifier(**PARAMS_CLS, scale_pos_weight=spw, random_state=seed).fit(X.iloc[tr][use], yb[tr])
                P[k][te] = m.predict_proba(X.iloc[te][use])[:, 1]
        for k in res:
            res[k].append(roc_auc_score(yb, P[k])); bal[k].append(balanced_accuracy_score(yb, P[k] > 0.5))
    out = pd.DataFrame([dict(setup=k, AUC=round(np.mean(v), 3), AUC_sd=round(np.std(v), 3), balanced_acc=round(np.mean(bal[k]), 3))
                        for k, v in res.items()])
    section('A17 피처 제거의 중첩 검증 — 바깥: 새 지역 5-fold × 3 seed, 안쪽: 학습 데이터만으로 새 지역 4-fold permutation',
            out, 'A17_nested_drop')
    freq = (dropped / (len(SEEDS) * 5)).sort_values(ascending=False).round(2)
    section('A17b 바깥 fold 15개 중 제거된 비율 (안쪽 규칙 기준)', freq[freq > 0].rename('dropped_frac').reset_index().rename(
        columns={'index': 'feature'}), 'A17_dropped_freq')


# ── A18 ────────────────────────────────────────────────────────────────────
def a18(C, T, K, TK):
    """모델 구조 비교 — 같은 피처·라벨·새 지역 검증. 설정은 미리 고정하고 한 번만 비교 (튜닝 없음)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from xgboost import XGBClassifier
    from train import load_features, label_high, high_quantile, spatial_groups, USE_CLS, PARAMS_CLS
    df, Tr = load_features(); y = Tr['total']; yb, _ = label_high(df, y, high_quantile())
    X, cols = add_history(df, y), cols_of(USE_CLS)
    imp = lambda: SimpleImputer(strategy='median', add_indicator=True)
    models = {
        '로지스틱 회귀 (L2)': lambda s, spw: make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)),
        '랜덤 포레스트': lambda s, spw: make_pipeline(imp(), RandomForestClassifier(500, min_samples_leaf=5, max_features=0.5, class_weight='balanced_subsample', random_state=s, n_jobs=-1)),
        'Extra Trees': lambda s, spw: make_pipeline(imp(), ExtraTreesClassifier(500, min_samples_leaf=5, max_features=0.5, class_weight='balanced', random_state=s, n_jobs=-1)),
        'HistGradientBoosting': lambda s, spw: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=5.0, class_weight='balanced', random_state=s),
        'XGBoost (현재)': lambda s, spw: XGBClassifier(**PARAMS_CLS, scale_pos_weight=spw, random_state=s),
    }
    for split, g in [('새 지역 (주 지표)', spatial_groups(df)), ('새 해변 (참고)', df.loc_id.values)]:
        auc = {k: [] for k in list(models) + ['앙상블 (로지스틱+RF+XGB 평균)']}; bal = {k: [] for k in auc}
        for seed in SEEDS:
            P = {k: np.zeros(len(yb)) for k in models}
            for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, yb, g):
                spw = (yb[tr] == 0).sum() / max((yb[tr] == 1).sum(), 1)
                for k, mk in models.items():
                    P[k][te] = mk(seed, spw).fit(X.iloc[tr][cols], yb[tr]).predict_proba(X.iloc[te][cols])[:, 1]
            P['앙상블 (로지스틱+RF+XGB 평균)'] = np.mean([P['로지스틱 회귀 (L2)'], P['랜덤 포레스트'], P['XGBoost (현재)']], 0)
            for k in auc:
                auc[k].append(roc_auc_score(yb, P[k])); bal[k].append(balanced_accuracy_score(yb, P[k] > 0.5))
        out = pd.DataFrame([dict(model=k, AUC=round(np.mean(v), 3), AUC_sd=round(np.std(v), 3), balanced_acc=round(np.mean(bal[k]), 3))
                            for k, v in auc.items()]).sort_values('AUC', ascending=False)
        section(f'A18 모델 구조 비교 — {split}, 세션당 총무게 \'많음\' (5-fold × 3 seed)', out,
                'A18_models_' + ('region' if '지역' in split else 'beach'))


# ── A19 ────────────────────────────────────────────────────────────────────
def a19(C, T, K, TK):
    """모델 선택의 중첩 검증: 안쪽 새 지역 CV에서 가장 좋은 모델을 고르는 '절차'의 바깥 성능 vs XGBoost/HGB 고정."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from xgboost import XGBClassifier
    from train import load_features, label_high, high_quantile, spatial_groups, USE_CLS, PARAMS_CLS
    df, Tr = load_features(); y = Tr['total']; yb, _ = label_high(df, y, high_quantile())
    X, cols, g = add_history(df, y), cols_of(USE_CLS), spatial_groups(df)
    imp = lambda: SimpleImputer(strategy='median', add_indicator=True)
    models = {
        'logit': lambda s, spw: make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000)),
        'rf': lambda s, spw: make_pipeline(imp(), RandomForestClassifier(300, min_samples_leaf=5, max_features=0.5, class_weight='balanced_subsample', random_state=s, n_jobs=-1)),
        'et': lambda s, spw: make_pipeline(imp(), ExtraTreesClassifier(300, min_samples_leaf=5, max_features=0.5, class_weight='balanced', random_state=s, n_jobs=-1)),
        'hgb': lambda s, spw: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=5.0, class_weight='balanced', random_state=s),
        'xgb': lambda s, spw: XGBClassifier(**PARAMS_CLS, scale_pos_weight=spw, random_state=s),
    }
    spw_of = lambda yy: (yy == 0).sum() / max((yy == 1).sum(), 1)
    res = {k: [] for k in ['중첩 선택 (정직한 값)', 'XGBoost 고정', 'HGB 고정']}; bal = {k: [] for k in res}; picks = []
    for seed in SEEDS:
        P = {k: np.zeros(len(yb)) for k in res}
        for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, yb, g):
            inner = {k: np.zeros(len(tr)) for k in models}
            for itr, ite in GroupKFold(4, shuffle=True, random_state=seed).split(X.iloc[tr], yb[tr], g[tr]):
                a, b = tr[itr], tr[ite]
                for k, mk in models.items():
                    inner[k][ite] = mk(seed, spw_of(yb[a])).fit(X.iloc[a][cols], yb[a]).predict_proba(X.iloc[b][cols])[:, 1]
            best = max(models, key=lambda k: roc_auc_score(yb[tr], inner[k])); picks.append(best)
            for k, name in [(best, '중첩 선택 (정직한 값)'), ('xgb', 'XGBoost 고정'), ('hgb', 'HGB 고정')]:
                P[name][te] = models[k](seed, spw_of(yb[tr])).fit(X.iloc[tr][cols], yb[tr]).predict_proba(X.iloc[te][cols])[:, 1]
        for k in res:
            res[k].append(roc_auc_score(yb, P[k])); bal[k].append(balanced_accuracy_score(yb, P[k] > 0.5))
    out = pd.DataFrame([dict(setup=k, AUC=round(np.mean(v), 3), AUC_sd=round(np.std(v), 3), balanced_acc=round(np.mean(bal[k]), 3))
                        for k, v in res.items()])
    section(f'A19 모델 선택 중첩 검증 — 새 지역 (fold마다 고른 모델: {pd.Series(picks).value_counts().to_dict()})', out, 'A19_nested_model')


# ── A20 ────────────────────────────────────────────────────────────────────
def a20(C, T, K, TK):
    """모든 모델 방식 비교 — 같은 피처(USE_CLS)·라벨·검증. 설정은 미리 고정, 튜닝 없음.
    점수는 AUC(순위)와 균형정확도(0.5 컷, 클래스 불균형 보정). 회귀→임계값은 예측 무게를 점수로 사용."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from train import load_features, label_high, high_quantile, spatial_groups, USE_CLS, PARAMS_CLS, PARAMS_FINAL
    df, Tr = load_features(); y = Tr['total']; yb, thr = label_high(df, y, high_quantile())
    X, cols = add_history(df, y), cols_of(USE_CLS)
    imp = lambda: SimpleImputer(strategy='median', add_indicator=True)
    pre = lambda *m: make_pipeline(imp(), StandardScaler(), *m)
    M = {  # (계열, 생성자(seed, scale_pos_weight))
        '로지스틱 회귀': ('선형', lambda s, w: pre(LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000))),
        'SVM (RBF)': ('커널', lambda s, w: pre(SVC(C=1.0, class_weight='balanced', probability=True, random_state=s))),
        'k-최근접 이웃 (k=15)': ('거리', lambda s, w: pre(KNeighborsClassifier(15, weights='distance'))),
        '나이브 베이즈': ('확률', lambda s, w: pre(GaussianNB())),
        '신경망 MLP (16)': ('신경망', lambda s, w: pre(MLPClassifier((16,), alpha=1e-1, max_iter=2000, random_state=s))),
        '랜덤 포레스트': ('배깅 트리', lambda s, w: make_pipeline(imp(), RandomForestClassifier(500, min_samples_leaf=5, max_features=0.5, class_weight='balanced_subsample', random_state=s, n_jobs=-1))),
        'Extra Trees': ('배깅 트리', lambda s, w: make_pipeline(imp(), ExtraTreesClassifier(500, min_samples_leaf=5, max_features=0.5, class_weight='balanced', random_state=s, n_jobs=-1))),
        'HistGradientBoosting': ('부스팅 트리', lambda s, w: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=5.0, class_weight='balanced', random_state=s)),
        'XGBoost (현재)': ('부스팅 트리', lambda s, w: XGBClassifier(**PARAMS_CLS, scale_pos_weight=w, random_state=s)),
        'LightGBM': ('부스팅 트리', lambda s, w: LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=2, num_leaves=4, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20, scale_pos_weight=w, random_state=s, verbose=-1)),
        'CatBoost': ('부스팅 트리', lambda s, w: CatBoostClassifier(iterations=300, learning_rate=0.03, depth=3, l2_leaf_reg=20, auto_class_weights='Balanced', random_seed=s, verbose=0, allow_writing_files=False)),
    }
    rows, roc, FPR = [], {}, np.linspace(0, 1, 201)
    for split, g in [('새 지역', spatial_groups(df)), ('새 해변', df.loc_id.values)]:
        auc = {k: [] for k in list(M) + ['회귀 → 임계값 (XGBoost 회귀)', '앙상블 (전 모델 순위 평균)']}; bal = {k: [] for k in auc}
        for seed in SEEDS:
            P = {k: np.zeros(len(yb)) for k in auc}
            for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(X, yb, g):
                w = (yb[tr] == 0).sum() / max((yb[tr] == 1).sum(), 1)
                for k, (_, mk) in M.items():
                    P[k][te] = mk(seed, w).fit(X.iloc[tr][cols], yb[tr]).predict_proba(X.iloc[te][cols])[:, 1]
                reg = XGBRegressor(**PARAMS_FINAL, random_state=seed).fit(X.iloc[tr][cols], y.values[tr])
                P['회귀 → 임계값 (XGBoost 회귀)'][te] = reg.predict(X.iloc[te][cols]) - thr + 0.5  # 기준 무게를 넘으면 >0.5
            P['앙상블 (전 모델 순위 평균)'] = np.mean([pd.Series(P[k]).rank(pct=True).values for k in M], 0)
            for k in auc:
                auc[k].append(roc_auc_score(yb, P[k]))
                cut = 0.5 if k != '앙상블 (전 모델 순위 평균)' else 1 - yb.mean()
                bal[k].append(balanced_accuracy_score(yb, P[k] > cut))
                if split == '새 지역':  # ROC를 공통 FPR 격자에 보간 → seed 평균 (행 단위 예측은 저장하지 않음)
                    fpr, tpr, _ = roc_curve(yb, P[k]); roc.setdefault(k, []).append(np.interp(FPR, fpr, tpr))
        if split == '새 지역':
            curves = pd.DataFrame({k: np.mean(v, 0) for k, v in roc.items()}).assign(**{'FPR': FPR})
            curves.iloc[0, :-1] = 0.0
            curves.round(4).to_csv(OUT / 'A20_roc_new_region.csv', index=False)
        for k in auc:
            fam = M[k][0] if k in M else ('회귀 후 분류' if '회귀' in k else '앙상블')
            rows.append(dict(split=split, model=k, family=fam, AUC=round(np.mean(auc[k]), 3), AUC_sd=round(np.std(auc[k]), 3),
                             balanced_acc=round(np.mean(bal[k]), 3)))
    out = pd.DataFrame(rows)
    wide = out.pivot(index=['model', 'family'], columns='split', values=['AUC', 'AUC_sd', 'balanced_acc'])
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide = wide.reset_index().sort_values('AUC_새 지역', ascending=False)
    section('A20 전체 모델 비교 — 세션당 총무게 \'많음\' 분류 (5-fold × 3 seed, 주 지표 = 새 지역 AUC)', wide, 'A20_all_models')


# ── 리포트 그림 ─────────────────────────────────────────────────────────────
def figures(*_):
    """report.md용 그림 2장 (A20, A16 결과 CSV에서). 색: 기준 팔레트 1·2번 슬롯 (검증 통과)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    S1, S2, INK, INK2, GRID, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 10,
                         'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
    Path('figures').mkdir(exist_ok=True)

    # 1) 모델별 AUC: 새 지역(주 지표, 오차막대) vs 새 해변(참고)
    m = pd.read_csv(OUT / 'A20_all_models.csv').sort_values('AUC_새 지역')
    fig, ax = plt.subplots(figsize=(8, 6.2))
    yy = np.arange(len(m))
    ax.hlines(yy, m['AUC_새 지역'], m['AUC_새 해변'], color=GRID, lw=2, zorder=1)
    ax.errorbar(m['AUC_새 지역'], yy, xerr=m['AUC_sd_새 지역'], fmt='o', ms=8, color=S1, ecolor=S1, elinewidth=2,
                capsize=0, zorder=3, label='새 지역 (주 지표: 10km 안에 학습 해변 없음)')
    ax.plot(m['AUC_새 해변'], yy, 'o', ms=8, mfc=SURF, mec=S2, mew=2, zorder=2, label='새 해변 (참고: 이웃 해변은 학습됨)')
    for i, v, e in zip(yy, m['AUC_새 지역'], m['AUC_sd_새 지역']):
        ax.text(v - e - 0.006, i, f'{v:.3f}', va='center', ha='right', color=INK, fontsize=9)
    ax.axvline(0.5, color=INK2, lw=1, ls='--'); ax.text(0.503, -0.75, '무작위 = 0.5', color=INK2, fontsize=8.5)
    ax.set_yticks(yy, m.model); ax.set_xlim(0.4, 0.86); ax.set_ylim(-1.1, len(m) - 0.5); ax.set_xlabel('AUC (높을수록 좋음)')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_title("모델별 '쓰레기 많음' 판정 성능 (5-fold × 3회, 막대 = ±1 표준편차)", loc='left', color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig('figures/model_comparison.png', dpi=150); plt.close(fig)

    # 2) 피처 중요도: 새 지역에서 그 피처를 섞었을 때 AUC 하락
    names = {'t_rock': '암반 정도 (위성)', 'month_sin': '계절 (봄 쪽)', 'visitors_log': '해수욕장 이용객 수', 't_urban': '배후 도시화 (위성)',
             'summer_days': '해수욕 시즌 일수', 'year': '연도', 'tide_spring_day': '수거일 사리 여부', 'month_cos': '계절 (겨울 쪽)',
             'tide_spring_win': '구간 사리 비율', 't_tidal_flat': '갯벌 (위성)', 't_armored': '호안·도로 (위성)',
             'gap_days': '직전 수거 후 경과일', 't_beach_len': '해변 길이 (위성)'}
    f = pd.read_csv(OUT / 'A16_importance.csv')
    f = pd.concat([f.head(8), f.tail(3)]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    col = [S1 if v > 0 else '#b5b3ad' for v in f.perm_AUC_drop]
    ax.barh(np.arange(len(f)), f.perm_AUC_drop, xerr=f.perm_sd, color=col, height=0.6, error_kw=dict(ecolor=INK2, lw=1))
    ax.set_yticks(np.arange(len(f)), [names.get(c, c) for c in f.feature])
    up = {'t_rock': '바위가 많을수록', 'month_sin': '봄(2–4월)일수록', 'visitors_log': '이용객이 많을수록', 't_urban': '도시화될수록',
          'summer_days': '여름이 낀 구간일수록', 'year': '최근일수록', 'tide_spring_day': '사리 날일수록', 'month_cos': '겨울일수록',
          'tide_spring_win': '사리 비율이 높을수록', 't_tidal_flat': '갯벌이 있으면'}
    for i, (c, v, e, d) in enumerate(zip(f.feature, f.perm_AUC_drop, f.perm_sd, f.direction)):
        lab = (f"{up.get(c, c)} {'많음' if '많음↑' in d else '적음'}" if v > 0 else '새 지역에선 노이즈 (섞으면 오히려 좋아짐)')
        ax.text(max(v + e, 0) + 0.003, i, lab, va='center', color=INK2, fontsize=8.5)
    ax.tick_params(axis='y', length=0)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel('테스트 지역에서 섞었을 때 AUC 하락 (클수록 중요)')
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.set_title('무엇이 판정에 영향을 줬나 — XGBoost, 새 지역 검증 (permutation 중요도)', loc='left', color=INK, fontsize=11)
    ax.set_xlim(-0.065, 0.11)
    fig.tight_layout(); fig.savefig('figures/feature_importance.png', dpi=150); plt.close(fig)

    # 3) ROC 곡선: 새 지역, seed 평균. 색 = 기준 팔레트 1~4번 슬롯 (검증 통과, 3·4번 대비 부족 → 범례에 AUC 수치 + 리포트 표)
    r, a = pd.read_csv(OUT / 'A20_roc_new_region.csv'), m.set_index('model')['AUC_새 지역']
    show = ['HistGradientBoosting', 'CatBoost', 'XGBoost (현재)', '로지스틱 회귀']
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls='--', label='무작위 (AUC 0.500)')
    for k, c in zip(show, [S1, S2, '#1baf7a', '#eda100']):
        ax.plot(r.FPR, r[k], color=c, lw=2, label=f'{k}  AUC {a[k]:.3f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal')
    ax.set_xlabel("잘못 울린 경보 비율 (실제 '적음' 중 '많음'으로 판정, FPR)")
    ax.set_ylabel("잡아낸 비율 (실제 '많음' 중 '많음'으로 판정, TPR)")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_title('ROC 곡선 — 새 지역 검증 (5-fold × 3회 평균)', loc='left', color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig('figures/roc_curves.png', dpi=150); plt.close(fig)
    print('figures/model_comparison.png, figures/feature_importance.png, figures/roc_curves.png')


if __name__ == '__main__':
    import sys
    C, T, K, TK = load_all()
    print(f'caresea rows {len(C)} / beaches {C.loc_id.nunique()} | KOEM rows {len(K)} / stations {K.loc_id.nunique()}')
    run = sys.argv[1:] or ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11']
    for name in run:
        f = globals()[name]
        f() if name == 'a2' else f(C, T) if name == 'a3' else f(C, T, K, TK)
