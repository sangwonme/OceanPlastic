"""원천 데이터 로드 → 조인 → augmentation
→ outputs/features.csv       (반려해변: 한 행 = 해변 × 수거일)
→ outputs/koem_features.csv  (국가 모니터링: 한 행 = 정점 × 조사일, 노력 표준화된 비교군)

[0] 로드  [1] 타깃 테이블  [2] 위치 정적 룩업  [3] ERA5 일별 시계열(캐시)  [4] 구간 날씨 집계  [5] 달력·조석
위치(해변/정점)는 loc_id 하나로 일반화 — 같은 피처 코드를 두 데이터셋에 그대로 쓴다.
history rollout은 타깃마다 달라서 train.py에서 만든다.
"""
import json, hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from matplotlib.path import Path as MplPath

D, OUT = Path('dataset'), Path('outputs')
CACHE = OUT / 'cache'
ERA5_FILES = ['era5_2018_2019', 'era5_2020_2021', 'era5_2022_2023', 'era5_2024_2025', 'era5_2026']
BOX = dict(lat=(33, 38.6), lng=(125.4, 131.1))  # ERA5 범위
FIRST_WINDOW_DAYS = 30  # 직전 기록이 없을 때 구간 길이
BUOY_METHODS = '수하|연승|가두리|부류|홍식'  # 부표를 띄우는 양식 방법 = 스티로폼 부표 공급원
N_DIR, FETCH_MAX = 16, 100  # 해안 fetch: 16방위, 최대 100km
LUNAR_MONTH, NEW_MOON = 29.530588853, pd.Timestamp('2000-01-06 18:14')
CAT_COLS = None  # load_sessions에서 채움


def show(name, df):
    print(f'\n[{name}] {df.shape}')
    print(df.head(3).to_string(max_cols=14))


def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def dist_matrix(a, b):
    return haversine(a.lat.values[:, None], a.lng.values[:, None], b.lat.values[None], b.lng.values[None])


def in_box(df):
    return df.lat.between(*BOX['lat']) & df.lng.between(*BOX['lng'])


# ── [0] 로드 ────────────────────────────────────────────────────────────────
def load_sessions():
    global CAT_COLS
    s = pd.read_csv(D / 'caresea/sessions_2026-11-01.csv', parse_dates=['date'])
    CAT_COLS = [c for c in s.columns if c.endswith('_g')]
    s['day'] = s['date'].dt.normalize()
    return s


def load_beaches():
    b = pd.read_csv(D / 'caresea/beach_daily_2026-11-08.csv').drop_duplicates('beach_id')
    b = b[['beach_id', 'beach_name', 'sido', 'sigungu', 'beach_lat', 'beach_lng']].rename(
        columns={'beach_lat': 'lat', 'beach_lng': 'lng'})
    b = b[in_box(b)].reset_index(drop=True)
    b['loc_id'] = 'b' + b.beach_id.astype(str)
    return b


def load_geo():
    ports = pd.read_csv(D / 'geo/fishing_ports.csv')
    ports = ports[ports.type.isin(['국가어항', '지방어항'])]  # 전국 기준이 같은 등급만
    farms = pd.read_csv(D / 'geo/aquaculture_farms_2022.csv')
    farms['buoy'] = farms.method.fillna('').str.contains(BUOY_METHODS)
    koem = pd.read_csv(D / 'koem/national_monitoring_2008_2023.csv')
    koem_prior = koem.dropna(subset=['station_lat']).groupby('station_name').agg(
        lat=('station_lat', 'first'), lng=('station_lng', 'first'), weight_kg=('weight_kg', 'mean')).reset_index()
    land = []
    for f in json.load(open(D / 'geo/korea_land_ne10m.geojson'))['features']:
        ring = np.array(f['geometry']['coordinates'][0])
        land.append((MplPath(ring), ring.min(0), ring.max(0)))
    visitors = pd.read_csv(D / 'geo/beach_visitors.csv')
    labels = pd.read_csv(D / 'geo/site_labels.csv')
    return dict(ports=ports, rivers=pd.read_csv(D / 'geo/river_mouths.csv'), farms=farms,
                koem_prior=koem_prior, land=land, visitors=visitors, labels=labels)


def encode_labels(lab):
    """위성 판독 라벨 → 숫자. 순서가 있는 건 0/1/2, 저질은 모래 기준 더미."""
    o = pd.DataFrame({'loc_id': lab.loc_id})
    o['t_rock'] = lab.rock_level
    o['t_tidal_flat'] = lab.tidal_flat
    o['t_enclosure'] = lab['shape'].map({'open': 0, 'semi': 1, 'pocket': 2})
    o['t_beach_len'] = lab.beach_len.map({'short': 0, 'medium': 1, 'long': 2})
    o['t_harbor'] = lab.harbor_500m
    o['t_armored'] = lab.armored_back
    o['t_urban'] = lab.backshore.map({'natural': 0, 'mixed': 1, 'urban': 2})
    o['t_stream'] = lab.stream_outlet
    o['t_aqua_visible'] = lab.aquaculture_visible
    for c in ('gravel', 'rock', 'mud'):
        o[f't_sub_{c}'] = (lab.substrate == c).astype(int)
    return o


# ── [1] 타깃 테이블 ─────────────────────────────────────────────────────────
def match_gps_only(s, beaches, max_km=1.0):
    """beach_id 없는 세션을 1km 안 최근접 등록 해변에 붙인다."""
    m = s.beach_id.isna()
    d = dist_matrix(s[m], beaches)
    s.loc[m, 'beach_id'] = np.where(d.min(1) <= max_km, beaches.beach_id.values[d.argmin(1)], np.nan)
    return s


def build_target(s, beaches):
    s = match_gps_only(s.copy(), beaches)
    s = s[s.beach_id.isin(beaches.beach_id)]
    # 수거가 있었다는 사실(gap)에는 estimated 세션도 포함, 무게 타깃은 measured만
    visits = s.groupby(['beach_id', 'day']).size().rename('n_visit').reset_index()
    visits['prev_day'] = visits.groupby('beach_id')['day'].shift()
    m = s[s.weight_method == 'measured']
    g = m.groupby(['beach_id', 'day'])
    t = g[CAT_COLS + ['item_count']].sum()
    t['n_sessions'] = g.size()
    t['distance_km'] = g.distance_km.sum(min_count=1)  # 참여자 이동 거리 합 (노력량)
    t = t.reset_index().merge(visits, on=['beach_id', 'day']).drop(columns='n_visit')
    t = finish_target(t.rename(columns={'day': 'date'}), 'beach_id')
    t['beach_id'] = t.beach_id.astype(int)
    t['loc_id'] = 'b' + t.beach_id.astype(str)
    assert np.allclose(t.total_weight_g, t[[c for c in CAT_COLS if c not in
                                            ('total_weight_g', 'plastic_weight_g', 'non_plastic_weight_g')]].sum(1))
    return t


def build_koem_target():
    k = pd.read_csv(D / 'koem/national_monitoring_2008_2023.csv', parse_dates=['survey_date'])
    k = k.dropna(subset=['survey_date', 'station_lat', 'weight_kg']).rename(
        columns={'survey_date': 'date', 'station_lat': 'lat', 'station_lng': 'lng'})
    k = k[in_box(k)].sort_values(['station_name', 'date']).drop_duplicates(['station_name', 'date'])
    k['prev_day'] = k.groupby('station_name').date.shift()
    k = finish_target(k, 'station_name')
    k = k[k.win_start >= '2018-01-01'].reset_index(drop=True)  # ERA5 시작 이후만
    ids = {n: f'k{i}' for i, n in enumerate(sorted(k.station_name.unique()))}
    k['loc_id'] = k.station_name.map(ids)
    return k


def finish_target(t, key):
    t['gap_days'] = (t.date - t.prev_day).dt.days
    t['is_first'] = t.prev_day.isna().astype(int)
    t['win_start'] = t.prev_day.fillna(t.date - pd.Timedelta(days=FIRST_WINDOW_DAYS))
    assert not t.duplicated([key, 'date']).any() and (t.win_start < t.date).all()
    return t.sort_values([key, 'date']).reset_index(drop=True)


# ── [2] 위치 정적 룩업 ──────────────────────────────────────────────────────
def coast_fetch(locs, land):
    """16방위 fetch(km): 위치에서 바다로 나간 뒤 다음 육지까지 거리. 5km 안에 바다가 없으면 0(내륙 방향).
    ponytail: Natural Earth 10m 해안선 — 작은 섬·만은 누락, 더 정밀하려면 국립해양조사원 해안선으로 교체."""
    ang = np.arange(N_DIR) * 2 * np.pi / N_DIR  # 0 = 북, 시계방향
    dist = np.arange(0.5, FETCH_MAX + 0.5, 0.5)
    F = np.zeros((len(locs), N_DIR))
    for i, (la, lo) in enumerate(zip(locs.lat, locs.lng)):
        plng = lo + np.sin(ang)[:, None] * dist[None] / (111.32 * np.cos(np.radians(la)))
        plat = la + np.cos(ang)[:, None] * dist[None] / 110.57
        pts = np.c_[plng.ravel(), plat.ravel()]
        on = np.zeros(len(pts), bool)
        for p, lo_, hi_ in land:
            if lo_[0] < lo + 1.3 and hi_[0] > lo - 1.3 and lo_[1] < la + 1 and hi_[1] > la - 1:
                on |= p.contains_points(pts)
        on = on.reshape(N_DIR, -1)
        for k in range(N_DIR):
            sea = np.flatnonzero(~on[k])
            if len(sea) == 0 or dist[sea[0]] > 5:
                continue
            hit = np.flatnonzero(on[k, sea[0]:])
            F[i, k] = dist[sea[0] + hit[0]] - dist[sea[0]] if len(hit) else FETCH_MAX
    return F


@lru_cache
def era5_grids():
    f = D / f'era5/{ERA5_FILES[-1]}.grib'
    kw = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
    u = xr.open_dataset(f, filter_by_keys={'shortName': '10u'}, **kw)
    w = xr.open_dataset(f, filter_by_keys={'shortName': 'swh'}, **kw).swh.isel(time=slice(0, 24))
    sea = np.argwhere(w.notnull().any('time').values)
    wave = pd.DataFrame({'lat': w.latitude.values[sea[:, 0]], 'lng': w.longitude.values[sea[:, 1]]})
    return u.latitude.values, u.longitude.values, wave


def era5_cells(locs):
    """대기 0.25° 최근접 셀, 파랑 0.5° 최근접 '바다' 셀(연안 셀은 NaN이라)."""
    alat, alng, wave = era5_grids()
    dw = dist_matrix(locs, wave)
    return pd.DataFrame({
        'atm_lat': alat[np.abs(locs.lat.values[:, None] - alat[None]).argmin(1)],
        'atm_lng': alng[np.abs(locs.lng.values[:, None] - alng[None]).argmin(1)],
        'wave_lat': wave.lat.values[dw.argmin(1)], 'wave_lng': wave.lng.values[dw.argmin(1)],
        'wave_km': dw.min(1)}, index=locs.index)


def build_lookup(locs, geo):
    """locs: loc_id, lat, lng (+ beach_name/sigungu면 관광 매칭)."""
    lk = pd.concat([locs, era5_cells(locs)], axis=1)

    # 해안 노출도 + 법선(해변→바다 단위벡터, fetch 가중)
    F = coast_fetch(lk, geo['land'])
    ang = np.arange(N_DIR) * 2 * np.pi / N_DIR
    lk['fetch_mean_km'] = F.mean(1)
    lk['open_sectors'] = (F >= 50).sum(1)
    nx, ny = (F * np.sin(ang)).sum(1), (F * np.cos(ang)).sum(1)
    # 해안선이 너무 거칠어 fetch가 전부 0이면 파랑 바다 셀 방향으로 대체
    fx = (lk.wave_lng - lk.lng) * np.cos(np.radians(lk.lat))
    fy = lk.wave_lat - lk.lat
    zero = np.hypot(nx, ny) == 0
    nx, ny = np.where(zero, fx, nx), np.where(zero, fy, ny)
    lk['normal_x'], lk['normal_y'] = nx / np.hypot(nx, ny), ny / np.hypot(nx, ny)
    for k in range(N_DIR):
        lk[f'F{k}'] = F[:, k]

    p = geo['ports']
    dp = dist_matrix(lk, p)
    lk['port_near_km'] = dp.min(1)
    lk['port_n_10km'] = (dp < 10).sum(1)
    lk['port_national_near_km'] = dp[:, (p.type == '국가어항').values].min(1)

    r = geo['rivers']
    dr = dist_matrix(lk, r)
    lk['river_near_km'] = dr.min(1)
    lk['river_basin_w'] = np.where(dr < 30, r.basin_area_km2.values[None] / np.maximum(dr, 1), 0).sum(1)

    fm = geo['farms']
    df_ = dist_matrix(lk, fm)
    area, buoy = fm.area_ha.values[None], fm.buoy.values[None]
    for km in (5, 10, 20):
        lk[f'buoy_farm_ha_{km}km'] = np.where((df_ < km) & buoy, area, 0).sum(1)
    lk['farm_ha_10km'] = np.where(df_ < 10, area, 0).sum(1)
    lk['buoy_farm_near_km'] = np.where(buoy, df_, np.inf).min(1)

    dk = dist_matrix(lk, geo['koem_prior'])
    lk['koem_prior_kg'] = geo['koem_prior'].weight_kg.values[dk.argmin(1)]

    lk = lk.merge(encode_labels(geo['labels']), on='loc_id', how='left')
    lk['visitors_log'], lk['is_designated'] = np.nan, 0
    if 'beach_name' in lk:
        v = geo['visitors']
        norm = lambda x: x.astype(str).str.replace(r'\s|해수욕장|해변|해안', '', regex=True)
        v = v.assign(key=norm(v['해수욕장명']), sgg=v['시군구'].astype(str).str[:2],
                     vis=np.log1p(v[['2022년', '2023년', '2024년', '2025년']].mean(1)))
        m = lk.assign(key=norm(lk.beach_name), sgg=lk.sigungu.astype(str).str[:2]).merge(
            v[['key', 'sgg', 'vis']].drop_duplicates(['key', 'sgg']), on=['key', 'sgg'], how='left')
        lk['visitors_log'], lk['is_designated'] = m.vis.values, m.vis.notna().astype(int).values
    return lk


# ── [3] ERA5 일별 시계열 (위치 단위, KST 일) ─────────────────────────────────
def era5_hourly(lk):
    kw = dict(engine='cfgrib', backend_kwargs={'indexpath': ''})
    H = {}
    for short, la, lo in [('10u', 'atm_lat', 'atm_lng'), ('10v', 'atm_lat', 'atm_lng'), ('tp', 'atm_lat', 'atm_lng'),
                          ('swh', 'wave_lat', 'wave_lng'), ('mwd', 'wave_lat', 'wave_lng')]:
        parts = []
        for f in ERA5_FILES:
            ds = xr.open_dataset(D / f'era5/{f}.grib', filter_by_keys={'shortName': short}, **kw)
            da = ds[list(ds.data_vars)[0]].sel(latitude=xr.DataArray(lk[la].values, dims='b'),
                                               longitude=xr.DataArray(lk[lo].values, dims='b'), method='nearest')
            if 'step' in da.dims:  # tp: 예보 기준시각 × step(1~12h), 각 값은 1시간 누적
                vt = (da.time.values[:, None] + da.step.values[None]).ravel()
                vals = da.transpose('time', 'step', 'b').values.reshape(-1, len(lk))
            else:
                vt, vals = da.time.values, da.transpose('time', 'b').values
            parts.append(pd.DataFrame(vals, index=vt, columns=lk.loc_id.values))
        v = pd.concat(parts)
        H[short] = v[~v.index.duplicated()].sort_index()
        print(f'  era5 {short} {H[short].shape} {H[short].index.min()} → {H[short].index.max()}')
    return H


def era5_daily(lk):
    key = hashlib.md5(lk[['loc_id', 'normal_x', 'normal_y'] + [f'F{k}' for k in range(N_DIR)]]
                      .round(4).to_csv().encode()).hexdigest()[:8]
    path = CACHE / f'era5_daily_{key}.pkl'
    if path.exists():
        return pd.read_pickle(path)
    H = era5_hourly(lk)
    idx = H['10u'].index
    u, v = H['10u'], H['10v']
    tp, swh, mwd = (H[k].reindex(idx) for k in ('tp', 'swh', 'mwd'))
    nx, ny = lk.normal_x.values[None], lk.normal_y.values[None]
    spd = np.hypot(u, v)
    sector = (np.round(((np.degrees(np.arctan2(-u, -v)) + 360) % 360) / (360 / N_DIR)).astype(int) % N_DIR).values
    Fm = lk[[f'F{k}' for k in range(N_DIR)]].values.T  # (16, n_loc)
    fetch = Fm[sector, np.arange(len(lk))[None]] / FETCH_MAX  # 바람이 불어오는 쪽 바다가 얼마나 열려 있나
    hourly = {
        'rain': (tp * 1000, 'sum'), 'rain_max_1h': (tp * 1000, 'max'),
        'onshore_mean': (-(u * nx + v * ny), 'mean'), 'onshore_max': (-(u * nx + v * ny), 'max'),
        'wind_fetch': (spd * fetch, 'mean'), 'strong_h': (spd > 10, 'sum'),
        'swh_mean': (swh, 'mean'), 'swh_max': (swh, 'max'), 'storm_h': (swh > 2, 'sum'),
        'wave_cos': (np.sin(np.radians(mwd)) * nx + np.cos(np.radians(mwd)) * ny, 'mean'),
    }
    daily = {}
    for name, (df, how) in hourly.items():
        df = df.set_axis(df.index + pd.Timedelta(hours=9))  # UTC → KST
        daily[name] = df.resample('D').agg(how).stack()
    out = pd.concat(daily, axis=1).rename_axis(['date', 'loc_id']).reset_index()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


# ── [4] 구간 날씨 집계 ──────────────────────────────────────────────────────
WIN_AGG = {  # (일별 컬럼, 집계)
    'rain_sum': ('rain', 'sum'), 'rain_max_1h': ('rain_max_1h', 'max'), 'rain_days_30': ('rain', 'n30'),
    'onshore_mean': ('onshore_mean', 'mean'), 'onshore_max': ('onshore_max', 'max'),
    'wind_fetch': ('wind_fetch', 'mean'), 'strong_h': ('strong_h', 'sum'),
    'swh_mean': ('swh_mean', 'mean'), 'swh_max': ('swh_max', 'max'), 'storm_h': ('storm_h', 'sum'),
    'wave_cos': ('wave_cos', 'mean'),
}
SHORT = ['rain_sum', 'onshore_mean', 'wind_fetch', 'swh_max', 'storm_h']  # 고정 7일·30일 창에 쓰는 것만


def add_weather(X, daily):
    """창 3종: gap = [직전 수거일, 이번 수거일), 7d/30d = 수거일 직전 7·30일. 당일은 제외."""
    by = {k: g.set_index('date').sort_index() for k, g in daily.groupby('loc_id')}
    rows = []
    for r in X.itertuples():
        d = by[r.loc_id]
        end = r.date - pd.Timedelta(days=1)
        row = {}
        for win, start, names in [('gap', r.win_start, list(WIN_AGG)),
                                  ('7d', r.date - pd.Timedelta(days=7), SHORT),
                                  ('30d', r.date - pd.Timedelta(days=30), SHORT)]:
            w = d.loc[start:end]
            for n in names:
                col, how = WIN_AGG[n]
                row[f'{n}_{win}'] = (w[col] > 30).sum() if how == 'n30' else w[col].agg(how) if len(w) else np.nan
        rows.append(row)
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ── [5] 달력·조석·관광 시즌 ────────────────────────────────────────────────
def spring_index(dates):
    """사리 지수: 삭·망(신월·보름)에 1, 상·하현에 -1. 조차가 큰 시기 = 만조선이 높이 올라감."""
    phase = ((pd.to_datetime(dates) - NEW_MOON).dt.total_seconds() / 86400) % LUNAR_MONTH
    return np.cos(4 * np.pi * phase / LUNAR_MONTH)


def add_calendar(X):
    m = X.date.dt.month
    X['month_sin'], X['month_cos'] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    X['year'] = X.date.dt.year
    X['tide_spring_day'] = spring_index(X.date).values
    win = [pd.date_range(s, e - pd.Timedelta(days=1)) for s, e in zip(X.win_start, X.date)]
    X['tide_spring_win'] = [spring_index(pd.Series(w)).mean() for w in win]
    X['summer_days'] = [w.month.isin([7, 8]).sum() for w in win]  # 해수욕 시즌(관광 쓰레기) 노출 일수
    return X


STATIC = ['fetch_mean_km', 'open_sectors', 'port_near_km', 'port_n_10km', 'port_national_near_km',
          'river_near_km', 'river_basin_w', 'buoy_farm_ha_5km', 'buoy_farm_ha_10km', 'buoy_farm_ha_20km',
          'farm_ha_10km', 'buoy_farm_near_km', 'koem_prior_kg', 'visitors_log', 'is_designated',
          't_rock', 't_tidal_flat', 't_enclosure', 't_beach_len', 't_harbor', 't_armored', 't_urban', 't_stream',
          't_aqua_visible', 't_sub_gravel', 't_sub_rock', 't_sub_mud']


def featurize(target, lk, daily, name):
    X = target.drop(columns=['lat', 'lng'], errors='ignore').merge(lk[['loc_id', 'lat', 'lng'] + STATIC], on='loc_id')
    X = add_weather(X, daily)
    X = add_calendar(X); show(f'{name} features', X)
    return X


def main():
    geo = load_geo()
    s = load_sessions(); show('sessions', s)
    beaches = load_beaches()
    target = build_target(s, beaches); show('target', target)
    kt = build_koem_target(); show('koem target', kt)

    lk_b = build_lookup(beaches[['loc_id', 'lat', 'lng', 'beach_name', 'sigungu']], geo)
    lk_k = build_lookup(kt.drop_duplicates('loc_id')[['loc_id', 'lat', 'lng']].reset_index(drop=True), geo)
    lk = pd.concat([lk_b, lk_k], ignore_index=True)
    show('lookup', lk[['loc_id', 'lat', 'lng', 'fetch_mean_km', 'open_sectors', 'normal_x', 'normal_y',
                       'buoy_farm_ha_10km', 'visitors_log']])
    daily = era5_daily(lk); show('era5_daily', daily)

    for t, name, out in [(target, 'caresea', 'features.csv'), (kt, 'koem', 'koem_features.csv')]:
        X = featurize(t, lk, daily, name)
        X.to_csv(OUT / out, index=False, encoding='utf-8-sig')
        print(f'→ {OUT / out} {X.shape}, locations {X.loc_id.nunique()}')
        print('  NA%:', (X.isna().mean() * 100).round(1)[lambda v: v > 0].to_dict())


if __name__ == '__main__':
    main()
