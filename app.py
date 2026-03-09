import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="属性別流入ダッシュボード",
    layout="wide"
)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["FC実施年月日"] = pd.to_datetime(df["FC実施年月日"], errors="coerce")
    df["年月"] = df["FC実施年月日"].dt.to_period("M").astype(str)

    df["入会フラグ"] = np.where(df["ステータス"] == "入会", 1, 0)
    df["入会ステータス"] = np.where(df["入会フラグ"] == 1, "入会", "非入会")

    age_order = ["10代前半", "18〜25", "26〜30", "31〜35", "36〜45", "46〜60", "61以上", "不明"]
    df["年代"] = pd.Categorical(df["年代"], categories=age_order, ordered=True)

    return df


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    st.sidebar.header("フィルタ")

    month_list = (
        df["年月"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )
    if len(month_list) == 0:
        st.error("年月データが存在しません。`FC実施年月日` の形式を確認してください。")
        return df, None

    start_month, end_month = st.sidebar.select_slider(
        "表示期間（年月）",
        options=month_list,
        value=(month_list[0], month_list[-1])
    )

    month_mask = (df["年月"] >= start_month) & (df["年月"] <= end_month)
    df = df.loc[month_mask].copy()

    genders = df["性別"].dropna().unique().tolist()
    if len(genders) > 0:
        selected_genders = st.sidebar.multiselect("性別", options=genders, default=genders)
        if selected_genders:
            df = df[df["性別"].isin(selected_genders)]

    ages = df["年代"].dropna().unique().tolist()
    if len(ages) > 0:
        selected_ages = st.sidebar.multiselect("年代", options=ages, default=ages)
        if selected_ages:
            df = df[df["年代"].isin(selected_ages)]

    countries = df["在住国"].dropna().unique().tolist()
    if len(countries) > 0:
        selected_countries = st.sidebar.multiselect("在住国", options=countries, default=countries)
        if selected_countries:
            df = df[df["在住国"].isin(selected_countries)]

    cefrs = df["CEFR"].dropna().unique().tolist()
    if len(cefrs) > 0:
        selected_cefrs = st.sidebar.multiselect("CEFR", options=cefrs, default=cefrs)
        if selected_cefrs:
            df = df[df["CEFR"].isin(selected_cefrs)]

    channel_axis = st.sidebar.radio(
        "チャネル軸の選択",
        options=["集客経路", "流入経路", "識別用のラベル"],
        index=0,
        help="標準は『集客経路』。必要に応じて他の軸でも分析できます。"
    )

    filters = {
        "start_month": start_month,
        "end_month": end_month,
        "channel_axis": channel_axis,
    }

    return df, filters


def aggregate_channel_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    base = (
        df.groupby(col)
        .agg(
            FC件数=("ステータス", "size"),
            入会件数=("入会フラグ", "sum"),
        )
        .reset_index()
    )
    base["入会率(%)"] = np.where(
        base["FC件数"] > 0,
        np.round(base["入会件数"] / base["FC件数"] * 100, 2),
        np.nan
    )
    return base


def aggregate_cefr_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("CEFR")
        .agg(
            FC件数=("ステータス", "size"),
            入会件数=("入会フラグ", "sum"),
        )
        .reset_index()
    )
    agg["入会率(%)"] = np.where(
        agg["FC件数"] > 0,
        np.round(agg["入会件数"] / agg["FC件数"] * 100, 2),
        np.nan
    )
    return agg


def monthly_composition(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    base = (
        df.groupby(["年月", group_col])
        .size()
        .reset_index(name="件数")
    )
    total = (
        df.groupby("年月")
        .size()
        .reset_index(name="月合計")
    )
    merged = base.merge(total, on="年月", how="left")
    merged["比率"] = np.where(merged["月合計"] > 0, merged["件数"] / merged["月合計"], np.nan)
    return merged


def monthly_composition_for_members(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df_in = df[df["入会フラグ"] == 1].copy()
    if df_in.empty:
        return pd.DataFrame(columns=["年月", group_col, "件数", "比率"])

    base = (
        df_in.groupby(["年月", group_col])
        .size()
        .reset_index(name="件数")
    )
    total = (
        df_in.groupby("年月")
        .size()
        .reset_index(name="月入会合計")
    )
    merged = base.merge(total, on="年月", how="left")
    merged["比率"] = np.where(merged["月入会合計"] > 0, merged["件数"] / merged["月入会合計"], np.nan)
    return merged


def format_crosstab_with_ratio(ct: pd.DataFrame) -> pd.DataFrame:
    """クロス集計表の各セルを 件数(比率%) 形式にする。margins 含む。比率の分母は全体合計。"""
    grand = ct.loc["合計", "合計"]
    out = ct.copy().astype(object)
    if grand == 0:
        for i in ct.index:
            for c in ct.columns:
                out.loc[i, c] = f"{int(ct.loc[i, c])}(0.00%)"
        return out

    for i in ct.index:
        for c in ct.columns:
            v = ct.loc[i, c]
            pct = v / grand * 100
            out.loc[i, c] = f"{int(v)}({pct:.2f}%)"
    return out


def make_dist(df: pd.DataFrame, col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """属性の分布を、円グラフ用（合計なし）と表用（合計あり）で返す。件数(比率)は小数第2位まで。"""
    s = df[col]
    if col == "年代":
        vc = s.value_counts(dropna=False, sort=False)
    else:
        vc = s.value_counts(dropna=False)

    dist = vc.reset_index()
    dist.columns = [label, "件数"]
    total = int(dist["件数"].sum())

    if total > 0:
        dist["比率"] = dist["件数"] / total
        dist["件数(比率)"] = dist.apply(lambda r: f"{int(r['件数'])}({r['比率']*100:.2f}%)", axis=1)
        total_row = pd.DataFrame([[
