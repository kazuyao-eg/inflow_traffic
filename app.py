import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from typing import Optional, Tuple, Dict


st.set_page_config(
    page_title="属性別流入ダッシュボード",
    layout="wide"
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["FC実施年月日"] = pd.to_datetime(df["FC実施年月日"], errors="coerce")
    df["年月"] = df["FC実施年月日"].dt.to_period("M").astype(str)
    df["年"] = df["FC実施年月日"].dt.year.astype(str)
    # 半年ごと: 1-6月→1-6月, 7-12月→7-12月（例: 2024-1-6月, 2024-7-12月）
    df["半年"] = (
        df["FC実施年月日"].dt.year.astype(str)
        + "-"
        + np.where(df["FC実施年月日"].dt.month <= 6, "1-6月", "7-12月")
    )

    df["入会フラグ"] = np.where(df["ステータス"] == "入会", 1, 0)
    df["入会ステータス"] = np.where(df["入会フラグ"] == 1, "入会", "非入会")

    age_order = ["10代前半", "18〜25", "26〜30", "31〜35", "36〜45", "46〜60", "61以上", "不明"]
    df["年代"] = pd.Categorical(df["年代"], categories=age_order, ordered=True)

    return df


def apply_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict]]:
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
        value=(month_list[0], month_list[-1]),
        key="sidebar_period"
    )

    df = df.loc[(df["年月"] >= start_month) & (df["年月"] <= end_month)].copy()

    genders = df["性別"].dropna().unique().tolist()
    if len(genders) > 0:
        selected_genders = st.sidebar.multiselect(
            "性別",
            options=genders,
            default=genders,
            key="sidebar_gender"
        )
        if selected_genders:
            df = df[df["性別"].isin(selected_genders)]

    ages = df["年代"].dropna().unique().tolist()
    if len(ages) > 0:
        selected_ages = st.sidebar.multiselect(
            "年代",
            options=ages,
            default=ages,
            key="sidebar_age"
        )
        if selected_ages:
            df = df[df["年代"].isin(selected_ages)]

    countries = df["在住国"].dropna().unique().tolist()
    if len(countries) > 0:
        selected_countries = st.sidebar.multiselect(
            "在住国",
            options=countries,
            default=countries,
            key="sidebar_country"
        )
        if selected_countries:
            df = df[df["在住国"].isin(selected_countries)]

    cefrs = df["CEFR"].dropna().unique().tolist()
    if len(cefrs) > 0:
        selected_cefrs = st.sidebar.multiselect(
            "CEFR",
            options=cefrs,
            default=cefrs,
            key="sidebar_cefr"
        )
        if selected_cefrs:
            df = df[df["CEFR"].isin(selected_cefrs)]

    channel_axis = st.sidebar.radio(
        "チャネル軸の選択",
        options=["集客経路", "流入経路", "識別用のラベル"],
        index=0,
        help="標準は『集客経路』。必要に応じて他の軸でも分析できます。",
        key="sidebar_channel_axis"
    )

    return df, {
        "start_month": start_month,
        "end_month": end_month,
        "channel_axis": channel_axis,
    }


def aggregate_channel_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    base = (
        df.groupby(col)
        .agg(
            FC件数=("ステータス", "size"),
            入会件数=("入会フラグ", "sum"),
        )
        .reset_index()
    )

    # 入会率(%)を小数第2位固定 + % を必ず付ける
    base["入会率(%)"] = np.where(
        base["FC件数"] > 0,
        np.round(base["入会件数"] / base["FC件数"] * 100, 2),
        np.nan
    )
    base["入会率(%)"] = base["入会率(%)"].apply(
        lambda v: f"{v:.2f}%" if pd.notna(v) else "—"
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

    # 入会率(%)を小数第2位固定 + % を必ず付ける
    agg["入会率(%)"] = np.where(
        agg["FC件数"] > 0,
        np.round(agg["入会件数"] / agg["FC件数"] * 100, 2),
        np.nan
    )
    agg["入会率(%)"] = agg["入会率(%)"].apply(
        lambda v: f"{v:.2f}%" if pd.notna(v) else "—"
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


def time_composition(
    df: pd.DataFrame,
    time_col: str,
    group_col: str,
    member_base: bool
) -> pd.DataFrame:
    if member_base:
        df2 = df[df["入会フラグ"] == 1].copy()
    else:
        df2 = df.copy()

    if df2.empty:
        return pd.DataFrame(columns=[time_col, group_col, "件数", "比率", "市場寄与度"])

    base = (
        df2.groupby([time_col, group_col])
        .size()
        .reset_index(name="件数")
    )
    total = (
        df2.groupby(time_col)
        .size()
        .reset_index(name="合計")
    )
    merged = base.merge(total, on=time_col, how="left")
    merged["比率"] = np.where(merged["合計"] > 0, merged["件数"] / merged["合計"], np.nan)

    # 市場寄与度の計算（セグメントの増減 / 全体の増減）
    merged = merged.sort_values([time_col, group_col]).copy()

    total_series = merged.groupby(time_col)["件数"].sum().sort_index()
    total_delta = total_series.diff()
    merged = merged.merge(
        total_delta.rename("全体増減"),
        on=time_col,
        how="left"
    )

    merged["増減"] = (
        merged.sort_values([group_col, time_col])
        .groupby(group_col)["件数"]
        .diff()
    )

    merged["市場寄与度"] = np.where(
        (merged["全体増減"] != 0) & merged["全体増減"].notna(),
        merged["増減"] / merged["全体増減"],
        np.nan
    )

    return merged


def format_crosstab_with_ratio(ct: pd.DataFrame) -> pd.DataFrame:
    ct2 = ct.fillna(0).astype(int)
    grand = int(ct2.loc["合計", "合計"])

    out = ct2.copy().astype(object)
    if grand == 0:
        for i in ct2.index:
            for c in ct2.columns:
                out.loc[i, c] = f"{int(ct2.loc[i, c])}(0.00%)"
        return out

    for i in ct2.index:
        for c in ct2.columns:
            v = int(ct2.loc[i, c])
            pct = v / grand * 100
            out.loc[i, c] = f"{v}({pct:.2f}%)"
    return out


def make_dist(df: pd.DataFrame, col: str, label: str):
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
        dist["件数(比率)"] = dist.apply(
            lambda r: f"{int(r['件数'])}({r['比率'] * 100:.2f}%)",
            axis=1
        )
        total_row = pd.DataFrame(
            [["合計", total, 1.0, f"{total}(100.00%)"]],
            columns=[label, "件数", "比率", "件数(比率)"],
        )
    else:
        dist["比率"] = 0.0
        dist["件数(比率)"] = dist["件数"].apply(lambda x: f"{int(x)}(0.00%)")
        total_row = pd.DataFrame(
            [["合計", 0, 0.0, "0(0.00%)"]],
            columns=[label, "件数", "比率", "件数(比率)"],
        )

    table = pd.concat([dist[[label, "件数", "比率", "件数(比率)"]], total_row], ignore_index=True)
    pie = dist[[label, "件数", "比率"]].copy()
    return pie, table[[label, "件数(比率)"]], total


def _time_col_from_mode(mode: str) -> str:
    if mode == "年月":
        return "年月"
    if mode == "年別":
        return "年"
    return "半年"  # 半年ごと


def _time_label_from_mode(mode: str) -> str:
    if mode == "年月":
        return "月別"
    if mode == "年別":
        return "年別"
    return "半年ごと"


def render_summary_tab(df: pd.DataFrame, base_label: str, time_col: str = "年月") -> None:
    time_labels = {"年月": "月別", "年": "年別", "半年": "半年ごと"}
    time_label = time_labels.get(time_col, "月別")

    st.subheader(f"{time_label} {base_label}件数")

    period_cnt = (
        df.groupby(time_col)
        .size()
        .reset_index(name="件数")
    )

    if not period_cnt.empty:
        sort_vals = sorted(period_cnt[time_col].unique().tolist())
        chart = (
            alt.Chart(period_cnt)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    f"{time_col}:N",
                    sort=sort_vals,
                    title=time_col
                ),
                y=alt.Y("件数:Q", title=f"{time_label} {base_label}件数"),
                tooltip=[
                    alt.Tooltip(f"{time_col}:N", title=time_col),
                    alt.Tooltip("件数:Q", title="件数", format=",d"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("現在のフィルタ条件ではデータがありません。")

    st.markdown("---")
    st.subheader("属性構成（参考）")

    # 性別
    st.caption("性別構成")
    col_t1, col_p1 = st.columns(2)
    pie_gender, table_gender, total_gender = make_dist(df, "性別", "性別")
    with col_t1:
        st.dataframe(table_gender, use_container_width=True)
    with col_p1:
        if total_gender > 0:
            pie = (
                alt.Chart(pie_gender)
                .mark_arc()
                .encode(
                    theta=alt.Theta("件数:Q"),
                    color=alt.Color("性別:N", scale=alt.Scale(scheme="blues"), title="性別"),
                    tooltip=[
                        alt.Tooltip("性別:N", title="性別"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="比率", format=".2%"),
                    ],
                )
                .properties(width=260, height=260)
            )
            st.altair_chart(pie, use_container_width=True)

    # 年代
    st.caption("年代構成")
    col_t2, col_p2 = st.columns(2)
    pie_age, table_age, total_age = make_dist(df, "年代", "年代")
    with col_t2:
        st.dataframe(table_age, use_container_width=True)
    with col_p2:
        if total_age > 0:
            pie = (
                alt.Chart(pie_age)
                .mark_arc()
                .encode(
                    theta=alt.Theta("件数:Q"),
                    color=alt.Color("年代:N", scale=alt.Scale(scheme="blues"), title="年代"),
                    tooltip=[
                        alt.Tooltip("年代:N", title="年代"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="比率", format=".2%"),
                    ],
                )
                .properties(width=260, height=260)
            )
            st.altair_chart(pie, use_container_width=True)

    # CEFR
    st.caption("CEFR構成")
    col_t3, col_p3 = st.columns(2)
    pie_cefr, table_cefr, total_cefr = make_dist(df, "CEFR", "CEFR")
    with col_t3:
        st.dataframe(table_cefr, use_container_width=True)
    with col_p3:
        if total_cefr > 0:
            pie = (
                alt.Chart(pie_cefr)
                .mark_arc()
                .encode(
                    theta=alt.Theta("件数:Q"),
                    color=alt.Color("CEFR:N", scale=alt.Scale(scheme="blues"), title="CEFR"),
                    tooltip=[
                        alt.Tooltip("CEFR:N", title="CEFR"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="比率", format=".2%"),
                    ],
                )
                .properties(width=260, height=260)
            )
            st.altair_chart(pie, use_container_width=True)

    # 在住国
    st.caption("在住国構成")
    col_t4, col_p4 = st.columns(2)
    pie_country, table_country, total_country = make_dist(df, "在住国", "在住国")
    with col_t4:
        st.dataframe(table_country, use_container_width=True)
    with col_p4:
        if total_country > 0:
            pie = (
                alt.Chart(pie_country)
                .mark_arc()
                .encode(
                    theta=alt.Theta("件数:Q"),
                    color=alt.Color("在住国:N", scale=alt.Scale(scheme="blues"), title="在住国"),
                    tooltip=[
                        alt.Tooltip("在住国:N", title="在住国"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="比率", format=".2%"),
                    ],
                )
                .properties(width=260, height=260)
            )
            st.altair_chart(pie, use_container_width=True)

    st.markdown("---")
    st.subheader("属性クロス集計（件数）")

    def show_ct(title: str, idx: str, col: str) -> None:
        st.caption(title)
        ct = pd.crosstab(
            df[idx], df[col],
            margins=True, margins_name="合計"
        )
        st.dataframe(format_crosstab_with_ratio(ct), use_container_width=True)

    show_ct("性別 × 年代（件数）", "性別", "年代")
    show_ct("性別 × CEFR（件数）", "性別", "CEFR")
    show_ct("性別 × 在住国（件数）", "性別", "在住国")
    show_ct("年代 × CEFR（件数）", "年代", "CEFR")
    show_ct("年代 × 在住国（件数）", "年代", "在住国")
    show_ct("在住国 × CEFR（件数）", "在住国", "CEFR")


def main():
    st.title("属性別流入ダッシュボード")
    st.markdown("月別の推移・属性別の構成・チャネル別・CEFR別の流入数、入会率を把握するためのダッシュボードです。")

    try:
        df_raw = load_data("fc_info.csv")
    except FileNotFoundError:
        st.error("`fc_info.csv` が見つかりません。`app.py` と同じフォルダに配置してください。")
        return

    df_filtered, filters = apply_filters(df_raw)
    if filters is None:
        return

    if df_filtered.empty:
        st.warning("現在のフィルタ条件ではデータがありません。条件を緩めてみてください。")
        return

    tab_summary, tab_attr, tab_channel, tab_cefr = st.tabs(
        ["サマリー", "流入像（属性）", "流入像（チャネル）", "CEFR分析"]
    )

    # ===== サマリー =====
    with tab_summary:
        base_mode = st.radio(
            "表示形式の切り替え①（ベース）",
            options=["FC件数ベース", "入会件数ベース"],
            horizontal=True,
            key="summary_base_mode"
        )
        time_mode_summary = st.radio(
            "表示形式の切り替え②（時間軸）",
            options=["年月", "年別", "半年ごと"],
            horizontal=True,
            key="summary_time_mode"
        )
        time_col_summary = _time_col_from_mode(time_mode_summary)

        if base_mode == "入会件数ベース":
            df_base = df_filtered[df_filtered["入会フラグ"] == 1].copy()
            if df_base.empty:
                st.warning("現在のフィルタ条件では入会データがありません。条件を緩めてください。")
            else:
                render_summary_tab(df_base, "入会", time_col=time_col_summary)
        else:
            render_summary_tab(df_filtered, "FC", time_col=time_col_summary)

    # ===== 流入像（属性）=====
    with tab_attr:
        st.subheader("流入像（属性）")

        base_mode_attr = st.radio(
            "表示形式の切り替え①（ベース）",
            options=["FC件数ベース", "入会件数ベース"],
            horizontal=True,
            key="attr_base_mode"
        )
        member_base_attr = base_mode_attr == "入会件数ベース"

        time_mode = st.radio(
            "表示形式の切り替え②（時間軸）",
            options=["年月", "年別", "半年ごと"],
            horizontal=True,
            key="attr_time_mode"
        )
        time_col = _time_col_from_mode(time_mode)

        display_mode_attr = st.radio(
            "表示形式の切り替え③（指標）",
            options=["絶対数（件数）", "割合（構成比）", "市場寄与度"],
            horizontal=True,
            key="attr_display_mode"
        )

        def plot_attr_ts(title: str, cols):
            st.caption(title)
            if isinstance(cols, str):
                group_col = cols
                df_work = df_filtered
            else:
                col1, col2 = cols
                group_col = f"{col1}×{col2}"
                df_work = df_filtered.copy()
                df_work[group_col] = df_work[col1].astype(str) + "×" + df_work[col2].astype(str)

            comp = time_composition(df_work, time_col, group_col, member_base_attr)
            if comp.empty:
                st.info("表示できるデータがありません。")
                return

            if display_mode_attr == "絶対数（件数）":
                y_field = "件数"
                y_title = "件数"
                axis_format = ",.0f"
            elif display_mode_attr == "割合（構成比）":
                y_field = "比率"
                y_title = "構成比"
                axis_format = ".2%"
            else:
                y_field = "市場寄与度"
                y_title = "市場寄与度"
                axis_format = ".2%"

            chart = (
                alt.Chart(comp)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        f"{time_col}:N",
                        sort=sorted(comp[time_col].unique().tolist()),
                        title=time_col
                    ),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        axis=alt.Axis(format=axis_format)
                    ),
                    color=alt.Color(f"{group_col}:N", title=group_col),
                    tooltip=[
                        alt.Tooltip(f"{time_col}:N", title=time_col),
                        alt.Tooltip(f"{group_col}:N", title=group_col),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".2%"),
                        alt.Tooltip("市場寄与度:Q", title="市場寄与度", format=".2%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

        plot_attr_ts("性別別 推移", "性別")
        plot_attr_ts("年代別 推移", "年代")
        plot_attr_ts("在住国別 推移", "在住国")
        plot_attr_ts("性別 × 年代別 推移", ("性別", "年代"))
        plot_attr_ts("性別 × CEFR 推移", ("性別", "CEFR"))
        plot_attr_ts("性別 × 在住国 推移", ("性別", "在住国"))
        plot_attr_ts("年代 × CEFR 推移", ("年代", "CEFR"))
        plot_attr_ts("年代 × 在住国 推移", ("年代", "在住国"))
        plot_attr_ts("在住国 × CEFR 推移", ("在住国", "CEFR"))

    # ===== 流入像（チャネル）=====
    with tab_channel:
        st.subheader(f"チャネル別（{filters['channel_axis']}） 入会分析")

        channel_col = filters["channel_axis"]
        channel_summary = aggregate_channel_summary(df_filtered, channel_col)
        st.dataframe(channel_summary.sort_values("FC件数", ascending=False), use_container_width=True)

        # --- 上位チャネル（5件）の月別推移（動的化）---
        channel_base_mode = st.radio(
            "表示形式の切り替え①（ベース）",
            options=["FC件数ベース", "入会件数ベース"],
            horizontal=True,
            key="channel_base_mode"
        )
        member_base_channel = channel_base_mode == "入会件数ベース"

        channel_time_mode = st.radio(
            "表示形式の切り替え②（時間軸）",
            options=["年月", "年別", "半年ごと"],
            horizontal=True,
            key="channel_time_mode"
        )
        time_col_channel = _time_col_from_mode(channel_time_mode)

        display_mode = st.radio(
            "表示形式の切り替え③（指標）",
            options=["絶対数（件数）", "割合（構成比）"],
            horizontal=True,
            key="channel_indicator_mode"
        )

        st.markdown(f"### 上位チャネル（最大10件）の{_time_label_from_mode(channel_time_mode)}推移")

        # 上位5チャネル（ベースに応じて決定）
        if member_base_channel:
            df_for_top = df_filtered[df_filtered["入会フラグ"] == 1].copy()
        else:
            df_for_top = df_filtered

        if df_for_top.empty:
            st.info("選択条件ではデータがありません。")
        else:
            top_channels = (
                df_for_top[channel_col]
                .value_counts()
                .head(10)
                .index
                .tolist()
            )

            df_top = df_for_top[df_for_top[channel_col].isin(top_channels)].copy()

            comp = time_composition(
                df_top,
                time_col_channel,
                channel_col,
                member_base_channel
            )

            if comp.empty:
                st.info("上位チャネルの推移を表示できません。")
            else:
                if display_mode == "絶対数（件数）":
                    y_field = "件数"
                    axis_fmt = ",.0f"
                    y_title = "件数"
                else:
                    y_field = "比率"
                    axis_fmt = ".2%"
                    y_title = "構成比"

                chart_chan = (
                    alt.Chart(comp)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            f"{time_col_channel}:N",
                            sort=sorted(comp[time_col_channel].unique().tolist()),
                            title=time_col_channel
                        ),
                        y=alt.Y(
                            f"{y_field}:Q",
                            title=y_title,
                            axis=alt.Axis(format=axis_fmt)
                        ),
                        color=alt.Color(f"{channel_col}:N", title=channel_col),
                        tooltip=[
                            alt.Tooltip(f"{time_col_channel}:N", title=time_col_channel),
                            alt.Tooltip(f"{channel_col}:N", title=channel_col),
                            alt.Tooltip("件数:Q", title="件数", format=",d"),
                            alt.Tooltip("比率:Q", title="構成比", format=".2%"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_chan, use_container_width=True)

    # ===== CEFR分析 =====
    with tab_cefr:
        st.subheader("CEFR別 入会分析")

        display_mode_cefr = st.radio(
            "表示形式の切り替え（CEFR）",
            options=["絶対数（件数）", "割合（構成比）"],
            horizontal=True,
            key="cefr_display_mode"
        )
        time_mode_cefr = st.radio(
            "表示形式の切り替え②（時間軸）",
            options=["年月", "年別", "半年ごと"],
            horizontal=True,
            key="cefr_time_mode"
        )
        time_col_cefr = _time_col_from_mode(time_mode_cefr)

        st.caption(f"{_time_label_from_mode(time_mode_cefr)} FC件数に対する CEFR 別構成比")
        cefr_fc = time_composition(df_filtered, time_col_cefr, "CEFR", member_base=False)
        if not cefr_fc.empty:
            y_field = "件数" if display_mode_cefr == "絶対数（件数）" else "比率"
            y_title = "件数" if display_mode_cefr == "絶対数（件数）" else "構成比"
            axis_fmt = ",.0f" if display_mode_cefr == "絶対数（件数）" else ".2%"
            st.altair_chart(
                alt.Chart(cefr_fc)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        f"{time_col_cefr}:N",
                        sort=sorted(cefr_fc[time_col_cefr].unique().tolist()),
                        title=time_col_cefr
                    ),
                    y=alt.Y(f"{y_field}:Q", title=y_title, axis=alt.Axis(format=axis_fmt)),
                    color=alt.Color("CEFR:N", title="CEFR"),
                    tooltip=[
                        alt.Tooltip(f"{time_col_cefr}:N", title=time_col_cefr),
                        alt.Tooltip("CEFR:N", title="CEFR"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".2%"),
                    ],
                )
                .properties(height=280),
                use_container_width=True
            )

        st.caption(f"{_time_label_from_mode(time_mode_cefr)} 入会件数に対する CEFR 別構成比（入会者ベース）")
        cefr_member = time_composition(df_filtered, time_col_cefr, "CEFR", member_base=True)
        if not cefr_member.empty:
            y_field = "件数" if display_mode_cefr == "絶対数（件数）" else "比率"
            y_title = "件数" if display_mode_cefr == "絶対数（件数）" else "構成比"
            axis_fmt = ",.0f" if display_mode_cefr == "絶対数（件数）" else ".2%"
            st.altair_chart(
                alt.Chart(cefr_member)
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        f"{time_col_cefr}:N",
                        sort=sorted(cefr_member[time_col_cefr].unique().tolist()),
                        title=time_col_cefr
                    ),
                    y=alt.Y(f"{y_field}:Q", title=y_title, axis=alt.Axis(format=axis_fmt)),
                    color=alt.Color("CEFR:N", title="CEFR"),
                    tooltip=[
                        alt.Tooltip(f"{time_col_cefr}:N", title=time_col_cefr),
                        alt.Tooltip("CEFR:N", title="CEFR"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".2%"),
                    ],
                )
                .properties(height=280),
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("CEFR別 サマリー（流入数・入会率）")
        st.dataframe(
            aggregate_cefr_summary(df_filtered).sort_values("FC件数", ascending=False),
            use_container_width=True
        )


if __name__ == "__main__":
    main()
