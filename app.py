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

    # 入会フラグ・二値ステータス
    df["入会フラグ"] = np.where(df["ステータス"] == "入会", 1, 0)
    df["入会ステータス"] = np.where(df["入会フラグ"] == 1, "入会", "非入会")

    # 年代の並び順
    age_order = ["10代前半", "18〜25", "26〜30", "31〜35", "36〜45", "46〜60", "61以上", "不明"]
    df["年代"] = pd.Categorical(df["年代"], categories=age_order, ordered=True)

    return df


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    st.sidebar.header("フィルタ")

    # 年月フィルタ
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

    # 性別フィルタ
    genders = df["性別"].dropna().unique().tolist()
    if len(genders) > 0:
        selected_genders = st.sidebar.multiselect(
            "性別",
            options=genders,
            default=genders
        )
        if selected_genders:
            df = df[df["性別"].isin(selected_genders)]

    # 年代フィルタ
    ages = df["年代"].dropna().unique().tolist()
    if len(ages) > 0:
        selected_ages = st.sidebar.multiselect(
            "年代",
            options=ages,
            default=ages
        )
        if selected_ages:
            df = df[df["年代"].isin(selected_ages)]

    # 在住国フィルタ
    countries = df["在住国"].dropna().unique().tolist()
    if len(countries) > 0:
        selected_countries = st.sidebar.multiselect(
            "在住国",
            options=countries,
            default=countries
        )
        if selected_countries:
            df = df[df["在住国"].isin(selected_countries)]

    # CEFR フィルタ
    cefrs = df["CEFR"].dropna().unique().tolist()
    if len(cefrs) > 0:
        selected_cefrs = st.sidebar.multiselect(
            "CEFR",
            options=cefrs,
            default=cefrs
        )
        if selected_cefrs:
            df = df[df["CEFR"].isin(selected_cefrs)]

    # チャネル軸（メインは集客経路）
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
    """チャネル別の FC件数・入会件数・入会率(%) を集計"""
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
    """CEFR別の FC件数・入会件数・入会率(%) を集計"""
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
    """
    月別 × 任意カテゴリ（性別・年代・在住国・チャネル・CEFRなど）の
    件数と構成比（FC件数ベース）を算出。
    """
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
    merged["比率"] = np.where(
        merged["月合計"] > 0,
        merged["件数"] / merged["月合計"],
        np.nan
    )
    return merged


def monthly_composition_for_members(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    月別 × 任意カテゴリの「入会者」構成比
    （分母は月ごとの入会件数）を算出。
    """
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
    merged["比率"] = np.where(
        merged["月入会合計"] > 0,
        merged["件数"] / merged["月入会合計"],
        np.nan
    )
    return merged


def main():
    st.title("属性別流入ダッシュボード")

    st.markdown(
        "月別の推移・属性別の構成・チャネル別・CEFR別の流入数、入会率を把握するためのダッシュボードです。"
    )

    # データ読み込み
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

    tab_summary, tab_segment, tab_cefr = st.tabs(["サマリー", "流入像（属性・チャネル）", "CEFR分析"])

    # ===== サマリータブ =====
    with tab_summary:
        st.subheader("月別 FC件数")

        monthly_fc = (
            df_filtered
            .groupby("年月")
            .size()
            .reset_index(name="件数")
        )

        if not monthly_fc.empty:
            fc_chart = (
                alt.Chart(monthly_fc)
                .mark_line(point=True)
                .encode(
                    x=alt.X("年月:N", sort=sorted(monthly_fc["年月"].unique()), title="年月"),
                    y=alt.Y("件数:Q", title="月別 FC件数"),
                    tooltip=[
                        alt.Tooltip("年月:N", title="年月"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(fc_chart, use_container_width=True)
        else:
            st.info("表示可能なデータがありません。")

        st.markdown("---")
        st.subheader("属性構成（参考）")

        # 性別構成：表 + 円グラフ
        st.caption("性別構成")
        col_t1, col_p1 = st.columns(2)

        with col_t1:
            gender_dist = (
                df_filtered["性別"]
                .value_counts(dropna=False)
                .reset_index()
            )
            gender_dist.columns = ["性別", "件数"]
            total = gender_dist["件数"].sum()
            if total > 0:
                gender_dist["比率"] = gender_dist["件数"] / total
                gender_dist["件数(比率)"] = gender_dist["件数"].astype(int).astype(str) + "(" + (
                    (gender_dist["比率"] * 100).round(0).astype(int).astype(str) + "%)"
                )
            else:
                gender_dist["比率"] = 0.0
                gender_dist["件数(比率)"] = "0(0%)"
            st.dataframe(gender_dist[["性別", "件数(比率)"]])

        with col_p1:
            if total > 0:
                pie_gender = (
                    alt.Chart(gender_dist)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("件数:Q", title="件数"),
                        color=alt.Color("性別:N", title="性別"),
                        tooltip=[
                            alt.Tooltip("性別:N", title="性別"),
                            alt.Tooltip("件数:Q", title="件数", format=",d"),
                            alt.Tooltip("比率:Q", title="比率", format=".1%"),
                        ],
                    )
                    .properties(width=260, height=260)
                )
                st.altair_chart(pie_gender, use_container_width=True)

        # 年代構成：表 + 円グラフ
        st.caption("年代構成")
        col_t2, col_p2 = st.columns(2)

        with col_t2:
            age_dist = (
                df_filtered["年代"]
                .value_counts(dropna=False)
                .reset_index()
            )
            age_dist.columns = ["年代", "件数"]
            total_age = age_dist["件数"].sum()
            if total_age > 0:
                age_dist["比率"] = age_dist["件数"] / total_age
                age_dist["件数(比率)"] = age_dist["件数"].astype(int).astype(str) + "(" + (
                    (age_dist["比率"] * 100).round(0).astype(int).astype(str) + "%)"
                )
            else:
                age_dist["比率"] = 0.0
                age_dist["件数(比率)"] = "0(0%)"
            st.dataframe(age_dist[["年代", "件数(比率)"]])

        with col_p2:
            if total_age > 0:
                pie_age = (
                    alt.Chart(age_dist)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("件数:Q", title="件数"),
                        color=alt.Color("年代:N", title="年代"),
                        tooltip=[
                            alt.Tooltip("年代:N", title="年代"),
                            alt.Tooltip("件数:Q", title="件数", format=",d"),
                            alt.Tooltip("比率:Q", title="比率", format=".1%"),
                        ],
                    )
                    .properties(width=260, height=260)
                )
                st.altair_chart(pie_age, use_container_width=True)

        # CEFR構成：表 + 円グラフ
        st.caption("CEFR構成")
        col_t3, col_p3 = st.columns(2)

        with col_t3:
            cefr_dist = (
                df_filtered["CEFR"]
                .value_counts(dropna=False)
                .reset_index()
            )
            cefr_dist.columns = ["CEFR", "件数"]
            total_cefr = cefr_dist["件数"].sum()
            if total_cefr > 0:
                cefr_dist["比率"] = cefr_dist["件数"] / total_cefr
                cefr_dist["件数(比率)"] = cefr_dist["件数"].astype(int).astype(str) + "(" + (
                    (cefr_dist["比率"] * 100).round(0).astype(int).astype(str) + "%)"
                )
            else:
                cefr_dist["比率"] = 0.0
                cefr_dist["件数(比率)"] = "0(0%)"
            st.dataframe(cefr_dist[["CEFR", "件数(比率)"]])

        with col_p3:
            if total_cefr > 0:
                pie_cefr = (
                    alt.Chart(cefr_dist)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("件数:Q", title="件数"),
                        color=alt.Color("CEFR:N", title="CEFR"),
                        tooltip=[
                            alt.Tooltip("CEFR:N", title="CEFR"),
                            alt.Tooltip("件数:Q", title="件数", format=",d"),
                            alt.Tooltip("比率:Q", title="比率", format=".1%"),
                        ],
                    )
                    .properties(width=260, height=260)
                )
                st.altair_chart(pie_cefr, use_container_width=True)

        # 在住国構成：表 + 円グラフ
        st.caption("在住国構成")
        col_t4, col_p4 = st.columns(2)

        with col_t4:
            country_dist = (
                df_filtered["在住国"]
                .value_counts(dropna=False)
                .reset_index()
            )
            country_dist.columns = ["在住国", "件数"]
            total_c = country_dist["件数"].sum()
            if total_c > 0:
                country_dist["比率"] = country_dist["件数"] / total_c
                country_dist["件数(比率)"] = country_dist["件数"].astype(int).astype(str) + "(" + (
                    (country_dist["比率"] * 100).round(0).astype(int).astype(str) + "%)"
                )
            else:
                country_dist["比率"] = 0.0
                country_dist["件数(比率)"] = "0(0%)"
            st.dataframe(country_dist[["在住国", "件数(比率)"]])

        with col_p4:
            if total_c > 0:
                pie_country = (
                    alt.Chart(country_dist)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("件数:Q", title="件数"),
                        color=alt.Color("在住国:N", title="在住国"),
                        tooltip=[
                            alt.Tooltip("在住国:N", title="在住国"),
                            alt.Tooltip("件数:Q", title="件数", format=",d"),
                            alt.Tooltip("比率:Q", title="比率", format=".1%"),
                        ],
                    )
                    .properties(width=260, height=260)
                )
                st.altair_chart(pie_country, use_container_width=True)

        st.markdown("---")
        st.subheader("属性クロス集計（件数）")

        # 配置順を指定どおりに変更
        st.caption("性別 × 年代（件数）")
        ct_gender_age = pd.crosstab(df_filtered["性別"], df_filtered["年代"]).fillna(0).astype(int)
        st.dataframe(ct_gender_age)

        st.caption("性別 × CEFR（件数）")
        ct_gender_cefr = pd.crosstab(df_filtered["性別"], df_filtered["CEFR"]).fillna(0).astype(int)
        st.dataframe(ct_gender_cefr)

        st.caption("性別 × 在住国（件数）")
        ct_gender_country = pd.crosstab(df_filtered["性別"], df_filtered["在住国"]).fillna(0).astype(int)
        st.dataframe(ct_gender_country)

        st.caption("年代 × CEFR（件数）")
        ct_age_cefr = pd.crosstab(df_filtered["年代"], df_filtered["CEFR"]).fillna(0).astype(int)
        st.dataframe(ct_age_cefr)

        st.caption("年代 × 在住国（件数）")
        ct_age_country = pd.crosstab(df_filtered["年代"], df_filtered["在住国"]).fillna(0).astype(int)
        st.dataframe(ct_age_country)

        st.caption("在住国 × CEFR（件数）")
        ct_country_cefr = pd.crosstab(df_filtered["在住国"], df_filtered["CEFR"]).fillna(0).astype(int)
        st.dataframe(ct_country_cefr)

    # ===== 流入像（属性・チャネル）タブ =====
    with tab_segment:
        st.subheader(f"チャネル別（{filters['channel_axis']}） 入会分析")

        channel_col = filters["channel_axis"]

        # チャネル別サマリー（テーブル表示）
        channel_summary = aggregate_channel_summary(df_filtered, channel_col)
        st.dataframe(channel_summary.sort_values("FC件数", ascending=False), use_container_width=True)

        # 表示形式の切り替え（表の下・グラフの上）
        display_mode = st.radio(
            "表示形式の切り替え",
            options=["絶対数（件数）", "割合（構成比）"],
            horizontal=True
        )

        st.markdown("### 上位チャネル（5件）の月別推移")

        # 上位5チャネルの抽出
        top_channels = (
            df_filtered[channel_col]
            .value_counts()
            .head(5)
            .index
            .tolist()
        )

        if len(top_channels) > 0:
            df_top = df_filtered[df_filtered[channel_col].isin(top_channels)].copy()
            chan_month = monthly_composition(df_top, channel_col)

            if not chan_month.empty:
                y_field = "件数" if display_mode == "絶対数（件数）" else "比率"
                y_title = "件数" if display_mode == "絶対数（件数）" else "構成比"

                chart_chan = (
                    alt.Chart(chan_month)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("年月:N",
                                sort=sorted(chan_month["年月"].unique()),
                                title="年月"),
                        y=alt.Y(
                            f"{y_field}:Q",
                            title=y_title,
                            axis=alt.Axis(format=",.0f" if display_mode == "絶対数（件数）" else ".0%")
                        ),
                        color=alt.Color(f"{channel_col}:N", title=channel_col),
                        tooltip=[
                            alt.Tooltip("年月:N", title="年月"),
                            alt.Tooltip(f"{channel_col}:N", title=channel_col),
                            alt.Tooltip("件数:Q", title="件数", format=",d"),
                            alt.Tooltip("比率:Q", title="構成比", format=".1%"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_chan, use_container_width=True)
            else:
                st.info("チャネル別の月別推移を表示できません。")
        else:
            st.info("チャネルデータが不足しています。")

        st.markdown("---")
        st.subheader("属性別 月別構成比（FC件数ベース）")

        # 性別
        st.caption("性別別 月別推移")
        gm = monthly_composition(df_filtered, "性別")
        if not gm.empty:
            y_field = "件数" if display_mode == "絶対数（件数）" else "比率"
            y_title = "件数" if display_mode == "絶対数（件数）" else "構成比"

            chart_gender = (
                alt.Chart(gm)
                .mark_line(point=True)
                .encode(
                    x=alt.X("年月:N",
                            sort=sorted(gm["年月"].unique()),
                            title="年月"),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        axis=alt.Axis(format=",.0f" if display_mode == "絶対数（件数）" else ".0%")
                    ),
                    color=alt.Color("性別:N", title="性別"),
                    tooltip=[
                        alt.Tooltip("年月:N", title="年月"),
                        alt.Tooltip("性別:N", title="性別"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".1%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_gender, use_container_width=True)

        # 年代
        st.caption("年代別 月別推移")
        am = monthly_composition(df_filtered, "年代")
        if not am.empty:
            y_field = "件数" if display_mode == "絶対数（件数）" else "比率"
            y_title = "件数" if display_mode == "絶対数（件数）" else "構成比"

            chart_age = (
                alt.Chart(am)
                .mark_line(point=True)
                .encode(
                    x=alt.X("年月:N",
                            sort=sorted(am["年月"].unique()),
                            title="年月"),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        axis=alt.Axis(format=",.0f" if display_mode == "絶対数（件数）" else ".0%")
                    ),
                    color=alt.Color("年代:N", title="年代"),
                    tooltip=[
                        alt.Tooltip("年月:N", title="年月"),
                        alt.Tooltip("年代:N", title="年代"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".1%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_age, use_container_width=True)

        # 在住国
        st.caption("在住国別 月別推移")
        cm = monthly_composition(df_filtered, "在住国")
        if not cm.empty:
            y_field = "件数" if display_mode == "絶対数（件数）" else "比率"
            y_title = "件数" if display_mode == "絶対数（件数）" else "構成比"

            chart_country = (
                alt.Chart(cm)
                .mark_line(point=True)
                .encode(
                    x=alt.X("年月:N",
                            sort=sorted(cm["年月"].unique()),
                            title="年月"),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        axis=alt.Axis(format=",.0f" if display_mode == "絶対数（件数）" else ".0%")
                    ),
                    color=alt.Color("在住国:N", title="在住国"),
                    tooltip=[
                        alt.Tooltip("年月:N", title="年月"),
                        alt.Tooltip("在住国:N", title="在住国"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".1%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_country, use_container_width=True)

    # ===== CEFR分析タブ =====
    with tab_cefr:
        st.subheader("CEFR別 入会分析")

        display_mode_cefr = st.radio(
            "表示形式の切り替え（CEFR）",
            options=["絶対数（件数）", "割合（構成比）"],
            horizontal=True
        )

        st.caption("月別 FC件数に対する CEFR 別構成比")
        cefr_month_fc = monthly_composition(df_filtered, "CEFR")

        if not cefr_month_fc.empty:
            y_field = "件数" if display_mode_cefr == "絶対数（件数）" else "比率"
            y_title = "件数" if display_mode_cefr == "絶対数（件数）" else "構成比"

            chart_cefr_fc = (
                alt.Chart(cefr_month_fc)
                .mark_line(point=True)
                .encode(
                    x=alt.X("年月:N",
                            sort=sorted(cefr_month_fc["年月"].unique()),
                            title="年月"),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        axis=alt.Axis(format=",.0f" if display_mode_cefr == "絶対数（件数）" else ".0%")
                    ),
                    color=alt.Color("CEFR:N", title="CEFR"),
                    tooltip=[
                        alt.Tooltip("年月:N", title="年月"),
                        alt.Tooltip("CEFR:N", title="CEFR"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".1%"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(chart_cefr_fc, use_container_width=True)

        st.caption("月別 入会件数に対する CEFR 別構成比（入会者ベース）")
        cefr_month_member = monthly_composition_for_members(df_filtered, "CEFR")

        if not cefr_month_member.empty:
            y_field = "件数" if display_mode_cefr == "絶対数（件数）" else "比率"
            y_title = "件数" if display_mode_cefr == "絶対数（件数）" else "構成比"

            chart_cefr_member = (
                alt.Chart(cefr_month_member)
                .mark_line(point=True)
                .encode(
                    x=alt.X("年月:N",
                            sort=sorted(cefr_month_member["年月"].unique()),
                            title="年月"),
                    y=alt.Y(
                        f"{y_field}:Q",
                        title=y_title,
                        axis=alt.Axis(format=",.0f" if display_mode_cefr == "絶対数（件数）" else ".0%")
                    ),
                    color=alt.Color("CEFR:N", title="CEFR"),
                    tooltip=[
                        alt.Tooltip("年月:N", title="年月"),
                        alt.Tooltip("CEFR:N", title="CEFR"),
                        alt.Tooltip("件数:Q", title="件数", format=",d"),
                        alt.Tooltip("比率:Q", title="構成比", format=".1%"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(chart_cefr_member, use_container_width=True)

        st.markdown("---")
        st.subheader("CEFR別 サマリー（流入数・入会率）")

        cefr_summary = aggregate_cefr_summary(df_filtered)
        st.dataframe(cefr_summary.sort_values("FC件数", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
