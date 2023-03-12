from itertools import islice, cycle
import os
from typing import Tuple, Union, List

import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import seaborn as sns

import streamlit.components.v1 as components

import holoviews as hv
import hvplot
import hvplot.pandas
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.formatters import DatetimeTickFormatter


# Loading data from local machine
def load_and_process_transactions(
    file: str = None,
    src: str = "",
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Load and process Tiller transactions.

    Parameters
    ----------
    file : str, optional
        _description_, by default "tiller_latest_transactions.csv"

    Load Tiller transaction data and preprocess the data.
    The preprocessing includes a few cleaning steps:

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # load data
    if df is None and file is not None:
        # load file from path
        transactions = os.path.join(src, file)
        data = pd.read_csv(transactions, index_col=0, header=0).reset_index(drop=True)
    else:
        # feed in data
        data = df

    # convert some columns to datetime types
    for ci in ["Date", "Month", "Week"]:
        data[ci] = pd.to_datetime(data[ci])

    # convert string amount to numeric
    amount = data["Amount"].str.replace("$", "", regex=True)
    amount = amount.str.replace(",", "", regex=True)
    data["Amount"] = amount.astype(float)

    # sort data by date column in reverse (soonest on top)
    data = data.sort_values("Date", ascending=False)
    # reset index
    data = data.reset_index(drop=True)

    # filter the data for a few notable columns
    df = data[
        [
            "Date",
            "Category",
            "Group",
            "Type",
            "Amount",
            "Institution",
        ]
    ]
    # add month day year columns for easy grouping
    df = df.assign(
        Year=df.Date.dt.to_period("Y").dt.to_timestamp(),
        Month=df.Date.dt.to_period("M").dt.to_timestamp(),
        Day=df.Date.dt.to_period("D").dt.to_timestamp(),
        Week=df.Date.dt.to_period("W").dt.to_timestamp(),
    )

    return df


def time_series_pivot_table(
    df: pd.DataFrame,
    t: str = "Month",
    y: str = "Category",
    transaction_type: str = "Income",
    last_n_months: int = 12,
    melt: bool = True,
) -> pd.DataFrame:
    """# Create Time by Category type.

    Args:
        df (pd.DataFrame): _description_
        t (str, optional): _description_. Defaults to "Month".
        y (str, optional): _description_. Defaults to "Category".
        type (str, optional): _description_. Defaults to "Income".

    Returns:
        pd.DataFrame: _description_
    """
    if last_n_months is not None:
        # filter most recent
        beginning_month = df.Date[0] - pd.offsets.MonthBegin(last_n_months + 1)
        df_ = df[df["Date"] >= beginning_month]
        df_.shape
    else:
        df_ = df

    # in order to compare both on the same positive y axis
    if transaction_type == "Income":
        factor = 1.0
    elif transaction_type == "Expense":
        factor = -1.0
    df_type = (
        df_.query(f"Type == '{transaction_type}'")
        .groupby(by=[t, y])
        .agg({"Amount": lambda x: factor * np.sum(x)})
    )
    pt_ = pd.pivot_table(
        df_type,
        values="Amount",
        index=t,
        columns=y,
    )

    if melt:
        pt_melt_ = pd.melt(
            pt_.reset_index(),
            id_vars=t,
        ).reset_index(drop=True)
        pt_melt_ = pt_melt_.rename(columns={"value": "Amount"})
        return pt_melt_

    return pt_


def transaction_viewer(
    df,
    t="Month",
    f="Type",
    transaction_type="Expense",
    n_months=12,
    color=None,
    plot=False,
    show_top_n=8,
):
    """_summary_

    Args:
        df (_type_): _description_
        t (str, optional): _description_. Defaults to "Month".
        f (str, optional): _description_. Defaults to "Type".
        transaction_type (str, optional): _description_. Defaults to "Expense".
        n_months (int, optional): _description_. Defaults to 12.
        color (_type_, optional): _description_. Defaults to None.
        plot (bool, optional): _description_. Defaults to False.
        show_top_n (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    # convert raw data to pivot table
    pt_ = time_series_pivot_table(
        df,
        t=t,
        y=f,
        transaction_type=transaction_type,
        last_n_months=n_months,
        melt=True,
    )
    pt_["Type"] = transaction_type

    # top 12 categories
    # df_expenses = df_recent.query("Type == 'Expense'")
    cat_values = pt_.groupby(f).agg({"Amount": "sum"})
    n_top = min(show_top_n, len(cat_values))
    top_n_cat = list(
        cat_values.sort_values("Amount", ascending=False)[:n_top].index,
    )
    pt_top_ = pt_.loc[pt_[f].isin(top_n_cat)]

    if plot is False:
        return pt_top_

    # grouped plot grouped bar plot
    # columns = pt_top_.columns
    if transaction_type == "Income":
        colors = sns.color_palette("Blues_r")
    elif transaction_type == "Expense":
        colors = sns.color_palette("Reds_r")
    if color is not None:
        colors = sns.color_palette(f"{color}")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        pt_top_,
        x=t,
        y="Amount",
        hue=f,
        palette=colors,
        edgecolor="white",
        ax=ax,
    )
    if t == "Month":
        x_dates = pt_top_[t].dt.strftime("%b %Y").unique()
        ax.set_xticklabels(labels=x_dates)
    elif t == "Week" or "Day":
        x_dates = pt_top_[t].dt.strftime("%b %d %Y").unique()
        ax.set_xticklabels(labels=x_dates)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(which="both", alpha=0.25)
    plt.xticks(rotation=20)
    plt.legend(
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )

    return fig, ax, pt_top_


def comparison_holoviz_view_w_net(
    df,
    t="Month",
    n_months=3,
):
    df_ = comparison_viewer(
        df,
        t,
        f="Type",
        n_months=n_months,
        transaction_type=["Income", "Expense"],
        plot=False,
    )

    df_type = pd.pivot_table(df_, index="Month", columns="Type")
    df_type.columns = df_type.columns.get_level_values(1)
    df_type["Net"] = df_type["Income"] - df_type["Expense"]
    df_net = pd.melt(df_type.reset_index(), id_vars="Month")
    df_net = df_net.groupby(["Month", "Type"]).mean()
    df_net.columns = ["Amount"]

    custom_cmap = {
        "Expense": "#f878a7",
        "Income": "#75c9af",
        "Net -": "#d72549",
        "Net +": "#7ab6d9",
    }

    formatter = NumeralTickFormatter(format="$0,0")

    df_test = df_net.reset_index().sort_values(["Type", "Month"])
    colors = []
    for ii, row in df_test.iterrows():
        if row["Type"] == "Net":
            if row["Amount"] >= 0:
                c = custom_cmap["Net +"]
            else:
                c = custom_cmap["Net -"]
            colors.append(c)
        else:
            colors.append(custom_cmap[row["Type"]])

    df_plot = df_net.copy()
    df_plot["Color"] = colors

    ts = pd.Series(df_plot.index.get_level_values(0).unique()).dt.strftime("%b %Y")
    df_plot.index = df_plot.index.set_levels(ts, level=0)

    hv_plot = df_plot.hvplot.bar(
        x="Month",
        y="Amount",
        by="Type",
        rot=90,
        color="Color",
        grid=True,
        alpha=0.8,
        yformatter=NumeralTickFormatter(format="$0,0"),
    ).opts(width=700)

    return hv_plot


def comparison_holoviz_view(
    df,
    t="Month",
    f="Type",
    n_months=3,
    t_types=["Income", "Expense"],
):
    df_compare = comparison_viewer(df, t, f, n_months=n_months, plot=False)
    df_compare = df_compare.fillna(0)

    df_grouped = df_compare.groupby([t, f, "Type"]).agg("sum")

    if t_types == ["Income"]:
        df_grouped = df_grouped.loc[df_grouped.index.get_level_values(2) == "Income"]
    if t_types == ["Expense"]:
        df_grouped = df_grouped.loc[df_grouped.index.get_level_values(2) == "Expense"]

    # for colors
    df_temp = pd.DataFrame(
        list(
            zip(
                list(df_grouped.index.get_level_values(1)),
                list(df_grouped.index.get_level_values(2)),
            )
        )
    ).drop_duplicates()
    df_temp

    def rgb_to_hex(r, g, b):
        r, g, b = round(r * 255.0), round(g * 255.0), round(b * 255.0)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    colors = []
    i_colors = islice(cycle(sns.color_palette("BuGn_r", 12)), 6, None)
    e_colors = islice(cycle(sns.color_palette("RdPu_r", 12)), 6, None)
    for ii, row in df_temp.iterrows():
        if row[1] == "Expense":
            c = rgb_to_hex(*next(e_colors))
        if row[1] == "Income":
            c = rgb_to_hex(*next(i_colors))
        colors.append(c)

    formatter = NumeralTickFormatter(format="$0,0")

    if f == "Type":
        df_grouped = df_grouped.set_index(df_grouped.index.droplevel(2))

    # change date to strings
    df_grouped.index = df_grouped.index.set_levels(
        pd.Series(df_grouped.index.get_level_values(0)).dt.strftime("%b %Y").unique(),
        level=0,
    )

    hv_plot = df_grouped.hvplot.bar(
        x=t,
        y="Amount",
        stacked=False,
        by=f,
        yformatter=formatter,
        rot=90,
        legend=True,
        color=colors,
        grid=True,
    ).opts(width=700)

    if f == "Type" and "Income" in t_types and "Expense" in t_types:
        hv_plot = comparison_holoviz_view_w_net(
            df,
            t,
            n_months=n_months,
        )

    return hv_plot


def comparison_viewer(
    df,
    t="Month",
    f="Type",
    transaction_type=["Income", "Expense"],
    n_months=12,
    colors=None,
    plot=True,
    show_top_n=8,
    log_scale=False,
):
    df_expense = pd.DataFrame(columns=[t, f, "Amount"])
    df_income = pd.DataFrame(columns=[t, f, "Amount"])
    if "Expense" in transaction_type:
        df_expense = transaction_viewer(
            df,
            t,
            f,
            transaction_type="Expense",
            n_months=n_months,
            plot=False,
            show_top_n=show_top_n,
        )
    if "Income" in transaction_type:
        df_income = transaction_viewer(
            df,
            t,
            f,
            transaction_type="Income",
            n_months=n_months,
            plot=False,
            show_top_n=show_top_n,
        )
    df_compare = pd.concat([df_income, df_expense], axis=0)
    df_compare["Amount"] = df_compare["Amount"].apply(np.around)

    if not plot:
        return df_compare

    # create color map for income vs expense
    # c_income = cycle(sns.color_palette("Blues_r"))
    # c_expense = cycle(sns.color_palette("Reds_r"))
    cmap_income = cycle(sns.color_palette("BuGn_r", 12))
    cmap_expense = cycle(sns.color_palette("RdPu_r", 12))
    c_income = islice(cmap_income, 6, None)
    c_expense = islice(cmap_expense, 6, None)

    cmap_income = {}
    for i, k in enumerate(df_income.iloc[:, 1].unique()):
        cmap_income[k] = next(c_income)
    cmap_expense = {}
    for i, k in enumerate(df_expense.iloc[:, 1].unique()):
        cmap_expense[k] = next(c_expense)
    cmap_compare = {**cmap_income, **cmap_expense}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        df_compare,
        x=t,
        y="Amount",
        hue=f,
        palette=cmap_compare,
        ax=ax,
        edgecolor="white",
    )

    from matplotlib.ticker import FormatStrFormatter

    start, end = ax.get_ylim()
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.f"))
    # else:
    #     ax.yaxis.set_ticks(np.arange(start, end, int((start - end) / 5)))
    if t == "Month":
        x_dates = df_compare[t].dt.strftime("%b %Y").unique()
        ax.set_xticklabels(labels=x_dates)
    elif t == "Week" or "Day":
        x_dates = df_compare[t].dt.strftime("%b %d %Y").unique()
        ax.set_xticklabels(labels=x_dates)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("${x:,.2f}"))
    ax.grid(which="both", alpha=0.25)
    plt.xticks(rotation=20)
    plt.legend(
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )

    return fig, ax, df_compare


def monthly_spending_holoviz_view(
    df,
    f="Type",
    n_months=2,
):
    n_months = min(n_months, 3)
    df_ = monthly_spending_trend_viewer(
        df,
        f,
        last_n_months=n_months,
        plot=False,
    )
    df_ = df_.apply(np.around)

    yformat = NumeralTickFormatter(format="$0,0")
    xformat = DatetimeTickFormatter(months="%b %d %Y")

    hv_plot = df_.hvplot(
        xformatter=xformat,
        alpha=[0.8, 0.5, 0.5],
        yformatter=yformat,
        grid=True,
        width=700,
        line_dash=["solid"] + ["dotdash"] * (n_months),
        line_width=[4] + [3] * (n_months),
        ylabel="Expenses",
        legend="top_left",
        group_label="Month",
    )
    return hv_plot


def monthly_spending_trend_viewer(
    df,
    f="Type",
    last_n_months=2,
    plot=True,
):
    time = "Day"
    df_expense = transaction_viewer(
        df,
        time,
        f,
        transaction_type="Expense",
        n_months=last_n_months,
    )
    df_expense["Month"] = df_expense[time].dt.to_period("M").dt.to_timestamp()
    df_expense = df_expense[[time, "Month", "Amount"]]
    df_ = df_expense

    # get current and previous month
    months = df_expense.Month.unique()
    current_month = pd.to_datetime(months[-1])
    # previous_month = pd.to_datetime(months[-2])
    previous_months = months[::-1][1:]

    # get n day time series for this month
    dr = pd.date_range(current_month, periods=current_month.daysinmonth, freq="D")
    df_ref = pd.DataFrame(dr, columns=["Day"])

    df0 = df_[df_.Month == current_month]
    df0_merged = df_ref.merge(
        df0[["Day", "Amount"]], how="left", left_on="Day", right_on="Day"
    )

    df_merge_temp = []
    for ii, pm in enumerate(previous_months):
        dfmi = df_[df_.Month == pm].copy()
        dfmi["Day"] = dfmi["Day"] + pd.DateOffset(months=ii + 1)
        dfmi_merged = df_ref.merge(
            dfmi[["Day", "Amount"]], how="left", left_on="Day", right_on="Day"
        )
        df_temp = dfmi_merged.set_index("Day")
        df_merge_temp.append(df_temp)

    df_merged_all = pd.concat(df_merge_temp, join="inner", axis=1)

    df_all = df0_merged.merge(
        df_merged_all.reset_index(),
        how="left",
        left_on="Day",
        right_on="Day",
    )
    df_all.set_index("Day", inplace=True)
    df_all.columns = pd.Series(pd.to_datetime(months)).dt.strftime("%b %Y").values[::-1]
    df_all = df_all.fillna(0)

    df_cumsum = df_all.cumsum()
    current = df_cumsum.iloc[:, 0]
    current.loc[current.index >= pd.to_datetime("today")] = np.nan
    df_cumsum.iloc[:, 0] = current
    df_cumsum.head()

    df_melt = pd.melt(df_cumsum.reset_index(), id_vars="Day")
    df_melt = df_melt.rename(columns={"variable": "Month", "value": "Expenses"})
    df_melt["Day"] = df_melt["Day"].dt.strftime("%b %d")
    df_melt.head()

    if not plot:
        return df_cumsum

    # plot previous time series
    fig, ax = plt.subplots(figsize=(7, 4))
    ax = sns.lineplot(
        data=df_melt,
        x="Day",
        y="Expenses",
        hue="Month",
        style="Month",
        ax=ax,
        # size="Month",
        palette=sns.color_palette("Set2")[: len(months)],
    )
    # x_dates = df_melt["Day"].dt.strftime("%b %d").unique()
    # ax.set_xticklabels(labels=x_labels)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))
    ax.grid(which="major", alpha=0.25)
    plt.xticks(rotation=70)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    return fig, ax, df_melt


def heatmap_holoviz_view(
    df,
    time="Month",
    cat="Group",
    n_months=6,
    t_type="Income",
    plot=True,
):
    df_income, df_expense = heatmap_transaction_viewer(
        df,
        t_type=["Income", "Expense"],
        time=time,
        category=cat,
        n_months=n_months,
        plot=False,
    )
    df_expense = -1 * df_expense

    if not plot:
        return df_income, df_expense

    cmap = {"Income": "BuGn", "Expense": "RdPu_r"}
    df_type = {"Income": df_income, "Expense": df_expense}

    if not plot:
        return df_income, df_expense

    hv_plot = (
        df_type[t_type]
        .T.hvplot.heatmap(
            x="columns",
            y="index",
            C="value",
            # title=f"{t_type} for the past {n_months} months",
            cmap=cmap[t_type],
            # xaxis='top',
            rot=80,
            width=700,
            alpha=0.8,
            line_color="grey",
            line_width=2,
            line_alpha=0.1,
            tools=["doubletap", "box_select"],
        )
        .opts(
            cformatter=NumeralTickFormatter(format="$0,0"),
        )
    )
    return hv_plot


def heatmap_transaction_viewer(
    df,
    t_type="Income",
    time="Month",
    category="Category",
    n_months=12,
    plot=True,
):
    # get long form of all income and expense data by month
    df_ = comparison_viewer(
        df,
        t=time,
        f=category,
        transaction_type=["Income", "Expense"],
        n_months=n_months,
        plot=False,
        show_top_n=20,
    )
    df_["Amount"] = df_["Amount"].apply(lambda x: np.floor(x))

    # use the expense month delta_t to get income
    pivot_columns = [category, "Type"]
    if category == "Type":
        pivot_columns = ["Type"]  # avoid duplicate col labels
    df_all = pd.pivot_table(
        data=df_,
        values="Amount",
        index=time,
        columns=pivot_columns,
    ).fillna(0)

    income_index = df_all.columns.get_level_values(len(pivot_columns) - 1) == "Income"
    df_income = df_all[df_all.columns[income_index]].copy()
    df_income.columns = df_income.columns.get_level_values(0)
    df_income = df_income.fillna(0)

    # get expense pivot table
    df_expense = pd.pivot_table(
        data=df_.query("Type == 'Expense'"),
        values="Amount",
        index=time,
        columns=[category],
    ).fillna(0)

    if time == "Month":
        df_income.index = pd.Series(df_income.index).dt.strftime("%b %Y")
        df_expense.index = pd.Series(df_expense.index).dt.strftime("%b %Y")
    if time == "Day":
        df_income.index = pd.Series(df_income.index).dt.strftime("%b %d %y")
        df_expense.index = pd.Series(df_expense.index).dt.strftime("%b %d %y")

    if not plot:
        return df_income, df_expense

    cmap_expense = sns.color_palette("RdPu", 15)
    cmap_income = sns.color_palette("BuGn", 15)

    if t_type == "Income":
        data = df_income
        cmap = cmap_income
    elif t_type == "Expense":
        data = df_expense
        cmap = cmap_expense

    fig, ax1 = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        data=data.T,
        cmap=cmap,
        cbar=True,
        # square=True,
        ax=ax1,
        cbar_kws={
            "orientation": "horizontal",
            "pad": 0.25,
        },
    )
    plt.xticks(rotation=70)

    return fig, ax1, data


##################################
def filter_data(
    df: pd.DataFrame,
    column: str = "Group",
    value: str = "Primary Income",
    delta_t: str = "Month",
    last_n_months: int = 12,
    query: str = None,
    additional_groupby: List[str] = [],
    new_col_name: str = None,
) -> pd.DataFrame:
    """Get filtered transaction data.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    column : str, optional
        _description_, by default "Group"
    value : str, optional
        _description_, by default "Primary Income"
    delta_t : str, optional
        _description_, by default "Month"
    last_n_months : int, optional
        _description_, by default 12
    query : str, optional
        _description_, by default None
    additional_groupby : List[str], optional
        _description_, by default []
    new_col_name : str, optional
        _description_, by default None

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # get start of last n month
    beginning_month = df.Date[0] - pd.offsets.MonthBegin(last_n_months + 1)
    df_recent = df[df["Date"] >= beginning_month]

    # group and filter the main dataframe
    if query is None:
        query = f"{column} == '{value}'"
    df_summary = (
        df_recent.query(query)
        .groupby([delta_t] + additional_groupby)
        .agg({"Amount": lambda x: np.sum(x)})
    )
    df_summary.reset_index(inplace=True)
    # make amount always positive for either expense or income
    df_summary["Amount"] = np.abs(df_summary["Amount"])

    if new_col_name is not None:
        df_summary.rename(columns={"Amount": new_col_name}, inplace=True)

    return df_summary


def bar_plot_monthly(
    df: pd.DataFrame,
    column: str = "Group",
    value: str = "Primary Income",
    delta_t: str = "Month",
    last_n_months: int = 12,
    query: str = None,
    additional_groupby: List[str] = [],
    colors=None,
) -> Tuple[plt.figure, plt.Axes]:
    """Group monthly data and plot.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    column : str, optional
        _description_, by default "Group"
    value : str, optional
        _description_, by default "Primary Income"
    delta_t : str, optional
        _description_, by default "Month"
    last_n_months : int, optional
        _description_, by default 12
    query : str, optional
        _description_, by default None
    additional_groupby : List[str], optional
        _description_, by default []

    Returns
    -------
    Tuple[plt.figure, plt.Axes]
        _description_
    """

    df_summary = filter_data(
        df,
        column,
        value,
        delta_t,
        last_n_months,
        query,
        additional_groupby,
    )

    if len(additional_groupby) == 0:
        hue = None
    else:
        hue = additional_groupby[0]

    if colors is None:
        colors = sns.color_palette("Set2")

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=df_summary, x=delta_t, y="Amount", ax=ax, hue=hue, palette=colors)
    if delta_t == "Month":
        x_dates = df_summary[delta_t].dt.strftime("%b %Y").unique()
        ax.set_xticklabels(labels=x_dates)
    elif delta_t == "Week" or "Day":
        x_dates = df_summary[delta_t].dt.strftime("%b %d %Y").unique()
        ax.set_xticklabels(labels=x_dates)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(which="both", alpha=0.25)

    plt.xticks(rotation=70)
    if hue is not None:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="best",
            ncol=0.5 * len(df_summary.columns),
        )
    return fig, ax, df_summary


# deprecated and may not work
def cumulative_spending_comparison(
    df: pd.DataFrame, last_n_months: int = 2, plot: bool = True
) -> pd.DataFrame:
    """Look at daily spending compared to previous months

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    last_n_months : int, optional
        _description_, by default 2
    plot : bool, optional
        _description_, by default True

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # Look back n months
    beginning_month = df.Date[0] - pd.offsets.MonthBegin(last_n_months)
    df_recent = df[df["Date"] >= beginning_month]

    # get cumulative expenses grouped by year, month and day
    df_temp = df_recent[df_recent.Type == "Expense"].copy()
    A = df_temp.groupby(["Year", "Month", "Day"]).agg(
        {"Amount": lambda x: -1 * np.sum(x)}
    )
    A.rename(columns={"Amount": "Running Total"}, inplace=True)
    df_cumsum = A.groupby(["Year", "Month"]).cumsum()
    # df_cumsum.sort_values("Month",inplace=True, ascending=False)
    df_cumsum = df_cumsum.reset_index(drop=False)

    # convert monthly columns to strings for plotting
    months = df_cumsum["Month"].sort_values(ascending=False).unique()
    time_series = {}
    months_dt = (
        df_cumsum["Month"].sort_values(ascending=False).dt.strftime("%b %Y").unique()
    )

    months = df_cumsum["Month"].sort_values(ascending=False).dt.to_period("M").unique()
    # pad the monthly data and shift to current month to compare
    for i, month in enumerate(months):
        idx_temp = pd.date_range(month.start_time, month.end_time)
        tsi = df_cumsum[df_cumsum["Month"] == month]
        tsi = tsi[["Day", "Running Total"]].set_index("Day")
        tsi.rename(columns={"Running Total": f"{months_dt[i]}"}, inplace=True)
        tsi.index = tsi.index.to_timestamp()
        time_series[month] = tsi.reindex(idx_temp).reset_index(names=["Date"])

    # combine all the data in a single dataframe
    ts_test = time_series[months[0]]
    ts_all = ts_test.copy()
    for i, month in enumerate(months[1:]):
        # test month ie current month
        ts_ref = time_series[month].fillna(method="ffill")
        ts_ref["Date"] = ts_ref.Date + pd.DateOffset(months=i + 1)
        ts_all = ts_all.merge(ts_ref, how="left", left_on="Date", right_on="Date")

    # split timeseres into current and previous months
    ts_all_new = ts_all.set_index("Date")
    ts_current = ts_all_new.iloc[:, 0].to_frame()
    ts_previous = pd.DataFrame(ts_all_new.iloc[:, 1:])

    if not plot:
        return ts_all_new

    # for group plotting purposes
    df_melt = pd.melt(ts_all_new.reset_index(), id_vars="Date")
    df_melt_previous = pd.melt(ts_previous.reset_index(), id_vars="Date")
    df_melt_current = pd.melt(ts_current.reset_index(), id_vars="Date")

    cmap = plt.get_cmap("tab10")
    colors = cmap.colors
    palette = sns.color_palette("mako_r", 6)

    # plot previous time series
    fig, ax = plt.subplots(figsize=(8, 5))
    # plot both current month time series with different style
    n_lines = len(df_melt.variable.unique())

    sns.lineplot(
        data=df_melt.reset_index(),
        x="Date",
        y="value",
        hue="variable",
        style="variable",
        ax=ax,
        size="variable",
        palette=sns.color_palette("Set2")[:n_lines],
    )
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(which="both", alpha=0.25)
    plt.xticks(rotation=70)
    plt.legend(
        bbox_to_anchor=(1.02, 1), loc="best", ncol=0.5 * len(df_melt_current.columns)
    )
    return fig, ax, ts_all_new


if __name__ == "__main__":
    # Test loading the data
    src = "data/"
    file = "Tiller Foundation Template - Transactions.csv"
    df = load_and_process_transactions(file, src)
    print(df.columns)
