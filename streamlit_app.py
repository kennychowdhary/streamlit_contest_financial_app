""" My first Tiller Streamlit App."""

import tiller_utils
import streamlit as st
import holoviews as hv
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
]

skey = st.secrets["gcp_service_account"]
credentials = Credentials.from_service_account_info(
    skey,
    scopes=scopes,
)
client = gspread.authorize(credentials)


# Perform SQL query on the Google Sheet.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def load_data(url, sheet_name="Transactions"):
    sh = client.open_by_url(url)
    df = pd.DataFrame(sh.worksheet(sheet_name).get_all_records())
    return df


def transaction_viewer_demo(df: pd.DataFrame = None):
    if df is None:
        # plt.style.use("dark_background")
        df = tiller_utils.load_and_process_transactions(
            file="Tiller Foundation Template - Transactions PPI removed.csv",
            src="data/",
        )

    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        # # st.write("Income/ Expense")
        t_types = st.multiselect(
            "Income/Expense",
            options=["Income", "Expense"],
            default=["Income", "Expense"],
        )

    with col_b:
        view_by = st.selectbox(
            "View by",
            ["Group", "Category", "Type"],
            index=2,
        )

    with col_c:
        # slide for monthly comparison (1 - 12 months)
        last_n_months = st.selectbox(
            "Compare with previous months:",
            options=list(range(1, 6 + 1)),
            index=1,
        )

    include_retirement = False
    if not include_retirement:
        df = df.loc[df.Category != "Retirement"]

    # hvplot
    hv_plot = tiller_utils.comparison_holoviz_view(
        df,
        t="Month",
        f=view_by,
        n_months=last_n_months,
        t_types=t_types,
    )

    st.bokeh_chart(
        hv.render(hv_plot, backend="bokeh"),
    )


def trend_viewer_demo(df: pd.DataFrame = None):
    if df is None:
        # plt.style.use("dark_background")
        df = tiller_utils.load_and_process_transactions(
            file="Tiller Foundation Template - Transactions PPI removed.csv",
            src="data/",
        )

    wcol1, wcol2, wcol3 = st.columns([1, 1, 1])
    with wcol2:
        # slide for monthly comparison (1 - 12 months)
        last_n_months = st.selectbox(
            "View last N months",
            options=[1, 2, 3, 4, 5, 6],
            index=1,
        )

    # fig2, ax2, df_ = tiller_utils.monthly_spending_trend_viewer(
    #     df,
    #     last_n_months=last_n_months,
    #     plot=True,
    # )
    # st.pyplot(fig2)

    hv_plot = tiller_utils.monthly_spending_holoviz_view(
        df,
        n_months=last_n_months,
    )
    st.bokeh_chart(
        hv.render(hv_plot, backend="bokeh"),
    )


def heatmap_viewer(df: pd.DataFrame = None):
    if df is None:
        # plt.style.use("dark_background")
        df = tiller_utils.load_and_process_transactions(
            file="Tiller Foundation Template - Transactions PPI removed.csv",
            src="data/",
        )

    ############################### widgets
    cols = st.columns(3)
    with cols[0]:
        income_v_expense = st.selectbox(
            "Income/ Expense",
            options=["Income", "Expense"],
            index=0,
        )
    with cols[1]:
        category = st.selectbox(
            "View by",
            options=["Group", "Category"],
            index=0,
        )
    with cols[2]:
        last_n_months = st.selectbox(
            "Months",
            options=list(range(1, 20 + 1)),
            index=11,
        )

    ############################### plotting

    # fig3, ax3, df_ = tiller_utils.heatmap_transaction_viewer(
    #     df,
    #     t_type=income_v_expense,
    #     time="Month",
    #     category=category,
    #     n_months=last_n_months,
    #     plot=True,
    # )
    # st.pyplot(fig3)

    hv_plot = tiller_utils.heatmap_holoviz_view(
        df,
        time="Month",
        cat=category,
        n_months=last_n_months,
        t_type=income_v_expense,
        plot=True,
    )

    st.bokeh_chart(
        hv.render(hv_plot, backend="bokeh"),
    )


if __name__ == "__main__":
    st.set_page_config("Tiller Money Analysis Viewer", "📚")

    # st.set_page_config(layout="wide")
    st.header("Budget Analysis")
    st.markdown(
        """
        The following demonstration employs the Google Sheets API
        with Streamlit to automatically load and process transaction data from the Tiller worksheet, all with Python. To ensure data security, we utilize an encrypted/ hidden key to authenticate and access the data without writing it to file. Once loaded, the data is automatically analyzed, processed, and visualized within this application. Additionally, updates to the Tiller sheet are automatically reflected in our app, eliminating the need for manual data movement.

        """
    )

    st.markdown(
        """
        ### Expenses versus Income
        Let's take a look at our income vs. expenses.
        """
    )

    # # load the data once in the beginning
    sheet_url = st.secrets["private_gsheets_url"]
    data = load_data(sheet_url)
    df = tiller_utils.load_and_process_transactions(df=data)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        view_by = st.selectbox(
            "View by",
            ["Type", "Group", "Category"],
            index=1,
        )

    with col_b:
        # slide for monthly comparison (1 - 12 months)
        last_n_months = st.selectbox(
            "Compare with previous n months:",
            options=list(range(1, 6 + 1)),
            index=1,
        )

    other_col1, other_col2 = st.columns([2, 1])

    with other_col1:
        t_types = st.multiselect(
            "Income/Expense",
            options=["Income", "Expense"],
            default=["Income", "Expense"],
        )
    with other_col2:
        retirement = st.selectbox("Include Retirement", options=["Yes", "No"], index=1)

    if retirement == "No":
        df = df.loc[df.Group != "Retirement"]

    # hvplot transaction comparison plot
    hv_plot = tiller_utils.comparison_holoviz_view(
        df,
        t="Month",
        f=view_by,
        n_months=last_n_months,
        t_types=t_types,
    )

    st.bokeh_chart(
        hv.render(hv_plot, backend="bokeh"),
    )

    st.markdown(
        """
        ### Daily Spending Comparison
        Let's take a look at our expenses during the
        current month compared to the N previous months.
        """
    )

    hv_plot2 = tiller_utils.monthly_spending_holoviz_view(
        df,
        n_months=last_n_months,
    )
    st.bokeh_chart(
        hv.render(hv_plot2, backend="bokeh"),
    )

    st.markdown(
        """
        ### Transaction Time Series Viewer
        Finally, let's take a big picture view of our
        transactions over the year/ months as a function
        of both time AND categories. This allows us to see
        big picture trends, while the former allows us
        to see the daily trends.
        """
    )

    hv_plot3 = tiller_utils.heatmap_holoviz_view(
        df,
        time="Month",
        cat=view_by,
        n_months=12,
        t_type="Income",
        plot=True,
    )
    st.bokeh_chart(
        hv.render(hv_plot3, backend="bokeh"),
    )

    hv_plot3 = tiller_utils.heatmap_holoviz_view(
        df,
        time="Month",
        cat=view_by,
        n_months=12,
        t_type="Expense",
        plot=True,
    )
    st.bokeh_chart(
        hv.render(hv_plot3, backend="bokeh"),
    )
