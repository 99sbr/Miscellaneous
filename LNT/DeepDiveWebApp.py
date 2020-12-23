import pandas as pd 
import streamlit as st 
import plotly.express as px 
import matplotlib.pyplot as plt 
import plotly.offline as pyoff 
import plotly.graph_objects as go 
import chart_studio.plotly as py 
import plotly.offline as py 
from plotly.graph_objects import Pie, Layout, Figure 
from datetime import datetime

st.title('DeepDive Web App')

@st.cache(persist=True)
def load_data():
  try:
    data = pd.read_excel('Case Study - Deep Dive Analysis.xlsx',sheet_name='input_data')
    return data 
  except Exception as e:
    print(str(e))


deep_dive_options = {
    'ProductLevel': ['Brand', 'Subbrand'],
    'Geographicalevel': ['Zone', 'Region']
}

def date_handler(date: str):
    return datetime.strptime(date, '%b%Y').strftime('%Y/%m/%d')

def calculate_growth_rate(df: pd.DataFrame, target_period: str,
                          reference_period: str) -> float:
    target_sales = df[df.month == target_period]['Value Offtake(000 Rs)'].sum()
    reference_sales = df[df.month ==
                         reference_period]['Value Offtake(000 Rs)'].sum()
    growth_rate = ((target_sales - reference_sales) / reference_sales) * 100
    return growth_rate

def calculate_contribution(df: pd.DataFrame, target_period: str,
                           target_period_total_value_sale: float) -> float:
    target_sales = df[df.month == target_period]['Value Offtake(000 Rs)'].sum()
    contribution = (target_sales / target_period_total_value_sale) * 100
    return contribution

def deep_dive_analysis(manufacturer: str, target_period: str,
                       reference_period: str) -> pd.DataFrame:
    analysis_data = data[data.Manufacturer == manufacturer]
    # Date Handler to take care of Proper Date Formating
    target_period = date_handler(target_period)
    reference_period = date_handler(reference_period)

    target_period_total_value_sale = analysis_data[
        analysis_data.month == target_period]['Value Offtake(000 Rs)'].sum()
    reference_period_total_value_sale = analysis_data[
        analysis_data.month ==
        reference_period]['Value Offtake(000 Rs)'].sum()

    gain = target_period_total_value_sale - reference_period_total_value_sale
    if gain >= 0:
        print(
            f"There is no drop in the sales for a {manufacturer} in the {target_period}"
        )
    else:
        # Let's deep dive
        result_list = []
        for option in deep_dive_options.keys():
            print(f'Doing {option} Analysis')
            levels = deep_dive_options[option]
            for level in levels:
                print(f'Level :{level}')
                focus_area_list = analysis_data[level].value_counts(
                ).index.to_list()
                for focus_area in focus_area_list:
                    growth_rate = calculate_growth_rate(
                        analysis_data[analysis_data[level] == focus_area],
                        target_period=target_period,
                        reference_period=reference_period)
                    contribution = calculate_contribution(
                        analysis_data[analysis_data[level] == focus_area],
                        target_period=target_period,
                        target_period_total_value_sale=target_period_total_value_sale)
                    product = growth_rate * contribution
                    result_list.append({
                        'Manufacturer': manufacturer,
                        'level': level,
                        'focus_area': focus_area,
                        'growth_rate': growth_rate,
                        'contribution': contribution,
                        'product': product
                    })
        deep_dive_df = pd.DataFrame(result_list)
        deep_dive_df.sort_values(by='product', inplace=True)
        return deep_dive_df

st.header('Deep Dive Aanlysis')
data = load_data()
manufacturer = st.selectbox('Select manufacturer', data.Manufacturer.value_counts().index.to_list())
target_period = st.text_area('target_period', 'Feb2019')
reference_period = st.text_area('reference_period', 'Jan2019')
result = deep_dive_analysis(manufacturer=manufacturer,target_period = target_period,reference_period = reference_period)
st.write(result)