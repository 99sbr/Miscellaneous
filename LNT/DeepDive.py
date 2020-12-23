__author__ = "Subir Verma"
__copyright__ = "Copyright 2020"

try:
    import pandas as pd

    pd.set_option('display.max_columns', None)
    from datetime import datetime
except Exception as e:
    raise ImportError(str(e))

try:
    data = pd.read_excel('Case Study - Deep Dive Analysis.xlsx', sheet_name='input_data')
except Exception as e:
    print(str(e))

# A handy dictionary to take care of deep dive levels
deep_dive_options = {
    'ProductLevel': ['Brand', 'Subbrand'],
    'Geographicalevel': ['Zone', 'Region']
}


def date_handler(date: str):
    """

    :param date: Expects String for date in '%b%Y' Format
    :return: '%Y/%m/%d' formatted date value
    """
    try:
        formatted_date = datetime.strptime(date, '%b%Y').strftime('%Y/%m/%d')
        return formatted_date
    except Exception as e:
        print(e)


def calculate_growth_rate(df: pd.DataFrame, target_period: str,
                          reference_period: str) -> float:
    """

    :param df: DataFrame
    :param target_period: Formatted Target Date to be analyzed
    :param reference_period:  Formatted Reference Date
    :return: Actual Growth Rate: ((target_sales - reference_sales) / reference_sales) * 100
    """
    target_sales = df[df.month == target_period]['Value Offtake(000 Rs)'].sum()
    reference_sales = df[df.month ==
                         reference_period]['Value Offtake(000 Rs)'].sum()
    growth_rate = ((target_sales - reference_sales) / reference_sales) * 100
    return growth_rate


def calculate_contribution(df: pd.DataFrame, target_period: str,
                           target_period_total_value_sale: float) -> float:
    """

    :param df: DataFrame
    :param target_period: Formatted Target Date to be analyzed
    :param target_period_total_value_sale: Total Sales Done with Target Period
    :return: Contribution: (target_sales / target_period_total_value_sale) * 100
    """
    target_sales = df[df.month == target_period]['Value Offtake(000 Rs)'].sum()
    contribution = (target_sales / target_period_total_value_sale) * 100
    return contribution


def deep_dive_analysis(manufacturer: str, target_period: str,
                       reference_period: str) -> pd.DataFrame:
    """

    :param manufacturer: Name of the Manufacturer
    :param target_period: Unformatted Target Date to be Analyzed
    :param reference_period:  Unformatted Reference Date
    :return: DataFrame containing the focus area sorted by product values
    """
    analysis_data = data[data.Manufacturer == manufacturer]
    # Date Handler to take care of Proper Date Formatting
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
        result_list = []
        for option in deep_dive_options.keys():
            levels = deep_dive_options[option]
            for level in levels:
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
                    product = growth_rate * (1 / 100) * contribution * (1 / 100)
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


if __name__ == '__main__':
    manufacturer = 'GLAXOSMITHKLINE'
    target_period = 'Apr2019'
    reference_period = 'Mar2019'
    print(deep_dive_analysis(manufacturer, target_period, reference_period))
