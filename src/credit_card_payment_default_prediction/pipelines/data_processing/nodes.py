from typing import Dict, Tuple

import pandas as pd


def preprocess_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw dataset's data types. 
    Some columns are already converted to numeric, but to add context and make the data more readable, we convert them 
    back to categorical variables.
    
    Data Dictionary:
    - `ID` - Unique ID of each client
    - `LIMIT_BAL` - Amount of given credit (NT dollars): It includes both the individual consumer credit and his/her family (supplementary) credit
    - `SEX` - Gender (1=male, 2=female)
    - `EDUCATION` - (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
    - `MARRIAGE` - Marital status (1=married, 2=single, 3=divorced)
    - `AGE` - Age of the client
    - `PAY_0` - Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
    - `PAY_2` - Repayment status in August, 2005 (scale same as above)
    - `PAY_3` - Repayment status in July, 2005 (scale same as above)
    - `PAY_4` - Repayment status in June, 2005 (scale same as above)
    - `PAY_5` - Repayment status in May, 2005 (scale same as above)
    - `PAY_6` - Repayment status in April, 2005 (scale same as above)
    - `BILL_AMT1` - Amount of bill statement in September, 2005 (NT dollar)
    - `BILL_AMT2` - Amount of bill statement in August, 2005 (NT dollar)
    - `BILL_AMT3` - Amount of bill statement in July, 2005 (NT dollar)
    - `BILL_AMT4` - Amount of bill statement in June, 2005 (NT dollar)
    - `BILL_AMT5` - Amount of bill statement in May, 2005 (NT dollar)
    - `BILL_AMT6` - Amount of bill statement in April, 2005 (NT dollar)
    - `PAY_AMT1` - Amount of previous payment in September, 2005 (NT dollar)
    - `PAY_AMT2` - Amount of previous payment in August, 2005 (NT dollar)
    - `PAY_AMT3` - Amount of previous payment in July, 2005 (NT dollar)
    - `PAY_AMT4` - Amount of previous payment in June, 2005 (NT dollar)
    - `PAY_AMT5` - Amount of previous payment in May, 2005 (NT dollar)
    - `PAY_AMT6` - Amount of previous payment in April, 2005 (NT dollar)
    - `default_payment_next_month` - Target Variable: Default payment (1=yes, 0=no)
    """
    data['SEX'] = pd.Categorical(data['SEX'].replace({1: 'Male', 2: 'Female'}), categories=['Male', 'Female'], ordered=False)
    education_mapping = {1: 'GraduateSchool', 2: 'University', 3: 'HighSchool', 4: 'Others', 5: 'Unknown-5', 6: 'Unknown-6', 0: 'MISSING'}
    data['EDUCATION'] = pd.Categorical(data['EDUCATION'].replace(education_mapping), categories=education_mapping.values(), ordered=False)
    marriage_mapping = {1: 'Married', 2: 'Single', 3: 'Divorced', 0: 'MISSING'}
    data['MARRIAGE'] = pd.Categorical(data['MARRIAGE'].replace(marriage_mapping), categories=marriage_mapping.values(), ordered=False)
    return data


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies, {"columns": companies.columns.tolist(), "data_type": "companies"}


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table
