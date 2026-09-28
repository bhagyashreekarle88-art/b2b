import os
import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')

linear_model = joblib.load(os.path.join(MODEL_DIR, 'linear_potential_deal_model.joblib'))
logistic_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_conversion_model.joblib'))
cluster_scaler = joblib.load(os.path.join(MODEL_DIR, 'cluster_scaler.joblib'))
kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans_opportunity_model.joblib'))
meta = joblib.load(os.path.join(MODEL_DIR, 'metadata.joblib'))

st.set_page_config(
    page_title='B2B Opportunity Prioritization Tool',
    page_icon='📊',
    layout='wide'
)

st.title('📊 B2B Opportunity Prioritization Tool')
st.caption('Analytics decision-support application built from the final B2B Sales & Marketing project models.')

st.info(
    'This tool provides model-based estimates from the historical dataset. '
    'It is a decision-support aid, not an automatic sales decision-maker.'
)

with st.sidebar:
    st.header('Model Information')
    st.write('**Conversion model:** Logistic Regression')
    st.write('**Opportunity value model:** Linear Regression')
    st.write('**Segmentation:** K-Means clustering')
    st.divider()
    st.write('**Conversion test ROC-AUC:** 0.695')
    st.write('**Conversion test recall:** 94.51%')
    st.write('**Deal-value test R²:** 0.940')
    st.divider()
    st.caption('Models use the same preprocessing logic and variable definitions as the final analytical notebook.')

st.subheader('1. Enter Opportunity Information')

with st.form('opportunity_form'):
    st.markdown('### Company Information')
    c1, c2, c3 = st.columns(3)
    with c1:
        industry = st.selectbox('Industry', ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail', 'Education', 'Other'])
        region = st.selectbox('Region', ['West', 'North', 'South', 'East'])
    with c2:
        company_size = st.selectbox('Company Size', ['Small', 'Medium', 'Large'])
        employees = st.number_input('Number of Employees', min_value=1, value=500, step=10)
    with c3:
        annual_revenue = st.number_input('Annual Revenue (Million)', min_value=0.0, value=250.0, step=10.0)

    st.markdown('### Lead Information')
    c1, c2, c3 = st.columns(3)
    with c1:
        lead_source = st.selectbox('Lead Source', ['LinkedIn', 'Google Ads', 'Referral', 'Partner', 'Website', 'Trade Show', 'Email Campaign'])
    with c2:
        decision_maker = st.selectbox('Decision Maker Level', ['Director', 'Manager', 'VP', 'C-Level'])
    with c3:
        product_interest = st.selectbox('Product Interest', ['Enterprise Software', 'Cloud Services', 'Cybersecurity', 'Data Analytics', 'IT Services', 'Other'])

    c1, c2 = st.columns(2)
    with c1:
        lead_score = st.number_input('Lead Score', min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    with c2:
        current_deal_estimate = st.number_input(
            'Current Estimated Deal Size',
            min_value=0.0,
            value=50000.0,
            step=5000.0,
            help='This is the current opportunity estimate used as an input to the conversion model. The Linear Regression model separately estimates Potential Deal Size from the other opportunity characteristics.'
        )

    st.markdown('### Marketing Engagement')
    c1, c2, c3 = st.columns(3)
    with c1:
        website_visits = st.number_input('Website Visits', min_value=0, value=10, step=1)
        email_opens = st.number_input('Email Opens', min_value=0, value=8, step=1)
    with c2:
        email_clicks = st.number_input('Email Clicks', min_value=0, value=3, step=1)
        webinars = st.number_input('Webinars Attended', min_value=0, value=1, step=1)
    with c3:
        engagement_score = st.number_input('Engagement Score', min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        marketing_spend = st.number_input('Marketing Spend', min_value=0.0, value=2500.0, step=100.0)

    st.markdown('### Sales Activity')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sales_calls = st.number_input('Sales Calls', min_value=0, value=5, step=1)
    with c2:
        meetings = st.number_input('Meetings Attended', min_value=0, value=3, step=1)
    with c3:
        sales_cycle = st.number_input('Sales Cycle (Days)', min_value=0, value=45, step=1)
    with c4:
        discount = st.number_input('Discount (%)', min_value=0.0, max_value=100.0, value=8.0, step=0.5)

    submitted = st.form_submit_button('🔍 Analyze Opportunity', use_container_width=True)

if submitted:
    # Build a single-row dataframe with the exact names expected by the saved pipelines.
    row = {
        'NumberOfEmployees': employees,
        'AnnualRevenue_Million': annual_revenue,
        'LeadScore': lead_score,
        'WebsiteVisits': website_visits,
        'EmailOpens': email_opens,
        'EmailClicks': email_clicks,
        'SalesCalls': sales_calls,
        'MeetingsAttended': meetings,
        'WebinarsAttended': webinars,
        'EngagementScore': engagement_score,
        'MarketingSpend': marketing_spend,
        'DiscountPercent': discount,
        'PotentialDealSize': current_deal_estimate,
        'SalesCycleDays': sales_cycle,
        'Industry': industry,
        'Region': region,
        'CompanySize': company_size,
        'LeadSource': lead_source,
        'DecisionMakerLevel': decision_maker,
        'ProductInterest': product_interest,
    }
    input_df = pd.DataFrame([row])

    # Predictions
    conversion_probability = float(logistic_model.predict_proba(input_df)[0, 1])
    predicted_deal_value = float(linear_model.predict(input_df.drop(columns=['PotentialDealSize']))[0])

    cluster_input = input_df[meta['cluster_features']]
    cluster_label_num = int(kmeans.predict(cluster_scaler.transform(cluster_input))[0])
    segment = meta['cluster_labels'][str(cluster_label_num)]

    st.divider()
    st.subheader('2. Model Outputs')
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric('Predicted Conversion Probability', f'{conversion_probability:.1%}')
    with m2:
        st.metric('Model-Estimated Potential Deal Size', f'₹{predicted_deal_value:,.0f}')
    with m3:
        st.metric('Opportunity Segment', segment)

    st.subheader('3. Business Interpretation')
    st.write(
        f'The Logistic Regression model estimates a conversion probability of **{conversion_probability:.1%}** for the entered opportunity. '
        f'The Linear Regression model estimates a potential deal value of **₹{predicted_deal_value:,.0f}**. '
        f'Based on the four clustering variables used in the project, the opportunity is assigned to the **{segment}**.'
    )

    st.warning(
        'Use these outputs together with sales judgement. The historical models identify statistical patterns; '
        'they do not prove that changing an input will cause the predicted outcome to change.'
    )

    st.subheader('4. Suggested Decision-Support View')
    if conversion_probability >= 0.75:
        conversion_note = 'The model places this opportunity in a relatively high predicted-conversion range; it may merit priority review.'
    elif conversion_probability >= 0.50:
        conversion_note = 'The model places this opportunity in an intermediate predicted-conversion range; review the opportunity context and next sales action.'
    else:
        conversion_note = 'The model places this opportunity in a lower predicted-conversion range; review lead quality, engagement and sales-process context before allocating additional effort.'

    st.write(conversion_note)
    st.write(
        'The segment should be interpreted as a profile of similar observations in the historical dataset, '
        'not as a universal quality ranking. High-Scale and Lower-Scale are descriptive labels used in the project analysis.'
    )

    with st.expander('See the opportunity inputs used by the models'):
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

st.divider()
st.caption('Academic deployment prototype | B2B Sales & Marketing Analytics | Google Colab / Python model logic operationalized through Streamlit')
