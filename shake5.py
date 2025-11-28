"""
TITAN - Single-file Streamlit application (Fully functional)

This file is an updated, fully-functional single-file Streamlit app that:
- Persists leads to a local SQLite database (no external DB required).
- Implements CRUD operations for leads (create, read, update, delete).
- Supports import (CSV/XLSX) that appends to the DB and export to Excel.
- Implements CPA reports, time series, and interactive dashboards matching the mockup.
- Trains a simple lead scoring model and persists it to disk (joblib).
- Includes sane defaults, styling, and instructions to run.

How to run:
1. Install dependencies:
   pip install streamlit pandas numpy scikit-learn plotly openpyxl joblib
2. Run:
   streamlit run titan_full_app.py

Notes:
- This app stores data in a local file `titan_leads.db` in the same folder.
- For production, replace SQLite with a proper database and secure credential storage.

"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Connection
from datetime import datetime, timedelta
import io
import base64
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import plotly.express as px

# ----------------------------
# Configuration
# ----------------------------
DB_FILE = 'titan_leads.db'
MODEL_FILE = 'titan_lead_scoring.joblib'

st.set_page_config(page_title="TITAN - Lead Pipeline", layout='wide')

APP_CSS = """
<style>
body {background-color: #071129; color: #e6eef8}
.header {display:flex; align-items:center; gap:12px}
.metric-card {background: linear-gradient(135deg,#0ea5a0,#06b6d4); padding:14px; border-radius:10px; color:white}
.small {font-size:12px; opacity:0.95}
.kpi {font-size:26px; font-weight:700}
.card-row {display:flex; gap:12px; flex-wrap:wrap}
.table-container {background:#0b1220; padding:12px; border-radius:8px}
.dataframe tbody tr th:only-of-type {vertical-align: middle}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# ----------------------------
# Database helpers
# ----------------------------

def get_conn() -> Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT UNIQUE,
            created_at TEXT,
            source TEXT,
            stage TEXT,
            estimated_value REAL,
            ad_cost REAL,
            converted INTEGER,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ----------------------------
# Utility helpers
# ----------------------------

def to_dataframe(sql: str, params=()):
    conn = get_conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def insert_leads(df: pd.DataFrame):
    conn = get_conn()
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at']).astype(str)
    rows = df[['lead_id','created_at','source','stage','estimated_value','ad_cost','converted','notes']].fillna('').values.tolist()
    c = conn.cursor()
    for r in rows:
        try:
            c.execute('''INSERT OR IGNORE INTO leads (lead_id,created_at,source,stage,estimated_value,ad_cost,converted,notes) VALUES (?,?,?,?,?,?,?,?)''', r)
        except Exception as e:
            print('Insert error', e)
    conn.commit()
    conn.close()

def upsert_lead(row: dict):
    conn = get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO leads (lead_id,created_at,source,stage,estimated_value,ad_cost,converted,notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(lead_id) DO UPDATE SET
                 created_at=excluded.created_at, source=excluded.source, stage=excluded.stage,
                 estimated_value=excluded.estimated_value, ad_cost=excluded.ad_cost, converted=excluded.converted, notes=excluded.notes
    ''', (row['lead_id'], row['created_at'], row['source'], row['stage'], row['estimated_value'], row['ad_cost'], row['converted'], row.get('notes','')))
    conn.commit()
    conn.close()

def delete_lead(lead_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM leads WHERE lead_id=?', (lead_id,))
    conn.commit()
    conn.close()

# ----------------------------
# Seed mock data if DB empty
# ----------------------------

def generate_mock_leads_df(n=300):
    rng = np.random.default_rng(42)
    created = [ (datetime.now() - timedelta(days=int(x))).isoformat() for x in rng.integers(0,120,size=n) ]
    sources = rng.choice(['Google Ads','Organic','Referral','Facebook Ads','Direct','Partner'], size=n)
    stages = rng.choice(['New','Contacted','Qualified','Estimate Sent','Won','Lost'], size=n, p=[0.18,0.25,0.2,0.15,0.12,0.1])
    est_value = np.round(rng.normal(2500,1800,size=n).clip(200,25000),2)
    cost = np.round(rng.normal(55,35,size=n).clip(0,800),2)
    lead_id = [f"L{200000+i}" for i in range(n)]
    converted = np.where(stages=='Won',1,0)
    df = pd.DataFrame({'lead_id':lead_id,'created_at':created,'source':sources,'stage':stages,'estimated_value':est_value,'ad_cost':cost,'converted':converted,'notes':''})
    return df

if to_dataframe('SELECT COUNT(*) as cnt FROM leads').loc[0,'cnt'] == 0:
    seed = generate_mock_leads_df(300)
    insert_leads(seed)

# ----------------------------
# Model helpers
# ----------------------------

def train_model(df: pd.DataFrame):
    df2 = df.copy()
    df2['created_at'] = pd.to_datetime(df2['created_at'])
    df2['age_days'] = (datetime.now() - df2['created_at']).dt.days
    X = pd.get_dummies(df2[['source','stage']].astype(str))
    X['ad_cost'] = df2['ad_cost']
    X['estimated_value'] = df2['estimated_value']
    X['age_days'] = df2['age_days']
    y = df2['converted']
    if len(y.unique())==1:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # persist model and feature columns
    joblib.dump({'model': model, 'columns': X.columns.tolist()}, MODEL_FILE)
    return model, acc

def load_model():
    if os.path.exists(MODEL_FILE):
        obj = joblib.load(MODEL_FILE)
        return obj.get('model'), obj.get('columns')
    return None, None

# ----------------------------
# UI helpers
# ----------------------------

def download_link(df: pd.DataFrame, filename: str):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
    return f'<a href="{href}" download="{filename}">Download {filename}</a>'

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title('TITAN Control Panel')
page = st.sidebar.selectbox('Choose page', ['Dashboard','Leads','CPA','ML Lead Scoring','Imports/Exports','Settings','Reports'])

# ----------------------------
# Pages
# ----------------------------

def page_dashboard():
    st.markdown("<div class='header'><h1>📊 TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR</h1></div>", unsafe_allow_html=True)
    st.markdown("*\_A high-level snapshot of leads, conversion, and acquisition efficiency across channels_*")
    df = to_dataframe('SELECT * FROM leads')
    total_leads = len(df)
    new_leads = (df['stage']=='New').sum()
    contacted = (df['stage']=='Contacted').sum()
    conv_rate = df['converted'].mean() if total_leads>0 else 0
    avg_value = df.loc[df['converted']==1,'estimated_value'].mean() if df['converted'].sum()>0 else df['estimated_value'].mean()
    total_revenue = df.loc[df['converted']==1,'estimated_value'].sum()
    avg_cpa = df['ad_cost'].sum() / df['converted'].sum() if df['converted'].sum()>0 else df['ad_cost'].mean()

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='kpi'>{total_leads}</div><div class='small'>Total leads</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='kpi'>{new_leads}</div><div class='small'>New leads</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='kpi'>{contacted}</div><div class='small'>Contacted</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='kpi'>{conv_rate*100:.1f}%</div><div class='small'>Conversion rate</div></div>", unsafe_allow_html=True)

    st.subheader('Leads by Source (last 120 days)')
    df_plot = df.copy()
    df_plot['created_at'] = pd.to_datetime(df_plot['created_at'])
    df_plot['day'] = df_plot['created_at'].dt.date
    grouped = df_plot.groupby(['day','source']).size().reset_index(name='count')
    if not grouped.empty:
        fig = px.area(grouped, x='day', y='count', color='source', title='Leads by Source over time')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Pipeline stages')
    stage_counts = df['stage'].value_counts().reset_index()
    stage_counts.columns = ['stage','count']
    st.plotly_chart(px.bar(stage_counts, x='stage', y='count', title='Count per pipeline stage'), use_container_width=True)

    st.subheader('Recent leads')
    st.dataframe(df.sort_values('created_at', ascending=False).head(20))


def page_leads():
    st.header('Leads Management')
    df = to_dataframe('SELECT * FROM leads')
    left, right = st.columns([2,1])
    with left:
        st.subheader('Create / Update lead')
        with st.form('lead_form'):
            lead_id = st.text_input('Lead ID (unique)', value='')
            created_at = st.text_input('Created at (ISO) or leave blank for now', value=datetime.now().isoformat())
            source = st.selectbox('Source', ['Google Ads','Organic','Referral','Facebook Ads','Direct','Partner','Other'])
            stage = st.selectbox('Stage', ['New','Contacted','Qualified','Estimate Sent','Won','Lost'])
            estimated_value = st.number_input('Estimated value', value=0.0)
            ad_cost = st.number_input('Ad cost', value=0.0)
            converted = st.checkbox('Converted (Won)')
            notes = st.text_area('Notes')
            submitted = st.form_submit_button('Save lead')
            if submitted:
                if not lead_id:
                    st.error('Lead ID is required')
                else:
                    row = {'lead_id':lead_id,'created_at':created_at,'source':source,'stage':stage,'estimated_value':estimated_value,'ad_cost':ad_cost,'converted':int(converted),'notes':notes}
                    upsert_lead(row)
                    st.success('Lead saved')
    with right:
        st.subheader('Filter & Edit')
        src_filter = st.multiselect('Source', options=df['source'].unique().tolist(), default=df['source'].unique().tolist())
        stg_filter = st.multiselect('Stage', options=df['stage'].unique().tolist(), default=df['stage'].unique().tolist())
        min_date = st.date_input('From date', value=(datetime.now()-timedelta(days=120)).date())
        filtered = df[(df['source'].isin(src_filter)) & (df['stage'].isin(stg_filter)) & (pd.to_datetime(df['created_at']).dt.date >= min_date)]
        st.markdown(f'**Filtered leads: {len(filtered)}**')
        if not filtered.empty:
            sel = st.selectbox('Select lead to delete', options=[''] + filtered['lead_id'].tolist())
            if sel:
                if st.button('Delete selected lead'):
                    delete_lead(sel)
                    st.success('Lead deleted')
        if st.button('Export filtered to Excel'):
            st.markdown(download_link(filtered, 'filtered_leads.xlsx'), unsafe_allow_html=True)


def page_cpa():
    st.header('Cost per Acquisition (CPA)')
    df = to_dataframe('SELECT * FROM leads')
    by_source = df.groupby('source').agg({'ad_cost':'sum','converted':'sum'}).reset_index()
    by_source['cpa'] = by_source.apply(lambda r: r['ad_cost']/r['converted'] if r['converted']>0 else np.nan, axis=1)
    st.dataframe(by_source.sort_values('cpa'))
    st.plotly_chart(px.bar(by_source, x='source', y='cpa', title='CPA by source'), use_container_width=True)

    st.subheader('CPA over time')
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['day'] = df['created_at'].dt.date
    daily = df.groupby('day').agg({'ad_cost':'sum','converted':'sum'}).reset_index()
    daily['cpa'] = daily.apply(lambda r: r['ad_cost']/r['converted'] if r['converted']>0 else np.nan, axis=1)
    st.line_chart(daily.set_index('day')['cpa'])


def page_ml():
    st.header('ML Lead Scoring')
    df = to_dataframe('SELECT * FROM leads')
    if st.button('Train model on current data'):
        model, acc = train_model(df)
        if model is None:
            st.warning('Not enough variability in target to train a model.')
        else:
            st.success(f'Model trained. Test accuracy: {acc:.2f}')
    model, columns = load_model()
    if model is None:
        st.info('No trained model found. Train one using the button above.')
    else:
        st.success('Model loaded from disk')
        # prepare features
        df2 = df.copy()
        df2['created_at'] = pd.to_datetime(df2['created_at'])
        df2['age_days'] = (datetime.now() - df2['created_at']).dt.days
        X = pd.get_dummies(df2[['source','stage']].astype(str))
        for c in columns:
            if c not in X.columns:
                X[c] = 0
        X['ad_cost'] = df2['ad_cost']
        X['estimated_value'] = df2['estimated_value']
        X['age_days'] = df2['age_days']
        X = X.reindex(columns=columns, fill_value=0)
        try:
            scores = model.predict_proba(X)[:,1]
            df2['score'] = scores
            st.subheader('Top leads by predicted conversion probability')
            st.dataframe(df2.sort_values('score', ascending=False).head(15))
            if st.button('Export top scored leads'):
                st.markdown(download_link(df2.sort_values('score', ascending=False).head(100), 'top_scored_leads.xlsx'), unsafe_allow_html=True)
        except Exception as e:
            st.error(f'Error scoring leads: {e}')


def page_imports_exports():
    st.header('Import leads / Export dataset')
    uploaded = st.file_uploader('Upload CSV or Excel', type=['csv','xlsx'])
    if uploaded is not None:
        try:
            if uploaded.type == 'text/csv' or uploaded.name.lower().endswith('.csv'):
                new = pd.read_csv(uploaded)
            else:
                new = pd.read_excel(uploaded)
            # normalize expected columns
            expected = ['lead_id','created_at','source','stage','estimated_value','ad_cost','converted']
            for col in expected:
                if col not in new.columns:
                    new[col] = ''
            # ensure types
            if 'converted' in new.columns:
                new['converted'] = new['converted'].fillna(0).astype(int)
            if 'notes' not in new.columns:
                new['notes'] = ''
            insert_leads(new[expected + ['notes']])
            st.success(f'Appended {len(new)} rows to database')
        except Exception as e:
            st.error(f'Error reading file: {e}')
    if st.button('Export full dataset to Excel'):
        df = to_dataframe('SELECT * FROM leads')
        st.markdown(download_link(df, 'titan_all_leads.xlsx'), unsafe_allow_html=True)


def page_settings():
    st.header('Settings & Integrations')
    st.subheader('Local settings')
    st.write('DB file: ', DB_FILE)
    st.write('Model file: ', MODEL_FILE)
    st.markdown('**WordPress integration placeholder**')
    wp_url = st.text_input('WordPress Site URL (optional)')
    wp_user = st.text_input('WP Username')
    wp_app_pass = st.text_input('WP Application Password', type='password')
    if st.button('Save integration settings'):
        st.success('Settings saved locally in session (not persisted to disk in this demo). For production, store securely.')


def page_reports():
    st.header('Reports')
    df = to_dataframe('SELECT * FROM leads')
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['month'] = df['created_at'].dt.to_period('M')
    monthly = df.groupby('month').agg(total_leads=('lead_id','count'), conversions=('converted','sum'), revenue=('estimated_value', lambda x: x[df['converted']==1].sum()))
    st.subheader('Monthly performance')
    st.dataframe(monthly)
    if st.button('Export monthly report'):
        # expand monthly back to dataframe for export
        out = monthly.reset_index()
        st.markdown(download_link(out, 'monthly_report.xlsx'), unsafe_allow_html=True)

# Route pages
if page == 'Dashboard':
    page_dashboard()
elif page == 'Leads':
    page_leads()
elif page == 'CPA':
    page_cpa()
elif page == 'ML Lead Scoring':
    page_ml()
elif page == 'Imports/Exports':
    page_imports_exports()
elif page == 'Settings':
    page_settings()
elif page == 'Reports':
    page_reports()

# Footer
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('TITAN - Local prototype. For production use: secure credentials, use a server-side DB, add authentication and backups.')
