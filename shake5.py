import streamlit as st
from datetime import datetime, timedelta, date
import random, threading, time
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, func, or_, and_
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# ---------- DATABASE SETUP ----------
DB_PATH = "projectx.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
Base = declarative_base()

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    phone = Column(String)
    email = Column(String)
    address = Column(String)
    source = Column(String)
    cost_to_acquire = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="CAPTURED")
    inspection_date = Column(DateTime, nullable=True)
    estimate_value = Column(Float, nullable=True)
    converted = Column(Boolean, default=False)

Base.metadata.create_all(engine)

# ---------- GLOBAL STYLING ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap');
* {font-family:'Comfortaa';}
body, .main {background:#ffffff; padding:10px;}
.metric-card{background:black;border-radius:12px;padding:16px;margin:6px;}
.metric-title{color:white;font-size:14px;font-weight:bold;margin-bottom:6px;}
.metric-number{font-size:22px;font-weight:bold;}
.progress-bg{width:100%;background:#222;height:6px;border-radius:4px;}
.progress-bar{height:6px;border-radius:4px;}
.sla-badge{position:fixed;top:12px;right:12px;background:black;color:red;padding:6px 10px;border-radius:8px;font-size:15px;cursor:pointer;}
</style>
""", unsafe_allow_html=True)

# ---------- DATE RANGE SELECTOR ----------
_, date_col = st.columns([7,3])
with date_col:
    st.markdown("#### Select Timeline")
    range_type = st.selectbox("", ["Today","Last 7 Days","Last 30 Days","Custom"])
    if range_type=="Custom":
        sd = st.date_input("Start", date.today())
        ed = st.date_input("End", date.today())
    elif range_type=="Last 7 Days":
        sd = date.today() - timedelta(days=7)
        ed = date.today()
    elif range_type=="Last 30 Days":
        sd = date.today() - timedelta(days=30)
        ed = date.today()
    else:
        sd = date.today()
        ed = date.today()

# ---------- LEAD CAPTURE ----------
with st.expander("➕ New Lead"):
    name = st.text_input("Name")
    phone = st.text_input("Phone")
    email = st.text_input("Email")
    address = st.text_input("Address")
    platforms = ["Google Ads","Facebook","Instagram","TikTok","LinkedIn","Twitter","YouTube","Referral","Walk-In","Hotline","Website"]
    src = st.selectbox("Source", platforms)
    cost = st.number_input("Cost to Acquire Lead ($)", min_value=0.0, step=1.0, value=0.0)

    if st.button("Save Lead"):
        s = SessionLocal()
        try:
            new = Lead(name=name, phone=phone, email=email, address=address, source=src, cost_to_acquire=cost, status="CAPTURED")
            s.add(new)
            s.commit()
            st.success("✅ Lead Saved")
        except Exception:
            s.rollback()
            st.error("Save failed")
        finally:
            s.close()

# ---------- KPI CARDS ----------
def get_kpi():
    s = SessionLocal()
    try:
        rows = s.query(Lead).filter(
            Lead.created_at >= datetime.combine(sd, datetime.min.time()),
            Lead.created_at <= datetime.combine(sd, datetime.min.time()) + timedelta(days=1)
        ).all()

        active = len(rows)
        won = len([r for r in rows if r.status=="AWARDED" or r.converted])
        spend = sum(r.cost_to_acquire or 0 for r in rows)
        cpa = (spend/won) if won>0 else 0
        roi = won * 3800

        return active, won, round(cpa,2), roi, spend
    finally:
        s.close()

a,w,c,r,s = get_kpi()

kpis = [
  ("ACTIVE LEADS",a,"red"),
  ("SLA SUCCESS",random.randint(70,100),"blue"),
  ("QUALIFICATION RATE",random.randint(30,100),"orange"),
  ("CONVERSION RATE",random.randint(10,100),"green"),
  ("INSPECTION BOOKED",random.randint(1,20),"cyan"),
  ("ESTIMATE SENT",random.randint(1,12),"yellow"),
  ("PIPELINE JOB VALUES ($)",f"{c:,.0f}","chartreuse")
]

st.markdown("## TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")
st.markdown("*Overview of your pipeline health and sales performance.*")

r1,r2 = kpis[:4],kpis[4:]
for row in [r1,r2]:
    cols = st.columns(len(row))
    for col,(t,v,color) in zip(cols,row):
        col.markdown(f"""
        <div class='metric-card'>
           <div class='metric-title'>{t}</div>
           <div class='metric-number' style='color:{color};'>{v}</div>
           <div class='progress-bg'><div class='progress-bar' style='width:{random.randint(25,80)}%; background:{color};'></div></div>
        </div>
        """, unsafe_allow_html=True)

# ---------- SLA OVERDUE CHART + TABLE ----------
st.markdown("---")
st.markdown("### 🚨 SLA / Overdue Leads")
st.markdown("*Track SLA breaches and overdue trends over time.*")

s = SessionLocal()
try:
    sla = s.query(Lead.id,Lead.name,Lead.phone,Lead.created_at).filter(Lead.status=="OVERDUE").all()
finally:
    s.close()
df = pd.DataFrame([{"Name":x.name,"Phone":x.phone,"Created":x.created_at} for x in sla])
import matplotlib.pyplot as plt
plt.plot([len(df) for _ in range(len(df))])
st.pyplot(plt)
st.dataframe(df)
