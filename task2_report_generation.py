"""
============================================================
CODTECH INTERNSHIP - TASK 2
Automated Report Generation
Reads sales data, analyzes it, generates a PDF report
using FPDF2
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fpdf import FPDF
from datetime import datetime
import os, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  CODTECH INTERNSHIP – TASK 2: AUTOMATED REPORT GENERATION")
print("=" * 60)

# ─── 1. Generate Sample CSV Data ─────────────────────────────
print("\n[1] Creating sample sales dataset...")
np.random.seed(7)
months     = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
products   = ['Laptop','Smartphone','Tablet','Headphones','Smartwatch']
regions    = ['North','South','East','West']

rows = []
for month in months:
    for product in products:
        for region in regions:
            base = {'Laptop':80000,'Smartphone':45000,'Tablet':30000,'Headphones':8000,'Smartwatch':20000}[product]
            qty  = np.random.randint(10, 80)
            price= base * np.random.uniform(0.9, 1.1)
            rows.append({'Month':month,'Product':product,'Region':region,
                         'Units_Sold':qty, 'Unit_Price':round(price,2),
                         'Revenue':round(qty*price,2),
                         'Cost':round(qty*price*0.65,2)})

df = pd.DataFrame(rows)
df['Profit'] = df['Revenue'] - df['Cost']
df.to_csv('/home/claude/sales_data.csv', index=False)
print(f"   ✅ Dataset: {len(df)} rows × {len(df.columns)} columns saved.")

# ─── 2. Analyze Data ─────────────────────────────────────────
print("\n[2] Analyzing data...")
total_revenue = df['Revenue'].sum()
total_profit  = df['Profit'].sum()
total_units   = df['Units_Sold'].sum()
profit_margin = (total_profit / total_revenue) * 100

by_product = df.groupby('Product').agg(
    Revenue=('Revenue','sum'), Profit=('Profit','sum'), Units=('Units_Sold','sum')
).sort_values('Revenue', ascending=False).reset_index()
by_region  = df.groupby('Region').agg(Revenue=('Revenue','sum'), Profit=('Profit','sum')).reset_index()
by_month   = df.groupby('Month').agg(Revenue=('Revenue','sum')).reindex(months).reset_index()

print(f"   Total Revenue : ₹{total_revenue:,.0f}")
print(f"   Total Profit  : ₹{total_profit:,.0f}")
print(f"   Profit Margin : {profit_margin:.1f}%")
print(f"   Units Sold    : {total_units:,}")

# ─── 3. Create Charts ─────────────────────────────────────────
print("\n[3] Generating charts...")
COLORS = ['#E63946','#457B9D','#2A9D8F','#E9C46A','#F4A261']
BG     = '#FFFFFF'

# Chart A – Revenue by Product
fig, ax = plt.subplots(figsize=(8,4), facecolor=BG)
bars = ax.bar(by_product['Product'], by_product['Revenue']/1e6, color=COLORS, edgecolor='white', linewidth=1.2)
ax.set_title('Revenue by Product (₹ Millions)', fontsize=13, fontweight='bold', pad=10)
ax.set_ylabel('Revenue (₹M)'); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, by_product['Revenue']/1e6):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f'₹{val:.1f}M', ha='center', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/claude/chart_product.png', dpi=120, bbox_inches='tight')
plt.close()

# Chart B – Monthly Revenue Trend
fig, ax = plt.subplots(figsize=(10,4), facecolor=BG)
ax.plot(by_month['Month'], by_month['Revenue']/1e6, marker='o', linewidth=2.5,
        markersize=7, color='#E63946', markerfacecolor='white', markeredgewidth=2)
ax.fill_between(range(len(months)), by_month['Revenue']/1e6, alpha=0.1, color='#E63946')
ax.set_xticks(range(len(months))); ax.set_xticklabels(months)
ax.set_title('Monthly Revenue Trend (₹ Millions)', fontsize=13, fontweight='bold', pad=10)
ax.set_ylabel('Revenue (₹M)'); ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/claude/chart_monthly.png', dpi=120, bbox_inches='tight')
plt.close()

# Chart C – Region Pie
fig, ax = plt.subplots(figsize=(6,5), facecolor=BG)
ax.pie(by_region['Revenue'], labels=by_region['Region'], colors=COLORS[:4],
       autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor':'white','linewidth':1.5})
ax.set_title('Revenue Share by Region', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('/home/claude/chart_region.png', dpi=120, bbox_inches='tight')
plt.close()

print("   ✅ Charts created.")

# ─── 4. Build PDF Report ──────────────────────────────────────
print("\n[4] Building PDF report...")

class SalesReport(FPDF):
    def header(self):
        self.set_fill_color(29, 53, 87)
        self.rect(0, 0, 210, 18, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 13)
        self.set_y(5)
        self.cell(0, 8, 'CODTECH INTERNSHIP  |  AUTOMATED SALES REPORT', align='C')
        self.set_text_color(0, 0, 0)
        self.ln(14)

    def footer(self):
        self.set_y(-13)
        self.set_fill_color(230, 57, 70)
        self.rect(0, self.get_y(), 210, 13, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 13, f'Generated by Python FPDF2  |  Page {self.page_no()}  |  {datetime.now().strftime("%d %b %Y")}', align='C')

    def section_title(self, title):
        self.set_fill_color(69, 123, 157)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, f'  {title}', fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kpi_box(self, label, value, x, y, w=44, h=20, color=(230,57,70)):
        self.set_fill_color(*color)
        self.rect(x, y, w, h, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 7)
        self.set_xy(x, y+2)
        self.cell(w, 5, label, align='C')
        self.set_font('Helvetica', 'B', 9)
        self.set_xy(x, y+8)
        self.cell(w, 8, value, align='C')
        self.set_text_color(0,0,0)

pdf = SalesReport()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 22)
pdf.set_text_color(29, 53, 87)
pdf.cell(0, 10, 'Annual Sales Analysis Report', align='C', ln=True)
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 6, f'Fiscal Year 2024  |  Generated: {datetime.now().strftime("%d %B %Y")}', align='C', ln=True)
pdf.ln(6)

# KPI Row
pdf.section_title('Key Performance Indicators')
pdf.ln(2)
kpis = [
    ('Total Revenue', f'Rs.{total_revenue/1e6:.1f}M',  10, (230,57,70)),
    ('Total Profit',  f'Rs.{total_profit/1e6:.1f}M',   58, (42,157,143)),
    ('Profit Margin', f'{profit_margin:.1f}%',         106, (69,123,157)),
    ('Units Sold',    f'{total_units:,}',              154, (233,196,106)),
]
y_kpi = pdf.get_y()
for label, val, x, col in kpis:
    pdf.kpi_box(label, val, x, y_kpi, w=44, h=22, color=col)
pdf.ln(28)

# Executive Summary
pdf.section_title('Executive Summary')
pdf.set_font('Helvetica', '', 10)
top_product = by_product.iloc[0]
top_region  = by_region.sort_values('Revenue', ascending=False).iloc[0]
summary = (
    f"The company achieved total annual revenue of Rs.{total_revenue/1e6:.1f} million in FY2024, "
    f"with a net profit of Rs.{total_profit/1e6:.1f} million and a healthy profit margin of {profit_margin:.1f}%. "
    f"A total of {total_units:,} units were sold across all product categories and regions. "
    f"The top-performing product was {top_product['Product']} generating Rs.{top_product['Revenue']/1e6:.1f}M in revenue. "
    f"The {top_region['Region']} region led all regions in sales contribution."
)
pdf.multi_cell(0, 6, summary)
pdf.ln(4)

# Product Table
pdf.section_title('Product-wise Performance')
pdf.set_fill_color(29,53,87); pdf.set_text_color(255,255,255); pdf.set_font('Helvetica','B',9)
for col,w in [('Product',52),('Revenue (Rs.)',42),('Profit (Rs.)',42),('Units Sold',42)]:
    pdf.cell(w, 7, col, border=0, fill=True, align='C')
pdf.ln()
pdf.set_text_color(0,0,0); pdf.set_font('Helvetica','',9)
for idx, row in by_product.iterrows():
    fill = idx % 2 == 0
    pdf.set_fill_color(240,248,255) if fill else pdf.set_fill_color(255,255,255)
    pdf.cell(52, 6, row['Product'],                   border=0, fill=fill, align='L')
    pdf.cell(42, 6, f"Rs.{row['Revenue']/1e6:.2f}M",  border=0, fill=fill, align='C')
    pdf.cell(42, 6, f"Rs.{row['Profit']/1e6:.2f}M",   border=0, fill=fill, align='C')
    pdf.cell(42, 6, f"{int(row['Units']):,}",          border=0, fill=fill, align='C')
    pdf.ln()
pdf.ln(4)

# Charts
pdf.section_title('Revenue by Product')
pdf.image('/home/claude/chart_product.png', x=10, w=185)
pdf.ln(4)

pdf.section_title('Monthly Revenue Trend')
pdf.image('/home/claude/chart_monthly.png', x=10, w=185)
pdf.ln(4)

pdf.section_title('Regional Revenue Distribution')
pdf.image('/home/claude/chart_region.png', x=50, w=110)

# Save
out_pdf = '/mnt/user-data/outputs/task2_sales_report.pdf'
pdf.output(out_pdf)
import shutil
shutil.copy('/home/claude/task2_report_generation.py', '/mnt/user-data/outputs/task2_report_generation.py')
print(f"   ✅ PDF report saved → {out_pdf}")
print("\n✅ Task 2 Complete!")
