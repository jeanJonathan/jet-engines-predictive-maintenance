"""
04_dashboard.py
================
Predictive Maintenance Dashboard — NASA C-MAPSS FD001
Author : Jean-Jonathan KOFFI

Features:
  - Multi-model selector (LSTM, XGBoost, Random Forest)
  - Fleet heatmap with risk zones
  - Engine detail view with gauge + prediction error
  - Model benchmark comparison tab
  - Anomaly flag table
"""

import os
import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from tensorflow import keras
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH  = os.path.join(BASE_DIR, 'data')
PROC_PATH  = os.path.join(DATA_PATH, 'processed')
MODEL_PATH = os.path.join(DATA_PATH, 'models')

# ─── Load predictions & models ────────────────────────────────────────────────
y_test     = np.load(os.path.join(PROC_PATH,  'y_test.npy'))
lstm_pred  = np.load(os.path.join(MODEL_PATH, 'lstm_pred_test.npy'))
rf_pred    = np.load(os.path.join(MODEL_PATH, 'rf_pred_test.npy'))

# XGBoost optional
xgb_path = os.path.join(MODEL_PATH, 'xgb_pred_test.npy')
xgb_pred = np.load(xgb_path) if os.path.exists(xgb_path) else None

# Benchmark
bench_path = os.path.join(MODEL_PATH, 'benchmark_results.json')
benchmark  = json.load(open(bench_path)) if os.path.exists(bench_path) else {}

PRED_MAP = {'LSTM (BiDir)': lstm_pred, 'Random Forest': rf_pred}
if xgb_pred is not None:
    PRED_MAP['XGBoost'] = xgb_pred
MODEL_OPTIONS = [{'label': k, 'value': k} for k in PRED_MAP]

N_ENGINES = len(y_test)

# ─── Risk classification ──────────────────────────────────────────────────────
def risk(rul):
    if rul < 30:  return 'CRITICAL', '#e74c3c'
    if rul < 80:  return 'WARNING',  '#e67e22'
    return 'HEALTHY', '#27ae60'

def build_fleet_df(pred):
    rows = []
    for i, (actual, predicted) in enumerate(zip(y_test, pred)):
        status, color = risk(predicted)
        error = float(predicted - actual)
        rows.append({
            'Engine': i + 1,
            'Actual RUL': round(float(actual), 1),
            'Predicted RUL': round(float(predicted), 1),
            'Error (cycles)': round(error, 1),
            'Status': status,
            '_color': color,
        })
    return pd.DataFrame(rows)

# ─── Palette & styles ─────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
CARD_BG   = '#161b22'
BORDER    = '#30363d'
TEXT      = '#c9d1d9'
ACCENT    = '#58a6ff'
FONT      = 'IBM Plex Mono, monospace'

CARD_STYLE = {
    'backgroundColor': CARD_BG,
    'border': f'1px solid {BORDER}',
    'borderRadius': '8px',
    'padding': '20px',
    'marginBottom': '16px',
}

# ─── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Predictive Maintenance — CMAPSS'

app.index_string = f'''
<!DOCTYPE html><html>
<head>
{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {DARK_BG}; color: {TEXT}; font-family: {FONT}; }}
  ::-webkit-scrollbar {{ width: 6px; }} ::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
  .tab-label {{ font-family: {FONT}; font-size: 12px; letter-spacing: 0.05em; }}
  .Select-control, .Select-menu-outer {{ background: {CARD_BG} !important; color: {TEXT} !important; border-color: {BORDER} !important; }}
</style>
</head>
<body>{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body></html>
'''

# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div(style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '24px'}, children=[

    # ── Header
    html.Div(style={'borderBottom': f'1px solid {BORDER}', 'paddingBottom': '16px', 'marginBottom': '24px'}, children=[
        html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '16px'}, children=[
            html.Div('⚙', style={'fontSize': '28px'}),
            html.Div([
                html.H1('Predictive Maintenance Dashboard', style={'fontSize': '20px', 'fontWeight': '600', 'color': ACCENT, 'fontFamily': 'IBM Plex Sans'}),
                html.P('NASA C-MAPSS FD001 · Turbofan RUL Prediction · Multi-Model Benchmark',
                       style={'fontSize': '11px', 'color': '#8b949e', 'marginTop': '4px', 'letterSpacing': '0.05em'}),
            ]),
            html.Div(style={'marginLeft': 'auto', 'display': 'flex', 'gap': '24px'}, children=[
                html.Div([html.P(f'{N_ENGINES}', style={'fontSize': '22px', 'fontWeight': '600', 'color': ACCENT}),
                          html.P('Engines', style={'fontSize': '10px', 'color': '#8b949e'})]),
                html.Div([html.P(str(len(PRED_MAP)), style={'fontSize': '22px', 'fontWeight': '600', 'color': '#a371f7'}),
                          html.P('Models', style={'fontSize': '10px', 'color': '#8b949e'})]),
            ])
        ])
    ]),

    # ── Controls
    html.Div(style={**CARD_STYLE, 'display': 'flex', 'gap': '32px', 'alignItems': 'center', 'marginBottom': '24px'}, children=[
        html.Div([
            html.Label('MODEL', style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '0.1em', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(id='model-select', options=MODEL_OPTIONS, value=list(PRED_MAP.keys())[0],
                         clearable=False, style={'width': '220px', 'fontSize': '12px'}),
        ]),
        html.Div([
            html.Label('ENGINE', style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '0.1em', 'marginBottom': '8px', 'display': 'block'}),
            dcc.Dropdown(id='engine-select',
                         options=[{'label': f'Engine #{i+1}', 'value': i} for i in range(N_ENGINES)],
                         value=0, clearable=False, style={'width': '180px', 'fontSize': '12px'}),
        ]),
        html.Div(id='kpi-strip', style={'marginLeft': 'auto', 'display': 'flex', 'gap': '32px'}),
    ]),

    # ── Tabs
    dcc.Tabs(id='tabs', value='fleet', style={'borderBottom': f'1px solid {BORDER}', 'marginBottom': '24px'}, children=[
        dcc.Tab(label='FLEET OVERVIEW',    value='fleet',     className='tab-label',
                style={'backgroundColor': DARK_BG, 'color': '#8b949e', 'border': 'none', 'padding': '10px 20px'},
                selected_style={'backgroundColor': CARD_BG, 'color': ACCENT, 'border': f'1px solid {BORDER}', 'borderBottom': f'1px solid {CARD_BG}', 'padding': '10px 20px'}),
        dcc.Tab(label='ENGINE DETAIL',     value='engine',    className='tab-label',
                style={'backgroundColor': DARK_BG, 'color': '#8b949e', 'border': 'none', 'padding': '10px 20px'},
                selected_style={'backgroundColor': CARD_BG, 'color': ACCENT, 'border': f'1px solid {BORDER}', 'borderBottom': f'1px solid {CARD_BG}', 'padding': '10px 20px'}),
        dcc.Tab(label='MODEL BENCHMARK',   value='benchmark', className='tab-label',
                style={'backgroundColor': DARK_BG, 'color': '#8b949e', 'border': 'none', 'padding': '10px 20px'},
                selected_style={'backgroundColor': CARD_BG, 'color': ACCENT, 'border': f'1px solid {BORDER}', 'borderBottom': f'1px solid {CARD_BG}', 'padding': '10px 20px'}),
        dcc.Tab(label='ANOMALY TABLE',     value='anomaly',   className='tab-label',
                style={'backgroundColor': DARK_BG, 'color': '#8b949e', 'border': 'none', 'padding': '10px 20px'},
                selected_style={'backgroundColor': CARD_BG, 'color': ACCENT, 'border': f'1px solid {BORDER}', 'borderBottom': f'1px solid {CARD_BG}', 'padding': '10px 20px'}),
    ]),
    html.Div(id='tab-content'),
])

# ─── KPI strip ────────────────────────────────────────────────────────────────
@app.callback(Output('kpi-strip', 'children'), [Input('model-select', 'value')])
def update_kpis(model_name):
    pred = PRED_MAP[model_name]
    critical = int(np.sum(pred < 30))
    warning  = int(np.sum((pred >= 30) & (pred < 80)))
    healthy  = int(np.sum(pred >= 80))
    kpis = [
        ('CRITICAL', str(critical), '#e74c3c'),
        ('WARNING',  str(warning),  '#e67e22'),
        ('HEALTHY',  str(healthy),  '#27ae60'),
    ]
    return [html.Div([
        html.P(label, style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '0.1em'}),
        html.P(val,   style={'fontSize': '20px', 'fontWeight': '600', 'color': color}),
    ]) for label, val, color in kpis]

# ─── Tab content ──────────────────────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value'), Input('model-select', 'value'), Input('engine-select', 'value')])
def render_tab(tab, model_name, engine_idx):
    pred = PRED_MAP[model_name]
    df   = build_fleet_df(pred)

    # ── FLEET ──────────────────────────────────────────────────────────────────
    if tab == 'fleet':
        # Scatter fleet
        scatter = go.Figure()
        for status, color in [('HEALTHY', '#27ae60'), ('WARNING', '#e67e22'), ('CRITICAL', '#e74c3c')]:
            mask = df['Status'] == status
            scatter.add_trace(go.Scatter(
                x=df[mask]['Engine'], y=df[mask]['Predicted RUL'],
                mode='markers', name=status,
                marker=dict(size=10, color=color, line=dict(width=1, color=DARK_BG)),
                text=df[mask].apply(lambda r: f"Engine #{r['Engine']}<br>Pred: {r['Predicted RUL']:.0f} cy<br>Actual: {r['Actual RUL']:.0f} cy", axis=1),
                hoverinfo='text'
            ))
        scatter.add_hline(y=30, line_dash='dash', line_color='#e74c3c', annotation_text='Critical', annotation_font_color='#e74c3c')
        scatter.add_hline(y=80, line_dash='dash', line_color='#e67e22', annotation_text='Warning',  annotation_font_color='#e67e22')
        scatter.update_layout(title=f'Fleet RUL — {model_name}', paper_bgcolor=CARD_BG, plot_bgcolor=DARK_BG,
                              font_color=TEXT, xaxis_title='Engine #', yaxis_title='Predicted RUL (cycles)',
                              legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1), height=420)

        # Error distribution
        errors = pred - y_test
        hist = go.Figure(go.Histogram(x=errors, nbinsx=30, marker_color=ACCENT, opacity=0.8))
        hist.add_vline(x=0, line_color='tomato', line_dash='dash')
        hist.update_layout(title='Prediction Error Distribution (Pred − Actual)',
                           paper_bgcolor=CARD_BG, plot_bgcolor=DARK_BG, font_color=TEXT,
                           xaxis_title='Error (cycles)', yaxis_title='Count', height=320)

        return html.Div([
            html.Div(dcc.Graph(figure=scatter), style=CARD_STYLE),
            html.Div(dcc.Graph(figure=hist),    style=CARD_STYLE),
        ])

    # ── ENGINE DETAIL ──────────────────────────────────────────────────────────
    elif tab == 'engine':
        actual    = float(y_test[engine_idx])
        predicted = float(pred[engine_idx])
        status, color = risk(predicted)
        error = predicted - actual

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=predicted,
            delta={'reference': actual, 'valueformat': '.0f', 'suffix': ' cy'},
            title={'text': f'Engine #{engine_idx+1} — Predicted RUL<br><span style="font-size:13px;color:{color}">{status}</span>'},
            gauge={
                'axis': {'range': [0, 130], 'tickcolor': TEXT},
                'bar':  {'color': color},
                'bgcolor': DARK_BG,
                'bordercolor': BORDER,
                'steps': [
                    {'range': [0, 30],  'color': '#3d1515'},
                    {'range': [30, 80], 'color': '#3d2a10'},
                    {'range': [80, 130],'color': '#0f2d1a'},
                ],
                'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.75, 'value': actual}
            }
        ))
        gauge.update_layout(paper_bgcolor=CARD_BG, font_color=TEXT, height=360)

        # Comparison across models
        model_names = list(PRED_MAP.keys())
        model_preds = [float(PRED_MAP[m][engine_idx]) for m in model_names]
        bar = go.Figure()
        bar.add_trace(go.Bar(name='Predicted', x=model_names, y=model_preds,
                             marker_color=[risk(p)[1] for p in model_preds]))
        bar.add_hline(y=actual, line_dash='dot', line_color='white',
                      annotation_text=f'Actual: {actual:.0f} cy', annotation_font_color='white')
        bar.update_layout(title=f'Engine #{engine_idx+1} — All Models vs Actual',
                          paper_bgcolor=CARD_BG, plot_bgcolor=DARK_BG, font_color=TEXT,
                          yaxis_title='Predicted RUL', height=360, showlegend=False)

        # Stats card
        stats = html.Div(style={**CARD_STYLE, 'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '16px'}, children=[
            html.Div([html.P('ACTUAL RUL',    style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '0.1em'}),
                      html.P(f'{actual:.0f} cy', style={'fontSize': '22px', 'color': TEXT})]),
            html.Div([html.P('PREDICTED RUL', style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '0.1em'}),
                      html.P(f'{predicted:.0f} cy', style={'fontSize': '22px', 'color': color})]),
            html.Div([html.P('ERROR',         style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '0.1em'}),
                      html.P(f'{error:+.1f} cy', style={'fontSize': '22px', 'color': '#e74c3c' if abs(error) > 20 else '#27ae60'})]),
            html.Div([html.P('STATUS',        style={'fontSize': '10px', 'color': '#8b949e', 'letterSpacing': '0.1em'}),
                      html.P(status, style={'fontSize': '22px', 'color': color})]),
        ])

        return html.Div([
            stats,
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}, children=[
                html.Div(dcc.Graph(figure=gauge), style=CARD_STYLE),
                html.Div(dcc.Graph(figure=bar),   style=CARD_STYLE),
            ]),
        ])

    # ── BENCHMARK ──────────────────────────────────────────────────────────────
    elif tab == 'benchmark':
        if not benchmark:
            return html.P("Run 03_train_model.py first to generate benchmark data.", style={'color': '#8b949e'})

        names  = list(benchmark.keys())
        rmses  = [benchmark[m]['rmse']  for m in names]
        maes   = [benchmark[m]['mae']   for m in names]
        r2s    = [benchmark[m]['r2']    for m in names]
        nscores= [benchmark[m]['nasa_score'] for m in names]
        colors = ['#58a6ff', '#ff9800', '#4caf50'][:len(names)]

        def bar_chart(x, y, title, yaxis=''):
            fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors[:len(x)],
                                   text=[f'{v:.2f}' for v in y], textposition='outside'))
            fig.update_layout(title=title, paper_bgcolor=CARD_BG, plot_bgcolor=DARK_BG,
                              font_color=TEXT, yaxis_title=yaxis, height=280, showlegend=False,
                              margin=dict(t=40, b=40))
            return fig

        return html.Div([
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}, children=[
                html.Div(dcc.Graph(figure=bar_chart(names, rmses,   'RMSE ↓ (lower is better)', 'cycles')), style=CARD_STYLE),
                html.Div(dcc.Graph(figure=bar_chart(names, maes,    'MAE ↓ (lower is better)',  'cycles')), style=CARD_STYLE),
                html.Div(dcc.Graph(figure=bar_chart(names, r2s,     'R² ↑ (higher is better)',  'R²')),     style=CARD_STYLE),
                html.Div(dcc.Graph(figure=bar_chart(names, nscores, 'NASA Score ↓ (asymmetric penalty)', 'Score')), style=CARD_STYLE),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.H3('Benchmark Table', style={'fontSize': '13px', 'color': ACCENT, 'marginBottom': '12px'}),
                dash_table.DataTable(
                    columns=[{'name': c, 'id': c} for c in ['Model', 'RMSE', 'MAE', 'R²', 'NASA Score', 'Train Time (s)']],
                    data=[{'Model': n, 'RMSE': f"{benchmark[n]['rmse']:.2f}",
                           'MAE': f"{benchmark[n]['mae']:.2f}", 'R²': f"{benchmark[n]['r2']:.3f}",
                           'NASA Score': f"{benchmark[n]['nasa_score']:.0f}",
                           'Train Time (s)': f"{benchmark[n].get('training_time_sec', '-')}"}
                          for n in names],
                    style_table={'overflowX': 'auto'},
                    style_cell={'backgroundColor': DARK_BG, 'color': TEXT, 'border': f'1px solid {BORDER}',
                                'fontFamily': FONT, 'fontSize': '12px', 'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': CARD_BG, 'color': ACCENT, 'fontWeight': '600',
                                  'border': f'1px solid {BORDER}'},
                )
            ])
        ])

    # ── ANOMALY TABLE ──────────────────────────────────────────────────────────
    elif tab == 'anomaly':
        anomalies = df[df['Status'].isin(['CRITICAL', 'WARNING'])].copy()
        anomalies = anomalies.sort_values('Predicted RUL')

        return html.Div(style=CARD_STYLE, children=[
            html.H3(f'{len(anomalies)} engines require attention ({model_name})',
                    style={'fontSize': '13px', 'color': '#e67e22', 'marginBottom': '16px'}),
            dash_table.DataTable(
                columns=[{'name': c, 'id': c} for c in ['Engine', 'Predicted RUL', 'Actual RUL', 'Error (cycles)', 'Status']],
                data=anomalies[['Engine', 'Predicted RUL', 'Actual RUL', 'Error (cycles)', 'Status']].to_dict('records'),
                sort_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={'backgroundColor': DARK_BG, 'color': TEXT, 'border': f'1px solid {BORDER}',
                            'fontFamily': FONT, 'fontSize': '12px', 'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': CARD_BG, 'color': ACCENT, 'fontWeight': '600',
                              'border': f'1px solid {BORDER}'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Status} = "CRITICAL"', 'column_id': 'Status'},
                     'color': '#e74c3c', 'fontWeight': '600'},
                    {'if': {'filter_query': '{Status} = "WARNING"', 'column_id': 'Status'},
                     'color': '#e67e22', 'fontWeight': '600'},
                ]
            )
        ])

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Predictive Maintenance Dashboard")
    print("  http://127.0.0.1:8050")
    print("="*55 + "\n")
    app.run(debug=False, port=8050)