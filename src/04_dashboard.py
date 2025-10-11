import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from tensorflow import keras
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
model = keras.models.load_model(os.path.join(DATA_PATH, 'models', 'lstm_model.keras'))

X_test = np.load(os.path.join(DATA_PATH, 'processed', 'X_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'processed', 'y_test.npy'))
y_pred = model.predict(X_test).flatten()

# Calculate risk levels
def get_risk_status(rul):
    if rul < 30:
        return 'CRITICAL', 'red'
    elif rul < 80:
        return 'WARNING', 'orange'
    else:
        return 'HEALTHY', 'green'

app = dash.Dash(__name__)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Montserrat', sans-serif;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.H1("Aircraft Engine Predictive Maintenance Dashboard",
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30, 'fontFamily': 'Montserrat'}),

    html.Div([
        html.Label("Select Engine:", style={'fontSize': 18, 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='engine-dropdown',
            options=[{'label': f'Engine {i+1}', 'value': i} for i in range(len(y_test))],
            value=0,
            style={'width': '300px'}
        ),
    ], style={'marginBottom': 30}),

    html.Div(id='engine-status', style={'marginBottom': 30}),

    html.Div([
        html.Div([
            dcc.Graph(id='rul-gauge')
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='prediction-comparison')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),

    html.Div([
        dcc.Graph(id='fleet-overview')
    ], style={'marginTop': 30}),

], style={'padding': 40, 'backgroundColor': '#ecf0f1'})

@app.callback(
    [Output('engine-status', 'children'),
     Output('rul-gauge', 'figure'),
     Output('prediction-comparison', 'figure'),
     Output('fleet-overview', 'figure')],
    [Input('engine-dropdown', 'value')]
)
def update_dashboard(engine_id):
    actual_rul = y_test[engine_id]
    predicted_rul = y_pred[engine_id]
    status, color = get_risk_status(predicted_rul)

    # Status card
    status_card = html.Div([
        html.Div([
            html.H3(f"Engine #{engine_id + 1}", style={'margin': 0}),
            html.H2(f"{int(predicted_rul)} cycles remaining", style={'margin': '10px 0'}),
            html.H4(f"Status: {status}", style={'margin': 0, 'color': color}),
        ], style={
            'backgroundColor': 'white',
            'padding': 20,
            'borderRadius': 10,
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'border': f'3px solid {color}'
        })
    ])

    # RUL gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        title={'text': "Remaining Useful Life (cycles)"},
        gauge={
            'axis': {'range': [0, 300]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightcoral"},
                {'range': [30, 80], 'color': "lightyellow"},
                {'range': [80, 300], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    gauge.update_layout(height=350)

    # Prediction comparison
    comparison = go.Figure()
    comparison.add_trace(go.Bar(
        x=['Actual', 'Predicted'],
        y=[actual_rul, predicted_rul],
        marker_color=[color, color],
        text=[f'{actual_rul:.0f}', f'{predicted_rul:.0f}'],
        textposition='auto',
    ))
    comparison.update_layout(
        title="Actual vs Predicted RUL",
        yaxis_title="Cycles",
        height=350,
        showlegend=False
    )

    # Fleet overview
    fleet_status = pd.DataFrame({
        'Engine': [f'E{i+1}' for i in range(len(y_pred))],
        'Predicted_RUL': y_pred,
        'Actual_RUL': y_test
    })
    fleet_status['Status'] = fleet_status['Predicted_RUL'].apply(lambda x: get_risk_status(x)[0])
    fleet_status['Color'] = fleet_status['Predicted_RUL'].apply(lambda x: get_risk_status(x)[1])

    fleet = go.Figure()
    fleet.add_trace(go.Scatter(
        x=fleet_status.index,
        y=fleet_status['Predicted_RUL'],
        mode='markers',
        marker=dict(
            size=12,
            color=fleet_status['Predicted_RUL'],
            colorscale=[[0, 'red'], [0.3, 'orange'], [1, 'green']],
            showscale=True,
            colorbar=dict(title="RUL")
        ),
        text=[f"Engine {i+1}: {r:.0f} cycles ({s})"
              for i, r, s in zip(fleet_status.index, fleet_status['Predicted_RUL'], fleet_status['Status'])],
        hoverinfo='text'
    ))
    fleet.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    fleet.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fleet.update_layout(
        title="Fleet-Wide RUL Overview",
        xaxis_title="Engine Index",
        yaxis_title="Predicted RUL (cycles)",
        height=400
    )

    return status_card, gauge, comparison, fleet

if __name__ == '__main__':
    print("\nStarting dashboard at http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
