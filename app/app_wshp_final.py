import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
import plotly.figure_factory as ff



from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

###############################################################################################################
## Loading Wine data from scikit-learn and dataframe creation
data = load_wine()
df = pd.DataFrame(
    data=np.concatenate((data['data'], np.reshape(data['target'], (data['target'].shape[0],1))), axis = 1),  
    columns=data['feature_names']+['WineClass']
    )
###############################################################################################################


app = dash.Dash(__name__)
app_color = {"graph_bg": '#1a1b1d', "graph_bg1": "#222629", "graph_line": "#86C232", 'heading': '#86C232'}

app.layout = html.Div(
    ######## This Div is the main comntainer of all the dash and html components; following list contains all its children
    [
        html.Div(
            ####### This is the App Header Div, contains title and description
            [
                
                html.Div(
                    [
                        html.H4("WINE CLASSIFICATION", 
                        # className="app__header__title"
                        style = {'color' : 'white', 'font-size': '30px', 'padding': '0px 0px 0px 0px', 'margin': '0px 0px 0px 0px'}
                        ),
                        html.P(
                            "Exploratory Data Analysis and Random Forest Classification Modeling of Wines",
                            style = {'color': app_color['heading'], 'font-size': '15px', 'padding': '0px 0px 0px 0px', 'margin': '5px 0px 0px 0px'}
                            # className="app__header__title--grey",
                        ),
                    ],
                    className = 'six columns'
                ),
                html.Div(
                    [
                        html.Img(id="logo", src=app.get_asset_url("lighteninginabottle.png"),
                        style = {'width':'120px'})
                        
                    ],
                    className = 'six columns',
                    style = {'text-align': 'right'}
                ),    
            ],
            className = 'row',
            style = {'display': 'flex', 'margin-right': '50px', 'border-bottom': '1px solid gray'}
            # className="app__header",
            
        
        ),
        
        html.Div(
            #######this will contain two panes of equal width , aligned horizontally, 1st for EDA, 2nd for ML
            [
                html.Div(
                    ########this will have two panes aligned vertically, top: Histogram, bot: X-plot
                    [
                        html.Div(
                                [
                                    html.H5('Exploratory Data Analysis')
                                ],
                                style = {'color' : 'white',  'text-align': 'center',  'padding': '0px 0px 0px 0px', 'margin': '0px 0px 0px 0px'}
                        ),

                        html.Div(
                            ######### Histogram
                            [
                                html.Div(
                                    [
                                        html.H6('Feature Distribution',
                                        style = {'color' : 'gray', 'font-size': '17px', 'margin': '10px 0px 0px 20px'}),

                                    ],
                                    style = {'margin': '0px 20px 10px 0px'}
                                    
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id = 'feat-drpdn',
                                            options=[{'label': df.columns[:-1][i], 'value': i} for i in range(len(df.columns[:-1]))],
                                            value = 0,
                                            style = {'width': '55%', 'margin': '0px 0px 0px 10px','background-color': app_color["graph_bg"]}
                                        ),
                                    ]
                                ),
                                

                                dcc.Graph(
                                    id="feat-histogram",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                    
                                ),
                            ],
                            style = {'display': 'flex', 'flex-direction': 'column', 'background-color': app_color["graph_bg"], 'margin': '0px 0px 10px 0px'}
                        ),
                        html.Div(
                            [
                               html.Div(
                                    [
                                        html.H6('Feature Correlation',
                                        style = {'color' : 'gray', 'font-size': '17px', 'margin': '10px 0px 0px 20px'}),

                                    ],
                                    style = {'margin': '0px 20px 10px 0px'}
                                    
                                ),

                                html.Div(
                                    ######## This Div Contains two dropdowns for feature selection for x-plot
                                    [
                                        html.Div(
                                            dcc.Dropdown(
                                            id = 'feat-xplot-1',
                                            options=[{'label': df.columns[:-1][i], 'value': i} for i in range(len(df.columns[:-1]))],
                                            value = 0,
                                            style = {'margin': '0px 0px 0px 10px', 'background-color': app_color["graph_bg"]}
                                            ),
                                            className = 'six columns'
                                        ),
                                        html.Div(
                                            dcc.Dropdown(
                                            id = 'feat-xplot-2',
                                            options=[{'label': df.columns[:-1][i], 'value': i} for i in range(len(df.columns[:-1]))],
                                            value = len(df.columns[:-1])-1,
                                            style = {'margin': '0px 10px 0px 0px', 'background-color': app_color["graph_bg"]}
                                            ),
                                            className = 'six columns'
                                        )
                                        
                                    ],
                                    className = 'row'
                                ),
                                dcc.Graph(
                                    id="feat-xplot",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                    
                                ),  
                            ],
                            style = {'display': 'flex', 'flex-direction': 'column', 'background-color': app_color["graph_bg"], 'margin': '10px 0px 0px 0px'}
                        )
                    ],
                    className = 'six columns',
                    style = {'margin': '0px 10px 0px 0px'}


                ),

                html.Div(
                    ######## This will have four panes aligned vertically, 1st: Model training params, 2nd: feature importance, 3rd: ROC, 4th: User Prediction
                    [
                        html.Div(
                            [
                                html.H5('Classification using Random Forest')
                            ],
                            style = {'color' : 'white',  'text-align': 'center',  'padding': '0px 0px 0px 0px', 'margin': '0px 0px 0px 0px'}
                        ),
                        
                        html.Div(
                            ############# Training Parameters Container
                            
                            [
                                html.Div(
                                    ############# Training Parameter Header
                                    [
                                        html.H6('Training Parameters',
                                        style = {'color' : 'gray', 'font-size': '17px', 'margin': '10px 0px 0px 20px'}),

                                    ],
                                    
                                ),
                                html.Div(
                                    ############### Training Parameter Input, has 4 components aligned in a row
                                    [
                                        html.Div(
                                            [
                                                html.Label("Test Split", style = {'color': 'white', 'text-align': 'center'}),
                                                dcc.Input(
                                                id="test-size", type="number", placeholder="Test Split",min=0.05, max=0.5, step=0.05, value = 0.2,
                                                style = { 'text-align': 'right', 'width': '100%','padding': '0px 5px 0px 10px', 'color':'white', 'background-color': app_color["graph_bg"]}
                                            )],
                                            className = 'three columns',
                                            style={'fontColor': 'white'}
                                            
                                        ),
                                        html.Div(
                                            [
                                                html.Label("No of Estimators", style = {'color': 'white', 'text-align': 'center'}),
                                                dcc.Input(
                                                id="n-estimators", type="number", placeholder="No of Estimators",min=10, max=150, step=10,value = 90,
                                                style = { 'text-align': 'right', 'width': '100%','padding': '0px 5px 0px 5px', 'color':'white', 'background-color': app_color["graph_bg"]}
                                            )],
                                            className = 'three columns',
                                            style={'fontColor': 'white'}
                                        
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Min Samples Split", style = {'color': 'white', 'text-align': 'center'}),
                                                dcc.Input(
                                                id="min-samples-split", type="number", placeholder="Min Samples Split",min=0.05, max=0.5, step=0.05, value = 0.2,
                                                style = { 'text-align': 'right', 'width': '100%','padding': '0px 5px 0px 5px', 'color':'white', 'background-color': app_color["graph_bg"]}
                                            )],
                                            className = 'three columns',
                                            style={'fontColor': 'white'}
                                        
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Train model", style = {'color': 'white', 'text-align': 'center'}),
                                                html.Button('Run', id='model-training-btn', n_clicks=0,
                                                style = { 'font-size':'15px','width': '100%','padding': '0px 10px 0px 5px', 'background-color': app_color["heading"]})

                                            ],
                                            className = 'three columns',
                                        
                                        ),
                                        
                                    ],
                                    className = 'row',
                                    style = { 'padding': '10px 10px 10px 10px'}
                                ),
                                

                            ],
                            style = {'display': 'flex', 'flex-direction': 'column', 'background-color': app_color["graph_bg"], 'margin': '0px 0px 10px 0px'}
                        ),
                        
                        html.Div(
                            ###################### Feature Importance
                            [
                                html.Div(
                                    ############# Feat Imp Header
                                    [
                                        html.H6('Feature Importance',
                                        style = {'color' : 'gray', 'font-size': '17px', 'margin': '10px 0px 0px 20px'}),

                                    ],
                                    
                                ),
                                dcc.Graph(
                                    id="feat-importance-plot",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                            height = 270
                                        )
                                    ),
                                    style = {'margin': '0px 0px 0px 0px'}
                                    
                                ), 
                            ],
                            style = {'display': 'flex', 'flex-direction': 'column', 'background-color': app_color["graph_bg"], 'margin': '0px 0px 10px 0px'}
                        ),
                        html.Div(
                            ############### ROC
                            [
                                html.Div(
                                    ############# Feat Imp Header
                                    [
                                        html.H6('Classification ROC-AUC',
                                        style = {'color' : 'gray', 'font-size': '17px', 'margin': '10px 0px 10px 20px'}),

                                    ],
                                    
                                ),
                                html.Div(
                                    ############ Div to contain all accuracy and roc values
                                    [
                                        html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label('Training Accuracy', style = {'color': 'white', 'text-align':'right'})
                                                ],
                                                className = 'four columns'
                                            ),
                                            html.Div(
                                                [
                                                   html.Label(id = 'training-acc-label', style = {'color': app_color['heading'], 'text-align':'left'}) 
                                                ],
                                                className = 'two columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label('Test Accuracy', style = {'color': 'white', 'text-align':'right',})
                                                ],
                                                className = 'four columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(id = 'test-acc-label', style = {'color': app_color['heading'],  'text-align':'left', }) 
                                                ],
                                                className = 'two columns'
                                            )
                                        ],
                                        className = 'row'
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label('Macro One vs One AUC ROC', style = {'color': 'white', 'text-align':'right'})
                                                ],
                                                className = 'four columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(id = 'macro-ovo-auc-roc-label', style = {'color': app_color['heading'],  'text-align':'left'})
                                                ],
                                                className = 'two columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label('Weighted One vs One AUC ROC', style = {'color': 'white', 'text-align':'right'})
                                                ],
                                                className = 'four columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(id = 'weighted-ovo-auc-roc-label', style = {'color': app_color['heading'],  'text-align':'left'})
                                                ],
                                                className = 'two columns'
                                            )
                                        ],
                                        className = 'row'
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label('Macro One vs Rest AUC ROC', style = {'color': 'white', 'text-align':'right'})
                                                ],
                                                className = 'four columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(id = 'macro-ovr-auc-roc-label', style = {'color': app_color['heading'],  'text-align':'left'})
                                                ],
                                                className = 'two columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label('Weighted One vs Rest AUC ROC', style = {'color': 'white', 'text-align':'right'})
                                                ],
                                                className = 'four columns'
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(id = 'weighted-ovr-auc-roc-label', style = {'color': app_color['heading'],  'text-align':'left'})
                                                ],
                                                className = 'two columns'
                                            )
                                        ],
                                        className = 'row',
                                        style = {'margin': '0px 0px 10px 0px'}
                                    ),
                                    ]
                                    
                                    
                                    
                                )
                                
                            
                            ],
                            style = {'display': 'flex', 'flex-direction': 'column', 'background-color': app_color["graph_bg"], 'margin': '0px 0px 10px 0px'}
                        ),
                        html.Div(
                            [
                                html.Div(
                                    ############# User Testing for New Data
                                    [
                                        
                                        html.H6('Classify New Data',
                                        style = {'color' : 'gray', 'font-size': '17px', 'margin': '10px 0px 10px 10px'}),

                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        dash_table.DataTable(
                                            
                                                            id='table-classify-data', columns=[{"name": i, "id": i} for i in df.drop('WineClass', axis = 1, inplace = False).columns],
                                                            data =df.drop('WineClass', axis = 1, inplace=False).sample(5).to_dict('records'),
                                                            style_table={'overflowX': 'auto', 'color': app_color['graph_bg']},
                                                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                                                            style_cell={
                                                                'backgroundColor': 'rgb(50, 50, 50)',
                                                                'color': 'white'
                                                            },
                                                            editable=True,
                                                        )
                                                    ],

                                                    className = 'ten columns'
                                                ),
                                                html.Div(
                                                    [
                                                        dash_table.DataTable(
                                            
                                                            id='table-classify-target', columns=[{"name": 'WineClass', "id": 'WineClass'}],
                                                            data = [{'WineClass': [None]}]*5,
                                                            style_table={'color': app_color['graph_bg']},
                                                            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'text-align':'center'},
                                                            style_cell={
                                                                'backgroundColor': 'rgb(50, 50, 50)',
                                                                'color': app_color['heading'],
                                                                'text-align':'center'
                                                            },
                                                            editable=False,
                                                        )
                                                    ],
                                                    className = 'two columns'
                                                )

                                            ],
                                            className = 'row',
                                            style = {'margin': '10px 0px 0px 0px'}
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                [
                                                    html.Button('Classify', id='classify-new-btn', n_clicks=0,
                                                        style = { 'font-size':'15px','padding': '0px 0px 0px 0px', 'background-color': app_color["heading"]}
                                                    ), 
                                                ],
                                                className = 'two Columns',
                                                style = {'margin': '0px 0px 0px 0px', 'width': '20%'}
                                                ),
                                                html.Div(
                                                    
                                                    [html.Label(id = 'classify-new-status', style = {'color':'white', 'font-size':'15px', 'width': '100%'})],
                                                className = 'ten columns',
                                                style = {'margin': '0px 0px 0px 0px'}
                                                ),
                                            ],
                                            style = { 'padding': '10px 10px 10px 0px'}, 
                                            className = 'row'
                                        )
                                        
                                        

                                    ],
                                    style = {'display': 'flex', 'flex-direction': 'column','margin': '0px 20px 0px 20px'}
                                    
                                ),
                                
                            ],
                            style = {'display': 'flex', 'flex-direction': 'column', 'background-color': app_color["graph_bg"], 'margin': '0px 0px 10px 0px'}
                        )

                    ],
                    className = 'six columns',
                    style = {'margin': '0px 0px 0px 10px'}

                )

            ],
            className="rows",
            style = {'display': 'flex', 'margin-top': '10px'}
        )

    ],
    # className="app__container",
    style = { 'margin': '3% 5%'}

)

def train_RF_model(test_size, n_estimators, min_smpls_split):
    data_X, data_y = df.drop(['WineClass'], axis = 1), df[['WineClass']]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=test_size, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_smpls_split, random_state=22)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    

    y_test_bin = label_binarize(y_test, classes = [0, 1, 2])
    y_prob = model.predict_proba(X_test)

    macro_roc_auc_ovo = roc_auc_score(y_test_bin, y_prob, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(y_test_bin, y_prob, multi_class="ovo", average="weighted")
    macro_roc_auc_ovr = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")
    filename = 'models\wine_model_for_dash.sav'
    pickle.dump(model, open(filename, 'wb'))


    return model, (train_acc, test_acc), (macro_roc_auc_ovo, weighted_roc_auc_ovo, macro_roc_auc_ovr, weighted_roc_auc_ovr)

def classify_new_data(test_X_new):
    
    filename = 'models\wine_model_for_dash.sav'
    model = pickle.load(open(filename, 'rb'))
    y_new = model.predict(test_X_new)

    return y_new

@app.callback(
    Output(component_id='feat-histogram', component_property='figure'),
    [Input(component_id='feat-drpdn', component_property='value')]
)
def update_feat_histogram(input_value):
    
    ft_max, ft_min = max(df.iloc[:,input_value]), min(df.iloc[:,input_value])
    bin_size = (ft_max-ft_min)/30
    hist_data = [df[df['WineClass']==wine_cls].iloc[:,input_value] for wine_cls in df['WineClass'].unique()]

    group_labels = ['Class 1', 'Class 2', 'Class 3']
    colors = ['rgb(77,92,56)','rgb(112, 155,51)','rgb(150, 253, 5)']

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(
        hist_data, 
        group_labels, 
        colors=colors, 
        bin_size=bin_size, 
        show_rug=False)
    

    fig.update_layout(
        barmode='stack',
        height=360,
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        xaxis={
            "title": df.columns[input_value],
            "showgrid": False,
            "showline": False,
            "fixedrange": True,
        },
        yaxis={
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "title": "Number of Samples",
            "fixedrange": True,
        },
        autosize=True,
        )
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    return fig

@app.callback(
    Output(component_id='feat-xplot', component_property='figure'),
    [Input(component_id='feat-xplot-1', component_property='value'),
    Input(component_id='feat-xplot-2', component_property='value')]
)
def update_features_xplot(ft1, ft2):
    fig = go.Figure()
    colors = ['rgb(77,92,56)','rgb(112, 155,51)','rgb(150, 253, 5)']
    sizes = [5,8,11]

    for i, wine_cls in enumerate(df['WineClass'].unique()):
        fig.add_trace(go.Scatter(
                    x=df[df['WineClass']==wine_cls].iloc[:,ft1], 
                    y=df[df['WineClass']==wine_cls].iloc[:,ft2],
                    mode='markers',
                    marker_color=colors[i],
                    marker_size=sizes[i],
                    marker_line_color='white',
                    marker_line_width = 1,
                    name = 'Class {}'.format(i+1)
                    )
        )
    
    fig.update_layout(
        height=360,
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        xaxis={
            "title": df.columns[ft1],
            "showgrid": False,
            "showline": False,
            "fixedrange": True,
        },
        yaxis={
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "title": df.columns[ft2],
            "fixedrange": True,
        },
        autosize=True,
        )
    return fig
    
@app.callback(
    [Output(component_id='feat-importance-plot', component_property='figure'),
    Output(component_id='training-acc-label', component_property='children'),
    Output(component_id='test-acc-label', component_property='children'),
    Output(component_id='macro-ovo-auc-roc-label', component_property='children'),
    Output(component_id='weighted-ovo-auc-roc-label', component_property='children'),
    Output(component_id='macro-ovr-auc-roc-label', component_property='children'),
    Output(component_id='weighted-ovr-auc-roc-label', component_property='children')],
    [
        Input(component_id='model-training-btn', component_property='n_clicks')
    ],
    [
        State(component_id='test-size', component_property='value'),
        State(component_id='n-estimators', component_property='value'),
        State(component_id = 'min-samples-split', component_property='value')
    ]
)
def update_feature_importance_roc_plot(n_clicks, test_size, n_estimators, min_smpls_split):
    if n_clicks > 0:
        if not (test_size==None or n_estimators==None or min_smpls_split==None):
            model, acc, auc_roc = train_RF_model(test_size=test_size, n_estimators=n_estimators, min_smpls_split=min_smpls_split)
            
            train_acc = np.round(acc[0], 3)
            test_acc = np.round(acc[1], 3)

            macro_roc_auc_ovo, weighted_roc_auc_ovo, macro_roc_auc_ovr, weighted_roc_auc_ovr = np.round(auc_roc[0], 3), np.round(auc_roc[1],3), np.round(auc_roc[2], 3) ,np.round(auc_roc[3],3) 

            indices_imp_fts = np.argsort(model.feature_importances_)[::-1][:5]
            

            fig = go.Figure(go.Bar(
                            x=model.feature_importances_[indices_imp_fts][::-1],
                            y=list(df.columns[indices_imp_fts])[::-1],
                            width = 0.7,
                            orientation='h',
                            marker=dict(
                                color=  app_color['heading'],
                                line=dict(color='black', width=2
                                    )
                                )
                            )
            )

            fig.update_layout(
                height=270,
                plot_bgcolor=app_color["graph_bg"],
                paper_bgcolor=app_color["graph_bg"],
                font={"color": "#fff"},
                xaxis={
                    "title": "Feature importance",
                    "showgrid": False,
                    "showline": False,
                    "fixedrange": True,
                },
                yaxis={
                    "showgrid": False,
                    "showline": False,
                    "zeroline": False,
                    # "title": df.columns[ft2],
                    "fixedrange": True,
                },
                autosize=True,
                margin=dict(l=10, r=10, t=30, b=20)
                )
            return fig, train_acc, test_acc, macro_roc_auc_ovo, weighted_roc_auc_ovo, macro_roc_auc_ovr, weighted_roc_auc_ovr

        else:
            raise PreventUpdate
  
    else:
        raise PreventUpdate


@app.callback(
    [Output(component_id='table-classify-target', component_property='data'),
    Output(component_id= 'classify-new-status', component_property='children')],
    [Input(component_id='classify-new-btn', component_property='n_clicks')],
    [State(component_id='table-classify-data', component_property='data')]
    
)

def update_new_classification_table(n_clicks, test_data):
    if n_clicks>0:

        data = [list(dicton.values()) for dicton in test_data] 
        flat_list = [item for sublist in data for item in sublist]

    
        if ('' in flat_list) or (None in flat_list):
            return [{'WineClass':'None'}]*5, 'Error: Some values are missing in the table, please fill them and try again!'
        else:
            y_new = classify_new_data(np.array(data))
            classes = ['Class1', 'Class 2', 'Class 3']
            return [{'WineClass': classes[int(y_new[i])]} for i in range(len(y_new))], ''
    else:
        raise PreventUpdate



if __name__ == "__main__":
    app.run_server(debug=True)