from dash import Dash, dcc, Input, Output, html

import dash_bootstrap_components as dbc  
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gobj
import numpy as np
import random
import itertools
import matplotlib.pyplot as pp
import pandas as pd
from plotly import graph_objects as go
import networkx as nx
from dash_bootstrap_templates import load_figure_template
##DO up front data processing prior to any callbacks

recs = pd.read_csv('./data/NCF_recommendations.csv', header=0)

mba_recs = pd.read_csv('./data/product_recommendations_Market_Basket_Analysis.csv', header=0)

print(mba_recs.head())

unique_prods = mba_recs['product_id'].unique()

##Read in all data files
#dept = pd.read_csv("./data/departments.csv",header=0)

orders_with_uid = pd.read_csv("./data/orders.csv",header=0)

orders_static = pd.read_csv("./data/order_products__prior.csv")

prod_desc= pd.read_csv("./data/products.csv")



cpn_degrees = pd.read_csv("./data/cpn_product_degrees_full.csv")

cpn_degrees.drop(columns=['product_name'],inplace=True)

cpn_degrees_short = pd.read_csv("./data/cpn_product_degrees.csv")

cpn_degrees_short.drop(columns=['product_name'],inplace=True)

cpn_degrees_short.rename(columns={'degree':'short_degree'},inplace=True)
#merge orders with product descriptions to get names of products
named_orders = pd.merge(orders_static,prod_desc,"left",None,"product_id","product_id")

# print(named_orders.head())

grouped = named_orders.groupby(["department_id","product_id"]).agg(



    count=("product_id","count")



    
).reset_index()










#create dictionary of departments

#dept_dict = pd.Series(dept['department'].values,index=dept["department_id"]).to_dict()


#create dropdown options menu for list of departments

droptions = []


# for key, val in dept_dict.items():

#     droptions.append({'label':'{}'.format(val, val), 'value':key})
    


user_options = []

test_users = recs['user_id'].unique()

for u in test_users:
    user_options.append({'label':html.Span([f'User {u}'], style={'color': 'Black', 'font-size': 20}), 'value':u,'color':'black'})

cpn = pd.read_csv('./data/cpn_filtered.csv')

cpn = cpn[(cpn['Source'].isin(unique_prods)) & (cpn['Target'].isin(unique_prods))]

grid_size = 100

unique_nodes = pd.concat([cpn['Source'],cpn['Target']]).unique()

unique_df = pd.DataFrame(unique_nodes,columns=['Node'])

np.random.seed(31415)

unique_df['x'] = np.random.randint(0,grid_size,unique_df.shape[0]) 
unique_df['y'] = np.random.randint(0,grid_size,unique_df.shape[0])

unique_df = pd.merge(unique_df,prod_desc,'left',None,'Node','product_id')

unique_df = pd.merge(unique_df,cpn_degrees,'left',None,'product_id','product_id')

unique_df = pd.merge(unique_df,cpn_degrees_short,'left',None,'product_id','product_id')


cpn = pd.merge(cpn,unique_df,'left',None,'Source','Node')

cpn.rename(columns={'x':'Source_x','y':'Source_y'},inplace=True)

cpn.drop(columns=['Node'],inplace=True,axis=1)

cpn = pd.merge(cpn,unique_df,'left',None,'Target','Node')

cpn.rename(columns={'x':'Target_x','y':'Target_y'},inplace=True)

node_count = len(unique_nodes)

maxWeight = cpn['Weight'].max()

minWeight = cpn['Weight'].min()

names = unique_df["product_name"]



combined = "Degree (Full Graph): " + unique_df['degree'].astype(str) + "\nDegree (filtered): " + unique_df['short_degree'].astype(str) + "\nName: " + unique_df['product_name'].astype(str) 

node_trace = go.Scatter(
            x=unique_df['x'], y=unique_df['y'],
            #line=dict(width=2, color='#888'),
            hoverinfo='text',
            mode='markers',
            marker=dict(
                size=12,
                color='#debd62' #set color equal to desired color
                
            ), #TODO uncomment if template
            text=combined,
            
            customdata=unique_df['Node'],
            )



dataThings = []

#set edge and dropdown Colors to match theme

edgeColor = 'rgba(255, 255, 255, 0.5)' #uncomment for white

dropdownColor = '#5d6167'

#edgeColor = 'rgba(0, 0, 0, 0.5)' #uncomment for black

for i, row in cpn.iterrows():
    dataThings.append(

        
        ##weight edges width based on a normalized scale (0,1) + 1, (1,2) times 2 (2,4)
        go.Scatter(
            x=[row['Source_x'],row['Target_x']], y=[row['Source_y'],row['Target_y']],
            line=dict(width=((row['Weight']-minWeight)/(maxWeight-minWeight) + 1)*2, color=edgeColor), #TODO
            #line=dict(width=((row['Weight']-minWeight)/(maxWeight-minWeight) + 1)*2),
            hoverinfo='text',

            mode='lines',
            text=unique_df['product_name'])
    )

dataThings.append(node_trace)

#set template for FIGs:
template = load_figure_template("slate")

#create network graph
fig = go.Figure(data=dataThings,
                    layout=go.Layout(
                        # plot_bgcolor='rgba(0, 0, 0, 0)', #TODO
                        # paper_bgcolor='rgba(255, 255, 255, 0.65)', #TODO
                        title=dict(
                            text=f"CPN of 25 Products with Highest Weight<br><sup>Line weight is proportional to # of common baskets</sup>",
                            font=dict(
                                size=20,
                               #color='#debd62'
                            )
                        ),
                        
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=100),
                        template=template,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )





app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE],assets_folder='./data')

#set up image
image_filename = 'gaTech.png'  
image_src = app.get_asset_url(image_filename)

app.layout = dbc.Container([

    

    dbc.Row([
             dbc.Col(html.H1("Instacart Recommendation and Basket Analysis Dashboard",style={"text-align":"center"})),
             
             html.Img(src=image_src,style={"width":"121px","height":"50px","position":"absolute","top":"0px","right":"0px"})
             
             ]),
    
    dbc.Row([

        dbc.Col([html.H3("Hover a product node to learn more."),dcc.Graph(id="network",figure = fig,hoverData={'points':[{'customdata':13176}]},className='m-4')]),

        
        dbc.Col([html.H3("Powered by NCF"),dbc.Spinner(dcc.Graph(id="rec_chart",figure = go.Figure()))]),
    ]),


    dbc.Row([

        

        dbc.Col([

            html.P("Select a user to see user info."),
            dcc.Dropdown(id="selectUser", options=user_options, value=7510, style={'marginBottom': 50, 'background-color':dropdownColor}),
            
        ], width=2),
        
        dbc.Col([

            dbc.Spinner(dcc.Graph(id="mba_chart",figure=go.Figure()))
            ,html.H3("Powered by MBA model")
        ], width=5),
        dbc.Col(dbc.Spinner(dcc.Graph(id="bar_chart",figure = go.Figure())),width=5),

    ])
    

])


@app.callback(
   [Output(component_id="rec_chart", component_property="figure")],
   #[Output(component_id="rank_label", component_property="children")],
    #[Output(component_id="dept_label", component_property="children")],
    [Output(component_id="mba_chart", component_property="figure")],
    [Input(component_id="network",component_property="hoverData")],
    [Input(component_id='selectUser', component_property='value')]

)

def top_5_charts(hovered,selected_user):

    """
    This function creates a network graph of the top 5 products and their relationships in shared carts.

    :param df: dataframe containing product id's grouped by their product count

    :type verbose: pd.DataFrame


    :param deptDict: dictionary that defines what department id maps to what department name

    :type deptDict: dict


    :param selected_user: user for whom we wish to visualize

    :type selected_user: int

    :param selected_dept: department id of the department we want to make charts for

    :type selected_dept: int

    :param orders: the orders dataframe

    :type orders: Pandas.DataFrame



    """

    #use the below to store the hovered product id

    base_product = int(hovered['points'][0]['customdata'])

    #print(f'Hovered prod_id is: {hovered['points'][0]['customdata']}')
    
    
    #Create recommendation chart
    user_orders = orders_with_uid[orders_with_uid['user_id'] == selected_user]

    user_orders = pd.merge(user_orders,orders_static,"left",None,"order_id","order_id")

    prod_counts = user_orders.groupby('product_id').agg(

            count=("order_id","count") # just choose a col

        ).reset_index()

    # prod_counts.sort_values(by=['count'],inplace=True,ascending=False)
    # prod_counts.reset_index() #do this so we can get product ranking for user

    x_score = []

    y_score = []

    

    

    rec_output = recs[(recs['base_product_id'] == base_product) & (recs['user_id'] == selected_user)].reset_index()

    print(rec_output.head())

    x_score.append(rec_output['recommended_product_1_similarity_score'][0])
    x_score.append(rec_output['recommended_product_2_similarity_score'][0])
    x_score.append(rec_output['recommended_product_3_similarity_score'][0])

    #Append base product purchase count
    cnt = prod_counts[prod_counts['product_id'] == rec_output[f'base_product_id'][0]]['count'].reset_index()


    #Debug Statement
    #print(cnt)

    # if cnt.shape[0] == 0:
    #     y_score.append(0)
    # else:
    #     y_score.append(cnt['count'][0])

    for i in range(1,4):
        testMe = prod_counts[prod_counts['product_id'] == rec_output[f'recommended_product_{i}_id'][0]]['count'].reset_index()

        if testMe.shape[0] == 0:
            y_score.append(0)
        else:
            y_score.append(testMe['count'][0])



    base_name = prod_desc[prod_desc['product_id'] == base_product].reset_index()['product_name'][0]
    
    colors = ['#606464','#FFFFFF','#debd62']

    product_labels = []
    short_labels   = []

    for i in range(1,4):
        string = rec_output[f'recommended_product_{i}_name'][0]
        short = string
        if(len(string) > 18):
            short = string[:15]+"..."
        
        product_labels.append(string)
        short_labels.append(short)

    print(product_labels)
    print(short_labels)
    #i flipped x and y somewhere along the way and couldnt be bothered to change names
    xmin = min(y_score)
    xmax = max(y_score)
    ymin = min(x_score)
    ymax = max(x_score)

    traces = []
    for i in range(3):

        node_trace = go.Scatter(
            
            x=[y_score[i]], y=[x_score[i]],
            name=short_labels[i],
            mode='markers+text',
            #hoverinfo='text',
            marker=dict(color=colors[i]),
            text = [product_labels[i]],
            textposition='top center',
            
            
            )
        traces.append(node_trace)

    
    


    chart_margin = 0.25

    #node_trace.text = product_labels

    fig2 = go.Figure(data=traces,
                layout=go.Layout(

                    #plot_bgcolor='rgba(0, 0, 0, 0)', #TODO
                    #paper_bgcolor='rgba(0, 0, 0, 0)', #TODO
                    
                    title=dict(
                        text=f"<br>Users like user {selected_user} also bought these with {base_name}",
                        font=dict(
                            size=20
                        )
                    ),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=60),
                   template=template, #TODO delete if templates are broken
                    xaxis=dict(showgrid=True, zeroline=False, showticklabels=True,title=dict(text="Times Purchased by User"),range=[xmin-2,xmax+2]),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=True,title=dict(text="Similarity Score"),range=[ymin-chart_margin,ymax+chart_margin]))
                    
                    )

    #fig2.update_layout(font_color='white')


    #Update Bar chart of confidences and recommended products for a given base product with mba


    #get row of base_product

    confidences = []
    prod_names = []


    mba_row = mba_recs[mba_recs['product_id'] == base_product].reset_index()

    for i in range(1,4):
        c = mba_row[f'confidence_{i}']
        print(c)
        c = c[0]
        if c == np.nan:
            break
        confidences.append(c)
        prod_names.append(mba_row[f'recommended_product_{i}'][0])
    


    bar = go.Bar(x=prod_names,
                y=confidences,
                #marker=dict(color='#4f7fc1'), #TODO uncomment if template fails
                #textfont=dict(size=13,color='white') #TODO uncomment if template fails
                 
                     
    )

    bar_fig = go.Figure(data=[bar],
                
                layout=go.Layout(
                    
                    #plot_bgcolor='rgba(0, 0, 0, 0)',
                    #paper_bgcolor='rgba(0, 0, 0, 0)',
                    title=f'Products often bought with {base_name}',
                    # title=dict(
                    #    text=f'Products often bought with {base_name}',
                    #    font=dict(
                    #        size=16,
                    #        color='white'
                    #    )
                    # ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    template=template, #TODO delete if template broekn
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,title=dict(text="Name")),
                    yaxis=dict(showgrid=True, zeroline=False, showticklabels=True,title=dict(text="Recommendation Confidence")))
    )
    

    # #add text describing department, product rank for user
    # find_rank = prod_counts[prod_counts['product_id'] == base_product]
    # rank_text = 'User has not purchased this!'
    # if find_rank.shape[0] != 0:
    #     rank_text = f"This is user's {find_rank.index[0]}th ranked product"

    # dept = dept_dict[prod_desc[prod_desc['product_id'] == base_product].reset_index()['department_id'][0]]

    # dept_text = base_name + ' are from the ' + dept + ' department.'

    
    return (fig2,bar_fig)
    #return (fig2,rank_text,dept_text)

#Call back for user specific changes only
@app.callback(
   [Output(component_id="bar_chart", component_property="figure")],
    [Input(component_id='selectUser', component_property='value')]

)

#function called when user is updated
def user_top_prods(selected_user):
    user_orders = orders_with_uid[orders_with_uid['user_id'] == selected_user]

    user_orders = pd.merge(user_orders,orders_static,"left",None,"order_id","order_id")

    prod_counts = user_orders.groupby('product_id').agg(

            count=("order_id","count") # just choose a col

        ).reset_index()

    prod_counts.sort_values(by=['count'],inplace=True,ascending=False)
    
    top_5_prods = prod_counts.head()

    top_5_prods = pd.merge(top_5_prods,prod_desc,'left',None,"product_id","product_id")

    top_5_prods.rename(columns={'product_name':'Product Name','count':'Times Ordered'}, inplace=True)

    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'

    top_5_prods['Short Name'] = [x[:20] for x in top_5_prods['Product Name']]

    bar = go.Bar(x=top_5_prods['Short Name'],
                 y=top_5_prods['Times Ordered'],
                 #marker=dict(color='#4f7fc1'), #TODO
                 #textfont=dict(size=13,color='white') #TODO
                 
                     
    )
    
    #make background transparent
    # bar.update_layout(

    #     {
    #     "paper_bgcolor": "rgba(0, 0, 0, 0)",
    #     "plot_bgcolor": "rgba(0, 0, 0, 0)",
        
    #     }
    # )

    #bar.update_traces(marker_color="#4f7fc1")
    #bar.update_traces(textfont_color="white")

    bar_fig = go.Figure(data=[bar],
                
                layout=go.Layout(
                    
                    # plot_bgcolor='rgba(0, 0, 0, 0)', #TODO
                    # paper_bgcolor='rgba(0, 0, 0, 0)', #TODO

                    # title=dict(
                    #    text=f'Top 5 Products Purchased by User {selected_user}',
                    #    font=dict(
                    #        size=16,
                    #        color='white'
                    #    )
                    # ),
                    title=f'Top 5 Products Purchased by User {selected_user}',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    template=template,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,title=dict(text="Name")),
                    yaxis=dict(showgrid=True, zeroline=False, showticklabels=True,title=dict(text="Times Purchased")))
                    )
    #bar_fig.update_layout(font_color="white") #TODO
    return (bar_fig,)

#### RUN APP

if __name__=='__main__':
    app.run_server(port=8051)
