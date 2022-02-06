import pandas    as pd
import streamlit as st
import numpy     as np
import folium
#import geopandas
import plotly.express as px


from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster
from datetime         import datetime


st.set_page_config(layout='wide')


@st.cache (allow_output_mutation=True)
def get_data (path):
    data = pd.read_csv(path)

    return data

# @st.cache (allow_output_mutation=True)
# def get_geofile (url):
#     geofile = geopandas.read_file(url)
#
#     return geofile


def set_feature (data):
    # Add new features
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview_data (data):
    st.title('Data Overview')

    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)  # Colunas
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())  # Linhas

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]

    else:
        data = data.copy()

    st.dataframe(data)

    c1, c2 = st.columns((1, 1))


    # Average Metrics

    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL DE HOUSES', 'PRICE', 'SQFT_LIVING', 'PRICE/M2']

    c1.header('Average Metrics')
    c1.dataframe(df, height=600)

    # Statistic Descriptive

    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    medi_ = pd.DataFrame(num_attributes.apply(np.median))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))
    std = pd.DataFrame(num_attributes.apply(np.std))

    df1 = pd.concat([media, medi_, max_, min_, std], axis=1).reset_index()

    df1.columns = ['ATTRIBUTES', 'MEDIA', 'MEDIANA', 'MAXIMO', 'MINIMO', 'STD']

    c2.header('Statistic Descriptive')
    c2.dataframe(df1, height=600)

    return None

def portfolio_density(data):
    st.title('Region Overview')
    c3, c4 = st.columns((1, 1))
    c3.header('Portifolio Density')

    df = data.sample(10)

    # Base Map - Folium

    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='PRICE R${0} on: {1} Features: {2} sqft, {3} bedrooms, {4} Bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'], row['sqft_living'], row['bedrooms'],
                          row['bathrooms'], row['yr_built'])).add_to(marker_cluster)

    with c3:
        folium_static(density_map)

    # Region Price Map

    # c4.header('Price Density')
    #
    # df = data[['price', 'zipcode']].groupby ('zipcode').mean().reset.index()
    # df.columns = ['ZIP', 'PRICE']
    # #df = df.sample(100)
    # geofile = geofile[geofile['ZIP'].isin (df['ZIP'].tolist ())]
    #
    # region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)
    #
    # region_price_map.choropleth(data = df, geodata = geofile , columns = ['ZIP', 'PRICE'],key_on='feature.properties.ZIP',
    #                             fill_color= 'YlOrRD', fill_opacity = 0.7, line_opacity = 0.2, legend_name = 'AVG PRICE' )
    #
    # with c4:
    #     folium_static (region_price_map)

    return None

def commercial_distribution  (data):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')
    st.sidebar.subheader('Select Max Year Built ')



    # filter
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())
    f_yearbuilt = st.sidebar.slider('Year Bulit', min_year_built, max_year_built, min_year_built)
    st.header('Average Price per Year Built')

    # data select
    df = data.loc[data['yr_built'] < f_yearbuilt]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average Price  per day
    st.header('Average price per day')
    st.sidebar.subheader('Select Max date')

    # filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')
    f_date = st.sidebar.slider('Date Max', min_date, max_date, min_date)

    # data select
    data['date'] = pd.to_datetime(data['date'])
    df = data[['date', 'price']].groupby('date').mean().reset_index()
    df = data.loc[data['date'] < f_date]

    # plot
    df = df[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Price Distribution')
    st.subheader('Selec Max Price')

    # filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # data filtering
    f_price = st.sidebar.slider('Max Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # plot

    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution (data):
    st.title('House Attributes')
    st.sidebar.title('Attributes Options')

    data['floors'] = data['floors'].astype(int)
    # filter
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))

    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

    f_floors = st.sidebar.selectbox('Max Number of Floors', data['floors'].unique())

    f_waterview = st.sidebar.checkbox('Only Houses With Water View', )

    c5, c6 = st.columns((1, 1))

    # data select

    # House per bedrooms
    c5.header('Houses per bedrooms')
    df = data.loc[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c5.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c6.header('Houses per bathrooms')
    df = data.loc[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c6.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns((1, 1))
    # House per floors
    c1.header('Houses per floors')
    df = data[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    st.plotly_chart(fig, use_container_width=True)

    # #House per waterview
    if f_waterview:
        df = data[data['waterfront'] == 1]

    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=19)
    st.plotly_chart(fig, use_container_width=True)

    return None


if __name__ == '__main__':
    #ETL

    #data extraticon
    # get data
    path = 'datasets/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    # geofile = get_geofile (url

    #data trsnformation
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    data = set_feature(data)
    overview_data (data)
    portfolio_density (data)
    commercial_distribution (data)
    attributes_distribution (data)





