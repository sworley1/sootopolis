import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    import altair as alt
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer, PolynomialFeatures
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.compose import ColumnTransformer
    from vega_datasets import data
    import pandas as pd
    import narwhals as nw
    import numpy as np

    import sootopolis as soot


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Sootopolis""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""

    ## Example 1
    Sootopolis provides a function to visualize the drivers of Linear Regression like model. This notebook provides a new examples of how to use it.    

    Let's start with the cars dataset and fit a Linear Regression to predict miles per gallon for a vehicle.
    """
    )
    return


@app.cell
def _():
    # Load the dataset into polars
    cars = pl.from_pandas( data.cars() )
    cars
    return (cars,)


@app.cell
def _(cars):
    alt.Chart(cars).mark_circle().encode(
        alt.Y("Miles_per_Gallon"),
        alt.X("Displacement"),
        alt.Color("Acceleration:Q")
    ).properties(title="Vega Cars Dataset: Negative Relationship between Displacement and MPG")
    return


@app.cell
def _(cars):
    # Fit the Linear Regression Model
    sample = cars.filter(pl.col("Miles_per_Gallon").is_not_null()).filter(pl.col("Horsepower").is_not_null() )
    X = sample.select(['Displacement','Acceleration','Weight_in_lbs' ])
    y = sample.select('Miles_per_Gallon')
    lr = LinearRegression().fit(X,y)

    return X, lr, sample


@app.cell(hide_code=True)
def _():
    mo.md(r"""`sootopolis` provides a function `setup_data` which shows the data input into the waterfall chart.""")
    return


@app.cell
def _(X, lr, sample):
    setup_data_ = soot.setup_data(
        model=lr,point_to_predict=sample[42],regressor_labels=X.columns
    )
    setup_data_
    return (setup_data_,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""`sootopolis` provides a function to take the data loaded from `setup_data` and plot the waterfall. `plot_altair_waterfall` returns a `altair` chart which can be customized in the `.properties()` method.""")
    return


@app.cell
def _(setup_data_):
    soot.plot_altair_waterfall(source=setup_data_).properties(
        title='Linear Regression for Predicting MPG'
    )
    return


app._unparsable_cell(
    r"""
    `sootopolis` also provides a function which returns the altair waterfall chart without going through the intermediate step of `setup_data` 
    """,
    name="_"
)


@app.cell
def _(X, lr, sample):
    soot.plot_waterfall(
        model=lr, # takes model as input
        point_to_predict=sample[42], # single row in the dataframe
        regressor_labels=X.columns # regressor labels
    ).properties(
        title='Linear Regression for Predicting MPG'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### Example for Using `sootopolis` to drive insights. 

    Thanks to [`marimo`](https://marimo.io/), you can use interaction to understand what may be driving a given estimate.  

    Click on point in the top altair chart and the waterfall chart will appear below.
    """
    )
    return


@app.cell
def _(cars):
    testbase = alt.Chart(cars)

    testCircles = testbase.mark_circle().encode(
        alt.X("Acceleration").scale(zero=False),
        alt.Y("Miles_per_Gallon")
        ,color=alt.Color('Origin:N')
        ,tooltip=[
            'Origin',
            'Name',
            alt.Tooltip("Acceleration"),
            alt.Tooltip('Miles_per_Gallon')
        ]
    )
    moTestChart = mo.ui.altair_chart(testCircles)

    return (moTestChart,)


@app.cell
def _(bottomChart, moTestChart):
    mo.vstack([
        moTestChart,
        bottomChart
    ])
    return


@app.cell
def _(X, lr, moTestChart):
    if len(moTestChart.value)==1:
        X_sample = moTestChart.value.select(X.columns)
        #y_pred = lr.predict(X_sample)
        #sourc3 = setup_data(lr, X_sample, X.columns)
        #bottomChart = plot_altair_waterfall(sourc3).properties(
        #    height=200,title='linear regression equation for predicting miles per gallon'
        #)
        bottomChart = soot.plot_waterfall(lr, X_sample, X.columns).properties(
            height=200,title='linear regression equation for predicting miles per gallon'
        )
    else:
        print(len(moTestChart.value))
        bottomChart = alt.Chart()
    return (bottomChart,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Example 2 - Gapminder and narwhals

    And using `narwhals`
    """
    )
    return


@app.cell
def _():
    gm = nw.from_native(  data.gapminder_health_income() ) # load the dataset
    alt.Chart(gm).mark_circle().encode(
        alt.X("income").scale(zero=False),
        alt.Y("health").scale(zero=False),
        #alt.Size("population")
    ).interactive()
    return (gm,)


@app.cell
def _(gm):
    gm2 = gm.with_columns(
         nw.col('health').log().alias("ln_health"),
         nw.col("income").log().alias("ln_income")
    )
    base = alt.Chart(
        gm2  
    )
    circles = base.mark_circle().encode(
        alt.X("ln_income").scale(zero=False),
        alt.Y("health").scale(zero=False),
        alt.Size('population').scale(zero=False),
        tooltip=['country']
    )

    gm_chart = (
        circles + base.transform_regression("ln_income", "health").mark_line(color='red').encode(
            alt.X("ln_income"),
            alt.Y("health")
        )
    ).properties(
        title="Using LN(Income) to predict Health"
    )

    mo_gm_chart = mo.ui.altair_chart(gm_chart)
    return gm2, mo_gm_chart


@app.cell
def _(gm2):
    XY_gm = gm2.select(['health','ln_income'])
    X2 = XY_gm.select(['ln_income'])
    y2 = XY_gm.select(['health'])
    lr2 = LinearRegression().fit(
        X2,y2
    )
    return X2, lr2


@app.cell
def _(X2, lr2, mo_gm_chart):
    if len(mo_gm_chart.value) == 1:
        soot2 = soot.plot_waterfall(
            model=lr2
            ,point_to_predict=mo_gm_chart.value
            ,regressor_labels=X2.columns
        ) 
        soot2 = mo.ui.altair_chart(soot2)
    else:
        soot2 = None
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""Again we can use interaction within altair and marimo to see interpretation of a given point.""")
    return


@app.cell
def _(X2, lr2, mo_gm_chart):
    mo.vstack([
        mo_gm_chart,
        mo_gm_chart.value,
        soot.setup_data(model=lr2, point_to_predict=mo_gm_chart.value, regressor_labels=X2.columns)
        ,
        soot.plot_waterfall(model=lr2, point_to_predict=mo_gm_chart.value, regressor_labels=X2.columns).properties(
            height=200
        )
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""## Example 3: Using scikit-learn pipeline""")
    return


@app.cell
def _():
    movies = data.movies() #.to_native()
    movies =(
        movies
        .dropna(subset=['Production_Budget','IMDB_Rating','IMDB_Votes', 'Rotten_Tomatoes_Rating','US_Gross'])
        #.insert(-1, 'Before_1990', np.where( movies['Release_Date'].str.slice(-4) < '1990', 1, 0) )
        .query("US_Gross > 0")
        #.insert(1, 'US_Gross_ln', np.log( movies['US_Gross'] ) ) 
    )
    movies['ln_US_Gross'] = np.log( movies['US_Gross'] )
    movies #.to_native()
    return (movies,)


@app.cell(hide_code=True)
def _(movies):
    cat_features_to_graph = ['Distributor','Source','Major_Genre','Creative_Type']

    movies_base = alt.Chart(movies)

    movies_circle = movies_base.mark_circle(opacity=0.4).encode(
        alt.X("US_Gross").title("US Gross $"),
        alt.Y("Major_Genre").title(None),
        alt.YOffset('jitter:Q'),
        alt.Color("Major_Genre").title(None).legend(None),
        tooltip=['Title:N', 'Major_Genre']
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
    )
    mean_circle = movies_base.mark_circle(opacity=1, size=100, stroke='black').encode(
        alt.X("mean(US_Gross)"),
        alt.Y("Major_Genre"),
        alt.Color("Major_Genre")
    )

    v_rule = movies_base.mark_rule().encode(
        alt.X("mean(US_Gross)").title('')
    )
    movies_chart = movies_circle + v_rule + mean_circle
    movies_chart = movies_chart.properties(
        title="Adventure leads other Genres with Highest Average Gross $ in US"
    )

    mo_movie_chart = mo.ui.altair_chart( movies_chart )
    return (mo_movie_chart,)


@app.cell
def _(movies):
    alt.Chart(
        movies
    ).mark_bar().encode(
        alt.X("ln_US_Gross").bin(maxbins=20),
        alt.Y("count()")
    )
    return


@app.cell
def _(movies):
    X3 = movies[['Production_Budget','IMDB_Rating','IMDB_Votes', 'Rotten_Tomatoes_Rating','Major_Genre']]
    y3 = movies['US_Gross']

    categorical_features = ["Major_Genre"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            #("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    numeric_features =['Production_Budget','IMDB_Rating','IMDB_Votes','Rotten_Tomatoes_Rating']
    numeric_transformer = Pipeline(
        steps=[
            ('scaler',StandardScaler())
        ]
    )
    numeric_transformer2 = Pipeline(
        steps=[
            ('polyfit', PolynomialFeatures())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            #('poly', numeric_transformer2, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    reg = Pipeline(
        steps=[("preprocessor", preprocessor), ("reg", Lasso())]
    )
    parameters = {
        'reg__alpha':[i/10 for i in range(1,11)] , 
        #'preprocessor__poly__polyfit':[tuple([i,i+1]) for i in range(1,5)]
    }

    grid_search = GridSearchCV(reg, param_grid=parameters)

    lr3 = grid_search.fit(X3, y3)

    best_estimator = lr3.best_estimator_

    return X3, lr3, y3


@app.cell(hide_code=True)
def _(lr3, mo_movie_chart):
    mo.vstack([
        mo_movie_chart,
        mo_movie_chart.value,
         soot.plot_waterfall(lr3.best_estimator_ , mo_movie_chart.value, []).properties(
            height=200
        )  if len( mo_movie_chart.value ) == 1 else None
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""*Note*: in this example, we applied a standard scaler transformer to the numeric columns.""")
    return


@app.cell
def _(lr3):
    # works for pipeline
    fnames = lr3.best_estimator_[0].get_feature_names_out()
    coefs_ = lr3.best_estimator_[-1].coef_
    for f,c in zip(fnames, coefs_):
        print(f"{f} : {c:.4f}")
    return


@app.cell
def _(X3, lr3, y3):
    resid = pl.DataFrame({
        'y_true':y3.tolist(),
        'y_pred':lr3.predict(X3).tolist()
    }).with_columns(
        (pl.col("y_pred")-pl.col("y_true") ).alias("resid")
    )
    return (resid,)


@app.cell
def _(resid):
    # just for my own curiosity
    alt.Chart(resid).mark_circle(color='black').encode(
        alt.X("y_true").scale(zero=False),
        alt.Y("y_pred").scale(zero=False)
    ) | alt.Chart(resid).mark_circle(color='black').encode(
        alt.X("y_true").scale(type='log'),
        alt.Y("resid")#.scale(type='log')
    )
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
