import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import polars as pl
    import altair as alt
    from sklearn.linear_model import LinearRegression
    from vega_datasets import data


@app.cell
def _():
    cars = pl.from_pandas( data.cars() )
    cars
    return (cars,)


@app.cell
def _(cars):
    alt.Chart(cars).mark_circle().encode(
        alt.Y("Miles_per_Gallon"),
        alt.X("Displacement"),
        alt.Color("Acceleration:Q")
    )
    return


@app.cell
def _(cars):
    sample = cars.filter(pl.col("Miles_per_Gallon").is_not_null()).filter(pl.col("Horsepower").is_not_null() )
    X = sample.select(['Displacement','Acceleration','Weight_in_lbs' ])
    y = sample.select('Miles_per_Gallon')
    lr = LinearRegression().fit(X,y)

    data_to_plot = []
    point_to_plot = X[47]
    for regressor,coef in zip(
        ['Intercept']+X.columns, 
        lr.intercept_.tolist() + lr.coef_.tolist()[0],
    ):
        if regressor != 'Intercept':
            value = coef * point_to_plot[regressor][0]
        else:
            value = coef

        data_to_plot.append({'label':regressor, 'amount':value})

    data_to_plot.append({'label':'End', 'amount':0.0})
    return X, data_to_plot, lr, y


@app.function
def setup_data(model , point_to_predict, regressor_labels):
    '''
    
    '''
    data_to_plot = []

    intercept_val = [0.0] if model.intercept_ == 0 else model.intercept_.tolist()

    for regressor,coef in zip(['Intercept'] +  regressor_labels , intercept_val + model.coef_.tolist()[0] ):
        if regressor != 'Intercept':
            value = coef * point_to_predict[regressor][0]
        else:
            value = coef

        data_to_plot.append({'label':regressor, 'amount':value})

    data_to_plot.append({'label':'End', 'amount':0.0})
    return pl.DataFrame(data_to_plot)


@app.cell
def _(data_to_plot):
    source=pl.DataFrame(data_to_plot)
    source
    return (source,)


@app.function
# Define frequently referenced fields
def plot_altair_waterfall(source) -> alt.Chart:
    amount = alt.datum.amount
    label = alt.datum.label
    window_lead_label = alt.datum.window_lead_label
    window_sum_amount = alt.datum.window_sum_amount

    # Define frequently referenced/long expressions
    calc_prev_sum = alt.expr.if_(label == "End", 0, window_sum_amount - amount)
    calc_amount = alt.expr.if_(label == "End", window_sum_amount, amount)
    calc_text_amount = (
        alt.expr.if_((label != "Intercept") & (label != "End") & calc_amount > 0, "+", "")
        + calc_amount
    )

    # The "base_chart" defines the transform_window, transform_calculate, and X axis
    base_chart = alt.Chart(source).transform_window(
        window_sum_amount="sum(amount)",
        window_lead_label="lead(label)",
    ).transform_calculate(
        calc_lead=alt.expr.if_((window_lead_label == None), label, window_lead_label),
        calc_prev_sum=calc_prev_sum,
        calc_amount=calc_amount,
        calc_text_amount=calc_text_amount,
        calc_center=(window_sum_amount + calc_prev_sum) / 2,
        calc_sum_dec=alt.expr.if_(window_sum_amount < calc_prev_sum, window_sum_amount, "None"),
        calc_sum_inc=alt.expr.if_(window_sum_amount > calc_prev_sum, window_sum_amount, "None"),
    ).encode(
        x=alt.X("label:O", axis=alt.Axis(title="", labelAngle=0), sort=None)
    )

    color_coding = (
        alt.when((label == "Intercept") | (label == "End"))
        .then(alt.value("#878d96"))
        .when(calc_amount > 0)
        .then(alt.value("green"))#24a148
        .otherwise(alt.value("red"))##fa4d56
    )

    bar = base_chart.mark_bar(size=45).encode(
        y=alt.Y("calc_prev_sum:Q", title="Amount"),
        y2=alt.Y2("window_sum_amount:Q"),
        color=color_coding,
    )

    # The "rule" chart is for the horizontal lines that connect the bars
    rule = base_chart.mark_rule(xOffset=-22.5, x2Offset=22.5).encode(
        y="window_sum_amount:Q",
        x2="calc_lead",
    )

    # Add values as text
    text_pos_values_top_of_bar = base_chart.mark_text(baseline="bottom", dy=-4).encode(
        text=alt.Text("calc_sum_inc:N",format=',.2f'),
        y="calc_sum_inc:Q",
    )
    text_neg_values_bot_of_bar = base_chart.mark_text(baseline="top", dy=4).encode(
        text=alt.Text("calc_sum_dec:N", format=',.2f'),
        y="calc_sum_dec:Q",
    )
    text_bar_values_mid_of_bar = base_chart.mark_text(baseline="middle").encode(
        text=alt.Text("calc_text_amount:N",format=',.2f'),
        y="calc_center:Q",
        color=alt.value("white"),
    )

    return alt.layer(
        bar,
        rule,
        text_pos_values_top_of_bar,
        text_neg_values_bot_of_bar,
        text_bar_values_mid_of_bar
    ).properties(
        width=800,
        height=450
    )


@app.cell
def _(source):
    plot_altair_waterfall(source)
    return


@app.cell
def _(X, y):
    lr2 = LinearRegression(fit_intercept=True).fit(X,y)

    s = X[12]

    sourc2 = setup_data(lr2, s, X.columns)
    plot_altair_waterfall(sourc2)
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
        bottomChart = plot_waterfall(lr, X_sample, X.columns).properties(
            height=200,title='linear regression equation for predicting miles per gallon'
        )
    else:
        print(len(moTestChart.value))
        bottomChart = alt.Chart()
    return (bottomChart,)


@app.function
def plot_waterfall(model, point_to_predict, regressor_labels):
    source = setup_data(
        model=model,
        point_to_predict=point_to_predict,
        regressor_labels=regressor_labels
    )
    return plot_altair_waterfall(source)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
