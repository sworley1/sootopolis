import altair as alt
import polars as pl

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

def plot_waterfall(model, point_to_predict, regressor_labels):
    source = setup_data(
        model=model,
        point_to_predict=point_to_predict,
        regressor_labels=regressor_labels
    )
    return plot_altair_waterfall(source)