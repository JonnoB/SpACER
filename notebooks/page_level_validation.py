import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import plotnine as p9
    import itertools
    import pandas as pd
    from itertools import product
    return p9, pd


@app.cell
def _(pd):
    page_values_df = pd.read_parquet("data/results_spiritualist/page_level_cer_comparison.parquet")
    return (page_values_df,)


@app.cell
def _(page_values_df):
    page_values_df
    return


@app.cell
def _(page_values_df, pd):

    combinations = page_values_df[['parsing_model', 'ocr_model']].drop_duplicates()

    rows = []
    for _, row in combinations.iterrows():
        mask = (
            (page_values_df['parsing_model'] == row['parsing_model']) &
            (page_values_df['ocr_model'] == row['ocr_model'])
        )
        subset = page_values_df.loc[mask, ['cer', 'spacer_total', 'cdd_total']]
        corr_matrix = subset.corr(method='spearman')

        rows.append({
            'parsing_model': row['parsing_model'],
            'ocr_model': row['ocr_model'],
            'spacer_cer_corr': corr_matrix.loc['spacer_total', 'cer'],
            'cdd_cer_corr': corr_matrix.loc['cdd_total', 'cer'],
        })

    corr_df = pd.DataFrame(rows)
    return (corr_df,)


@app.cell
def _(page_values_df):
    page_values_df
    return


@app.cell
def _(p9, page_values_df):
    p9.ggplot(page_values_df, p9.aes(x = 'cer', y = 'spacer_total', colour = 'parsing_model')) + p9.geom_point() 
    return


@app.cell
def _(p9, page_values_df):
    plot_df = page_values_df#.loc[(page_values_df['parsing_model']=='heron') ]

    p9.ggplot(plot_df, p9.aes(x = 'cer', y = 'spacer_total', colour = 'parsing_model')) + p9.geom_point() + p9.ylim(0, 0.075)+ p9.xlim(0, 0.2)
    return


@app.cell
def _(page_values_df):
    page_values_df.columns
    return


@app.cell
def _(p9, page_values_df):
    _plot_df = page_values_df

    p9.ggplot(_plot_df, p9.aes(x = 'ocr_model', y = 'spacer_total', fill = 'ocr_model')) + p9.geom_boxplot() + p9.ylim(0,0.2)
    return


@app.cell
def _(page_values_df):
    page_values_df.groupby(['parsing_model', 'ocr_model'])[['cer', 'spacer_total', 'cdd_total']].median()
    return


@app.cell
def _(corr_long):
    corr_long
    return


@app.cell
def _(corr_df, p9):


    # Convert to long format
    corr_long = corr_df.melt(
        id_vars=['parsing_model', 'ocr_model'],
        value_vars=['spacer_cer_corr', 'cdd_cer_corr'],
        var_name='metric',
        value_name='correlation'
    )

    corr_long = corr_long.loc[corr_long['parsing_model']!='gt']

    # Clean up metric labels
    corr_long['metric'] = corr_long['metric'].map({
        'spacer_cer_corr': 'SpACER vs CER',
        'cdd_cer_corr': 'CDD vs CER'
    })

    # Facetted heatmap
    plot = (
        p9.ggplot(corr_long, p9.aes(x='ocr_model', y='parsing_model', fill='correlation'))
        + p9.geom_tile()
        + p9.geom_text(p9.aes(label='correlation.round(2)'), size=14, color='white')
        + p9.facet_wrap('~metric')
        + p9.labs(
            title='Spearman Correlation by Model Combination',
            x='OCR Model',
            y='Parsing Model'
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(12, 5),
            axis_text_x=p9.element_text(rotation=45, hjust=1),
            panel_grid=p9.element_blank(),
            strip_text=p9.element_text(size=14, weight='bold')
        )
    )
    plot.save(filename='data/figures/CVD_CER_correlation.pdf', dpi = 300)
    plot.draw()
    return (corr_long,)


@app.cell
def _(pd):
    test = pd.read_parquet("data/results_spiritualist/end_to_end/crops/olmocr_crop_results.parquet")
    return (test,)


@app.cell
def _(test):
    test
    return


if __name__ == "__main__":
    app.run()
