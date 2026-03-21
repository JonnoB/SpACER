import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import os
    return os, pd


@app.cell
def _(pd):
    gt_chars = pd.read_parquet('data/spatial_uncertainty/characters_gt.parquet')

    gt_chars = gt_chars.loc[gt_chars['char_text']!=' ']
    word_chars = pd.read_parquet('data/spatial_uncertainty/characters_word.parquet')
    line_chars = pd.read_parquet('data/spatial_uncertainty/characters_line.parquet')
    para_chars = pd.read_parquet('data/spatial_uncertainty/characters_para.parquet')
    return gt_chars, word_chars


@app.cell
def _(gt_chars, word_chars):
    comp = gt_chars.merge(word_chars.loc[:, ['char_id', 'x','y']], suffixes = ['', '_inf'], on='char_id')
    comp['x_diff'] = comp['x'] - comp['x_inf'] 
    comp['x_perc_diff'] = comp['x_diff']/comp['w']
    comp['y_diff'] = comp['y'] - comp['y_inf'] 
    comp['y_perc_diff'] = comp['y_diff']/comp['h']
    return (comp,)


@app.cell
def _(comp):
    comp
    return


@app.cell
def _(comp):
    comp[['x_diff', 'x_perc_diff', 'y_diff', 'y_perc_diff']].describe()
    return


@app.cell
def _(pd):
    crop_data = pd.read_parquet('data/spatial_uncertainty/crop_cer_results.parquet')
    crop_data['w_frac'] = (crop_data['w_frac']*100).astype(int)
    crop_data['h_frac'] = (crop_data['h_frac']*100).astype(int)
    return (crop_data,)


@app.cell
def _(crop_data):
    crop_data['alignment'].unique()
    return


@app.cell
def _(crop_data):
    crop_data.groupby(['w_frac', 'h_frac'])[['cer_line', 'cer_word', 'cer_para']].mean()
    return


@app.cell
def _(crop_data):
    crop_data[['cer_line', 'cer_word', 'cer_para']].mean()
    return


@app.cell
def _(crop_data, os):
    import seaborn as sns
    import matplotlib.pyplot as plt

    outcomes = [ 'cer_word','cer_line', 'cer_para']
    titles = ['CER Word', 'CER Line', 'CER Para']

    grouped = (
        crop_data.groupby(['w_frac', 'h_frac'])[outcomes]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, outcome, title in zip(axes, outcomes, titles):
        heatmap_data = grouped.pivot(index='h_frac', columns='w_frac', values=outcome)
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='viridis_r',
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'label': title},
            ax=ax
        )
        ax.set_title(f'{title} by Crop Fractions', fontsize=14)
        ax.set_xlabel('Percent of total image width')
        ax.set_ylabel('Percent of total image height')

    plt.tight_layout()
    os.makedirs('data/figures', exist_ok=True)
    plt.savefig('data/figures/spatial_granularity_error.pdf', bbox_inches='tight')
    plt.show()
    return plt, sns


@app.cell
def _(crop_data, plt, sns):
    _outcomes = ['cer_line', ]
    _titles = ['CER Line',  ]

    # Collapse h_frac — mean across all h_frac values, group only by w_frac
    _collapsed = (
        crop_data.groupby('w_frac')[_outcomes]
        .mean()
        .reset_index()
    )

    _palette = sns.color_palette('Set2', n_colors=len(_outcomes))

    _fig, _ax = plt.subplots(figsize=(10, 5))

    for _color, _outcome, _title in zip(_palette, _outcomes, _titles):
        _ax.plot(
            _collapsed['w_frac'],
            _collapsed[_outcome],
            marker='o',
            color=_color,
            label=_title,
            linewidth=2,
            markersize=7,
        )

    _ax.set_title('CER by Image Width Crop Fraction', fontsize=14)
    _ax.set_xlabel('Percent of total image width')
    _ax.set_ylabel('Mean CER')
    _ax.legend(title='Metric')
    _ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(crop_data, plt, sns):

    _outcomes = ['cer_line', 'cer_word', 'cer_para']
    _titles = ['CER Line', 'CER Word', 'CER Para']
    _n_col_values = [1, 2, 3]

    _grouped = (
        crop_data.groupby(['w_frac', 'h_frac', 'n_columns'])[_outcomes]
        .mean()
        .reset_index()
    )

    _fig, _axes = plt.subplots(3, 3, figsize=(22, 18))

    for _row_idx, _n_col in enumerate(_n_col_values):
        _subset = _grouped[_grouped['n_columns'] == _n_col]
        for _col_idx, (_outcome, _title) in enumerate(zip(_outcomes, _titles)):
            _ax = _axes[_row_idx, _col_idx]
            _heatmap_data = _subset.pivot(index='h_frac', columns='w_frac', values=_outcome)
            sns.heatmap(
                _heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': _title},
                ax=_ax
            )
            _ax.set_title(f'{_title} | n_columns={_n_col}', fontsize=13)
            _ax.set_xlabel('Percent of total image width')
            _ax.set_ylabel('Percent of total image height')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(crop_data, plt, sns):

    _outcomes = ['cer_line', 'cer_word', 'cer_para']
    _titles = ['CER Line', 'CER Word', 'CER Para']
    _n_col_values = ['left', 'centered', 'justified']

    _grouped = (
        crop_data.groupby(['w_frac', 'h_frac', 'alignment'])[_outcomes]
        .mean()
        .reset_index()
    )

    _fig, _axes = plt.subplots(3, 3, figsize=(22, 18))

    for _row_idx, _n_col in enumerate(_n_col_values):
        _subset = _grouped[_grouped['alignment'] == _n_col]
        for _col_idx, (_outcome, _title) in enumerate(zip(_outcomes, _titles)):
            _ax = _axes[_row_idx, _col_idx]
            _heatmap_data = _subset.pivot(index='h_frac', columns='w_frac', values=_outcome)
            sns.heatmap(
                _heatmap_data,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': _title},
                ax=_ax
            )
            _ax.set_title(f'{_title} | alignment={_n_col}', fontsize=13)
            _ax.set_xlabel('Percent of total image width')
            _ax.set_ylabel('Percent of total image height')

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
