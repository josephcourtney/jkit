import pandas as pd
import rich


def pprint_dataframe(
    df: pd.DataFrame,
    title=None,
    color_cycle=[
        "#D95F4C",  # red
        "#E3739B",  # pink
        "#E3A040",  # orange
        "#E5C04F",  # yellow-orange
        "#EBE070",  # yelllow
        "#BFCE51",  # yellow-green
        "#91BA56",  # green
        "#5DABE9",  # blue
        "#A645C1",
    ],
    border_color="white",
):
    console = rich.console.Console()
    df = df.reset_index().rename(columns={"index": ""})
    table = rich.table.Table(show_footer=False, title=title)
    num_colors = len(color_cycle)

    column_names = df.columns
    column_names.insert(0, "index")
    for i, col in enumerate(column_names):
        table.add_column(
            str(col),
            style="bold " + color_cycle[i % num_colors],
            header_style="bold " + color_cycle[i % num_colors],
        )

    rows = slice(0, 10)
    for _, row in df.iloc[rows].iterrows():
        row = [str(item) for item in row]
        table.add_row(*list(row))
    for i in range(len(table.columns)):
        table.columns[i].justify = "right"
    table.border_style = border_color
    console.print(table)


def pprint(
    obj,
    *args,
    **kwargs,
):
    match obj:
        case pd.DataFrame():
            pprint_dataframe(obj, *args, **kwargs)
        case _:
            rich.pretty.pprint(obj, *args, **kwargs)
