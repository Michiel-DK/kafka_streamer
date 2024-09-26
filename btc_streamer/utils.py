from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, DateRangeSlider, CustomJS
from bokeh.plotting import figure
from bokeh.layouts import column
import numpy as np
import plotly.graph_objects as go
import pandas as pd


def plot_blokeh(data, path_to_plotly_html:'str'='.'):
# Convert datetime objects to numpy datetime64 for compatibility with Bokeh
    data[:, 3] = np.array([np.datetime64(item) for item in data[:, 3]])

    # Create a ColumnDataSource with the data
    source = ColumnDataSource(data=dict(
        x=data[:, 3],
        y=data[:, 2],
        color=['red' if val == 0 else 'green' for val in data[:, 0]]
    ))

    # Create a Bokeh figure
    p = figure(x_axis_type="datetime", title="Scatter plot with color (0: Red, 1: Green)",
            x_axis_label='DateTime (index 3)', y_axis_label='Value (index 2)', width=800)

    # Add circle glyphs to the plot
    p.circle('x', 'y', size=3, color='color', source=source)

    # Create a DateRangeSlider
    date_range_slider = DateRangeSlider(title="Date Range: ", 
                                        start=min(data[:, 3]), 
                                        end=max(data[:, 3]), 
                                        value=(min(data[:, 3]), max(data[:, 3])), 
                                        step=1)

    # JavaScript callback to update the plot based on the date range selected
    callback = CustomJS(args=dict(source=source, slider=date_range_slider), code="""
    var data = source.data;
    var start = new Date(slider.value[0]);
    var end = new Date(slider.value[1]);
    
    var new_x = [];
    var new_y = [];
    var new_color = [];
    
    for (var i = 0; i < data['x'].length; i++) {
        var date = new Date(data['x'][i]);
        if (date >= start && date <= end) {
            new_x.push(data['x'][i]);
            new_y.push(data['y'][i]);
            new_color.push(data['color'][i]);
        }
    }
    
    data['x'] = new_x;
    data['y'] = new_y;
    data['color'] = new_color;
    
    source.change.emit();
    """)


    # Attach the callback to the DateRangeSlider
    date_range_slider.js_on_change('value', callback)

    # Layout for the plot and slider
    layout = column(p, date_range_slider)

    full_path = f"{path_to_plotly_html}/bokeh_plot.html"
    # Output the plot to an HTML file
    output_file(full_path)

    # Save the layout as an HTML file
    save(layout)

    print(f"The plot has been saved as {full_path}.")
    
    return full_path

def plot_probas(plot_set, path_to_plotly_html:'str'='.'):
    
    # Sample data
    plot_set = pd.DataFrame(plot_set, columns=['label', 'prob', 'price', 'time'])
    plot_set['time'] = pd.to_datetime(plot_set['time'])
    plot_set.set_index('time', inplace=True)
    
    resampled = plot_set.resample('d').agg({'label': 'sum', 'prob':'count', 'price':'last'}).reset_index()
    resampled['buy_percentage'] = resampled['label'] / resampled['prob']
    resampled = np.array(resampled)
        
    time = resampled[:, 0]  # datetime for x-axis
    probabilities = resampled[:, -1]  # index 2 for y-axis
    probabilities = np.where(probabilities < 0.5, 0, probabilities)
    prices = resampled[:, -2]  # index 0 for color

    # Create figure
    fig = go.Figure()

    # Add probabilities line on the left y-axis
    fig.add_trace(
    go.Bar(x=time, y=probabilities, name='Probabilities', yaxis='y1', marker=dict(color='green'))
)

    # Add prices line on the right y-axis
    fig.add_trace(
        go.Scatter(x=time, y=prices, mode='lines', name='Prices', yaxis='y2', marker=dict(color='red'))
    )

    # Update layout for two y-axes
    fig.update_layout(
        title="Probabilities and Prices",
        xaxis_title="Time",
        yaxis_title="Probability > 50%",
        xaxis=dict(tickformat='%Y-%m-%d %H:%M', tickangle=45),
        yaxis=dict(
            title="Probabilities",
            range=[0, 1],
            showgrid=False
        ),
        yaxis2=dict(
            title="Prices",
            overlaying='y',
            side='right',
            #range=[10000, 20000]
        ),
        barmode='group',
        legend=dict(x=0.01, y=0.99)
    )

    full_path = f"{path_to_plotly_html}/probas_plot.html"

    fig.write_html(full_path, auto_play=False)
    
    return full_path


    
        # import ipdb; ipdb.set_trace()
    
        # # Extract columns for plotting
        # x_data = plot_set[:, 3]  # datetime for x-axis
        # y_data = plot_set[:, 2]  # index 2 for y-axis
        # color_data = plot_set[:, 0]  # index 0 for color
        
        # import plotly.graph_objs as go
        # colors = ['red' if val == 0 else 'green' for val in color_data]

        # # Create the scatter plot using Plotly
        # fig = go.Figure()

        # # Add trace with specific colors
        # fig.add_trace(go.Scatter(
        #     x=x_data, 
        #     y=y_data, 
        #     mode='markers',
        #     marker=dict(color=colors, size=5),  # Specify colors and marker size
        #     text=color_data
        # ))

        # # Update layout for better visualization
        # fig.update_layout(
        #     title='BTC Price predictions',
        #     yaxis_title='BTC Price',
        #     xaxis=dict(tickformat='%Y-%m-%d %H:%M', tickangle=45)
        # )
        
        # table = wandb.Table(columns=["Price predictions"])
        # # Create path for Plotly figure
        # path_to_plotly_html = "./plotly_figure.html"
        
        # # Write Plotly figure to HTML
        # # Set auto_play to False prevents animated Plotly charts
        # # from playing in the table automatically
        # fig.write_html(path_to_plotly_html, auto_play=False)

        # # Add Plotly figure as HTML file into Table
        # table.add_data(wandb.Html(path_to_plotly_html))