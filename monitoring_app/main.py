import requests
import time
import json
import tkinter as tk
import pandas as pd
from pandastable import Table, TableModel
# import config


# # # # # # # # # # #  # 
# Code to parse config #
# # # # # # # # # # #  # 



url="http://20.124.51.8:8001/api/v1/metrics" # config.metrics_url
# init_metrics_table = pd.DataFrame.from_dict(requests.get(url).json())


def color_boolean(val):
    color =''
    if isinstance(val, bool):
        if val == True and type(val) == bool:
            color = 'red'
        elif val == False:
            color = 'lightgreen'
    return 'background-color: %s' % color

def color_num(val):
    color =''
    if isinstance(val, float):
        if val < 0.05 == float:
            color = 'red'
        elif val >= 0.05:
            color = 'lightgreen'
    return 'background-color: %s' % color

def update(old_response=None):
    try:
        response = pd.DataFrame.from_dict(requests.get(url).json()) #.style.applymap(color_num).applymap(color_boolean)
        t.model.df = response
        t.redraw()
        old_response = response
    except:
        response = old_response
        t.model.df = response
        t.redraw()
    
    main.after(30000, update, old_response) # config.update_frequency
    


if __name__ == '__main__':

    init_metrics_table = pd.DataFrame(['Awaiting initial server response...'])
   

    main = tk.Tk()
    main.geometry('1400x400+200+100')
    main.title('MCI-APP Monitor | by: Mohamed Benaicha')
    f = tk.Frame(main)
    f.pack(fill='both', expand=True)
    t = Table(f, dataframe=init_metrics_table, showtoolbar=True, showstatusbar=True)
    t.show()
    
    main.after(30000, update, init_metrics_table)
    
    main.mainloop()
