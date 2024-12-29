import tkinter as tk
root_tk = tk.Tk()
root_tk.geometry(f'{300}x{200}')

optionList = ['1', '2', '3']
optionStr = tk.StringVar(); optionStr.set('1')
optionChoice = 0

def tk_action():
    global optionList, optionStr, optionChoice
    # print(f'{optionStr.get()}')
    optionChoice = int(optionStr.get())
    root_tk.quit()

tk.Label(root_tk, text="請選擇要demo的作品").pack()
root_menu = tk.OptionMenu(root_tk, optionStr, *optionList)
root_menu.config(width=10)
root_menu.pack(pady=10)

tk.Button(root_tk, text='Ok', command=tk_action).pack()

root_tk.mainloop()

import sys
sys.path.append('term_project') # 這邊是程式碼路徑 
if optionChoice == 1:
    print('choose 1')
    import snack
elif optionChoice == 2: 
    print('choose 2')
    import main_2
elif optionChoice == 3: 
    print('choose 3')
    import main
# else if optionChoice == 3:
    # 到時候大概就會像這樣