import tkinter as tk

root = tk.Tk() # 建立物件
root.geometry(f'{400}x{300}+{0}+{0}') # 設定大小: widthxheight+x+y

num = tk.StringVar() # 文字變數
num.set('5') # 預設

def modify():
    global num
    print(num.get()) # .get() -> 取值

label = tk.Label(root, text='set gaussian block:', font='3px')
label.pack() # 放入

# textvariable -> 字內容的變數名稱，如果變數被修改，文字就會發生變化
entry = tk.Entry(root, textvariable=num)
entry.pack()

button = tk.Button(root, text='確認', command=modify)
button.pack()

root.mainloop()