from flask import Flask
import asyncio
import threading
import time
import your_script  # 將你剛剛的程式命名為 `your_script.py`，然後匯入進來

app = Flask(__name__)
start_time = time.time()  # 記錄啟動時間（秒）

@app.route('/')
def home():
    elapsed = time.time() - start_time
    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes = int((elapsed % 3600) // 60)
    return f'幣圈監控機器人啟動中！<br>已執行 {days} 天 {hours} 小時 {minutes} 分鐘'

# 啟動背景任務的函式
def start_background_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(your_script.run_loop_forever())

# 啟動背景任務
threading.Thread(target=start_background_loop, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
