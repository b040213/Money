from flask import Flask
import asyncio
import threading
import your_script  # 將你剛剛的程式命名為 `your_script.py`，然後匯入進來

app = Flask(__name__)

@app.route('/')
def home():
    return '幣圈監控機器人啟動中！'

# 啟動背景任務的函式
def start_background_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(your_script.run_loop_forever())

# 啟動背景任務
threading.Thread(target=start_background_loop, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
