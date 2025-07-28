from flask import Flask
import asyncio
import signal
import sys
import your_script  # 將你剛剛的程式命名為 `your_script.py`，然後匯入進來

app = Flask(__name__)

@app.route('/')
def home():
    return '幣圈監控機器人啟動中！'

# 啟動背景任務
loop = asyncio.get_event_loop()
task = loop.create_task(your_script.run_loop_forever())  # 在你的主程式內建議包成 async def run_loop_forever()

def shutdown_handler(sig, frame):
    print("Shutting down...")
    task.cancel()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)