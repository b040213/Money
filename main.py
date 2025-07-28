from flask import Flask
import os
import threading
import time

app = Flask(__name__)

@app.route('/')
def home():
    return '幣圈監控機器人啟動中！'

def run_flask():
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

def job_loop():
    while True:
        print("正在執行判斷邏輯...（這裡加你主程式的邏輯）")
        time.sleep(1500)  # 25分鐘

if __name__ == '__main__':
    threading.Thread(target=run_flask).start()
    job_loop()