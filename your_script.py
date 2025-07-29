import asyncio
import pandas as pd
import numpy as np
from bingx_py import BingXAsyncClient
import time
import requests
import aiohttp

api_key = "L9ywGJGME1uqTkIRd1Od08IvXyWCCyA2YKGwMPnde8BWOmm8gAC5xCdGAZdXFWZMt1euiT574cgAvQdQTw"
api_secret = "NYY1OfADXhu26a6F4Tw67RbHDvJcQ2bGOcQWOI1vXccWRoutdIdfsvxyxVtdLxZAGFYn9eYZN6RX7w2fQ"

async def MA(t): #ma平均移動線*3
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=t,
            interval="1h",
            limit=500
        )

        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()

        df.dropna(subset=['MA5', 'MA10'], inplace=True)
        if len(df) < 3:
            return 0  # 需要至少3筆資料以便比對最後兩根

        # 取最後三根K線（為了能計算兩次交叉判斷）
        prev2 = df.iloc[-3]
        prev1 = df.iloc[-2]
        last  = df.iloc[-1]

        signal1 = 0
        signal2 = 0

        # 判斷倒數第2根的交叉訊號
        if prev2['MA5'] <= prev2['MA10'] and prev1['MA5'] > prev1['MA10']:
            signal1 = 1
        elif prev2['MA5'] >= prev2['MA10'] and prev1['MA5'] < prev1['MA10']:
            signal1 = -1

        # 判斷倒數第1根的交叉訊號
        if prev1['MA5'] <= prev1['MA10'] and last['MA5'] > last['MA10']:
            signal2 = 1
        elif prev1['MA5'] >= prev1['MA10'] and last['MA5'] < last['MA10']:
            signal2 = -1

        # 回傳邏輯
        if signal1 == signal2:
            return signal1
        elif signal1 != 0 and signal2 == 0:
            return signal1
        elif signal2 != 0 and signal1 == 0:
            return signal2
        else:
            return 0



async def BE_BIG(t): #成交量放大*3
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=t,
            interval="1h",
            limit=30
        )

        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df = df.sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df['vol_mean20'] = df['volume'].rolling(window=20).mean()
        df['vol_spike'] = df['volume'] > 2 * df['vol_mean20']
        df['price_up'] = df['close'] > df['close'].shift(1)
        df['price_down'] = df['close'] < df['close'].shift(1)

        df['signal'] = 0
        df.loc[df['vol_spike'] & df['price_up'], 'signal'] = 1
        df.loc[df['vol_spike'] & df['price_down'], 'signal'] = -1

        # 取最後兩根K棒的訊號
        last_two = df['signal'].iloc[-2:]

        # 檢查最後兩根K棒是否有訊號
        last_two = df['signal'].iloc[-2:]
        signals = set(last_two.values)

        if signals == {1}:
            return 1
        elif signals == {-1}:
            return -1
        elif signals == {1, -1}:
            return 0  # 衝突訊號，回傳無訊號
        else:
            return 0
        
        
        
        
async def MACD(t): #MACD判斷*2
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=t,
            interval="1h",
            limit=300
        )

        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df = df.sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # 計算 MACD 指標
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp12 - exp26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['DIF'] - df['DEA']

        # 建立訊號欄位：1=買進，-1=賣出，0=無
        df['signal'] = 0

        # MACD 死亡交叉條件：DIF 從上往下穿越 DEA，且 MACD_hist 由正轉負
        cond_sell = (df['DIF'] < df['DEA']) & (df['DIF'].shift(1) >= df['DEA'].shift(1)) & (df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) >= 0)
        # MACD 黃金交叉條件：DIF 從下往上穿越 DEA，且 MACD_hist 由負轉正
        cond_buy = (df['DIF'] > df['DEA']) & (df['DIF'].shift(1) <= df['DEA'].shift(1)) & (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)

        df.loc[cond_buy, 'signal'] = 1
        df.loc[cond_sell, 'signal'] = -1

        # 取最後兩根訊號
        last_two = df['signal'].iloc[-2:]
        signals = set(last_two.values)

        if signals == {1}:
            return 1
        elif signals == {-1}:
            return -1
        elif signals == {1, -1}:
            return 0  # 衝突訊號
        else:
            return 0




async def RSI(t): #RSI*2
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=t,
            interval="1h",
            limit=300
        )

        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df = df.sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # 計算 RSI(14)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['signal'] = 0

        # 優先判斷穿越 20/80
        df.loc[(df['RSI'] > 20) & (df['RSI'].shift(1) <= 20), 'signal'] = 2
        df.loc[(df['RSI'] < 80) & (df['RSI'].shift(1) >= 80), 'signal'] = -2

        # 再判斷穿越 30/70，且未被 20/80 訊號覆蓋才設定
        df.loc[((df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)) & (df['signal'] == 0), 'signal'] = 1
        df.loc[((df['RSI'] < 70) & (df['RSI'].shift(1) >= 70)) & (df['signal'] == 0), 'signal'] = -1

        # 取最後兩根訊號
        last_two = df['signal'].iloc[-2:]

        # 如果有 0 或兩根訊號不一樣，回傳 0
        signals = set(last_two.values)
        if 0 in signals or len(signals) > 1:
            return 0

        last_signal = last_two.iloc[-1]

        # 判斷背離 (最近20根)
        window = 20
        recent = df.tail(window)
        close_vals = recent['close'].values
        rsi_vals = recent['RSI'].values

        price_new_low = close_vals[-1] <= close_vals.min()
        rsi_not_new_low = rsi_vals[-1] > rsi_vals.min()
        price_new_high = close_vals[-1] >= close_vals.max()
        rsi_not_new_high = rsi_vals[-1] < rsi_vals.max()

        divergence = False
        if price_new_low and rsi_not_new_low:
            divergence = True  # 正背離
        elif price_new_high and rsi_not_new_high:
            divergence = True  # 負背離

        # 有背離則加強訊號 (+1 或 -1)
        if divergence:
            if last_signal > 0:
                last_signal += 1
            elif last_signal < 0:
                last_signal -= 1

        return last_signal


async def THREE(t): #三根陰陽線*1
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=t,
            interval="1h",
            limit=10
        )
        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df = df.sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df.set_index('time', inplace=True)

        # 判斷三根連續上漲與下跌
        df['c1_up'] = df['close'] > df['close'].shift(1)
        df['c2_up'] = df['close'].shift(1) > df['close'].shift(2)
        df['c3_up'] = df['close'].shift(2) > df['close'].shift(3)
        df['buy_signal'] = df['c1_up'] & df['c2_up'] & df['c3_up']

        df['c1_down'] = df['close'] < df['close'].shift(1)
        df['c2_down'] = df['close'].shift(1) < df['close'].shift(2)
        df['c3_down'] = df['close'].shift(2) < df['close'].shift(3)
        df['sell_signal'] = df['c1_down'] & df['c2_down'] & df['c3_down']

        # 取最後兩根訊號
        last_two_buy = df['buy_signal'].iloc[-2:]
        last_two_sell = df['sell_signal'].iloc[-2:]

        buy_exist = last_two_buy.any()
        sell_exist = last_two_sell.any()

        if buy_exist and not sell_exist:
            return 1
        elif sell_exist and not buy_exist:
            return -1
        elif buy_exist and sell_exist:
            return 0  # 衝突訊號
        else:
            return 0  # 無訊號




async def BREAK_OUT(t): #價格突破阻力*1
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=t,
            interval="1h",
            limit=30  # 取30根足夠做10~20根區間判斷
        )

        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df = df.sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # 計算過去10到20根的最高價與最低價(不含當前根)
        df['max_10_20'] = df['high'].shift(1).rolling(window=11).max()  # 取前11根中最高，包含第10根
        df['min_10_20'] = df['low'].shift(1).rolling(window=11).min()

        # 建立訊號欄位，判斷是否突破前10-20根最高/最低價
        df['signal'] = 0
        df.loc[df['close'] > df['max_10_20'], 'signal'] = 1    # 突破最高價 → 看漲
        df.loc[df['close'] < df['min_10_20'], 'signal'] = -1   # 跌破最低價 → 看跌

        # 取最後兩根訊號
        last_two = df['signal'].iloc[-2:]
        signals = set(last_two.values)

        if signals == {1}:
            return 1
        elif signals == {-1}:
            return -1
        elif signals == {1, -1}:
            return 0  # 衝突訊號
        else:
            return 0
        
def calculate_kdj(df, n=9, k_period=3, d_period=3): #KDJ*1
    low_min = df['low'].rolling(window=n).min()
    high_max = df['high'].rolling(window=n).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=(k_period - 1), adjust=False).mean()
    df['D'] = df['K'].ewm(com=(d_period - 1), adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df

async def KDJ(t): #KDJ*1
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=t,
            interval="1h",
            limit=200
        )

        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df = df.sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = calculate_kdj(df)

        # 建立訊號欄位：1=買進，-1=賣出，0=無
        df['signal'] = 0
        df.loc[(df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)), 'signal'] = 1
        df.loc[(df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)), 'signal'] = -1

        # 檢查最後兩根K棒是否有訊號
        last_two = df['signal'].iloc[-2:]
        signals = set(last_two.values)

        if signals == {1}:
            return 1
        elif signals == {-1}:
            return -1
        elif signals == {1, -1}:
            return 0  # 衝突訊號，回傳無訊號
        else:
            return 0



async def get_current_price(symbol):
    try:
        async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
            res = await client.market.get_symbol_price(symbol=symbol)
            return float(res['price'])
    except Exception as e:
        print(f"❗ 取得 {symbol} 現價時發生錯誤：{e}")
        return None
        

dc = "https://discord.com/api/webhooks/1387480183698886777/RAzRv4VECjgloChid-aL0vg24DnEqpAHw66ASMSLszpMJTNxm9djACseKE4x7kjydD63"


symbols = [
    "BTC-USDT", "ETH-USDT", "DOT-USDT", "SOL-USDT", "XRP-USDT",
    "AAVE-USDT", "INJ-USDT", "CRV-USDT", "LINK-USDT", "OM-USDT",
    "CHZ-USDT","THETA-USDT","NEAR-USDT","VET-USDT","AVAX-USDT",
    "FIL-USDT","ICP-USDT","BNB-USDT","ALGO-USDT","GRT-USDT",
    "OP-USDT","HBAR-USDT","ARB-USDT","TONCOIN-USDT","APT-USDT",
    "GALA-USDT","IMX-USDT","JASMY-USDT","ORDI-USDT","TIA-USDT"
    ]

# 權重列表，對應指標順序：MA, BE_BIG, MACD, RSI, THREE, BREAK_OUT, KDJ
weights = [3, 3, 2, 2, 1, 2, 1]

skip_counts = {}  # 全域字典，記錄幣種跳過次數



async def get_current_price(symbol):
    try:
        url = f"https://open-api.bingx.com/openApi/swap/v2/quote/price?symbol={symbol}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                return float(data['data']['price'])
    except Exception as e:
        print(f"❗ 取得 {symbol} 現價時發生錯誤：{e}")
        return None

async def get_kline_data(symbol, period=14, timeframe="1h"):
    try:
        url = f"https://open-api.bingx.com/openApi/swap/v2/quote/klines?symbol={symbol}&interval={timeframe}&limit={period}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                return data['data']
    except Exception as e:
        print(f"❗ 取得 {symbol} K線資料時發生錯誤：{e}")
        return None

async def ATR(symbol, period=14):
    try:
        kline_data = await get_kline_data(symbol, period)
        if not kline_data or len(kline_data) < period:
            return None

        tr_list = []
        for i in range(len(kline_data)):
            high = float(kline_data[i]['high'])
            low = float(kline_data[i]['low'])
            close_prev = float(kline_data[i-1]['close']) if i > 0 else float(kline_data[i]['close'])
            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            tr_list.append(tr)
        
        atr = np.mean(tr_list[-period:])
        return atr
    except Exception as e:
        print(f"❗ 計算 {symbol} ATR時發生錯誤：{e}")
        return None

emoji_map = {
    "BTC": "•", "ETH": "•", "DOT": "•", "SOL": "•", "XRP": "•",
    "AAVE": "•", "INJ": "•", "CRV": "•", "LINK": "•", "OM": "•",
    "CHZ": "•", "THETA": "•", "NEAR": "•", "VET": "•", "AVAX": "•",
    "FIL": "•", "ICP": "•", "BNB": "•", "ALGO": "•", "GRT": "•",
    "OP": "•","HBAR": "•","ARB": "•","TONCOIN": "•","APT": "•",
    "GALA": "•","IMX": "•","JASMY": "•","ORDI": "•","TIA": "•"
}

async def send_to_discord(message: str):
    async with aiohttp.ClientSession() as session:
        json_data = {"content": message}
        async with session.post(dc, json=json_data) as resp:
            if resp.status == 204:
                print("N")
            else:
                text = await resp.text()
                print(f"發送失敗，狀態碼：{resp.status}，訊息：{text}")

async def evaluate_symbol(symbol):

    # 如果該幣跳過計數 > 0，直接跳過並扣減一次
    if skip_counts.get(symbol, 0) > 0:
        skip_counts[symbol] -= 1
        print(f"跳過 {symbol} 偵測，剩餘跳過次數：{skip_counts[symbol]}")
        return  # 不做評估
    indicators = ['MA', 'BE_BIG', 'MACD', 'RSI', 'THREE', 'BREAK_OUT', 'KDJ']
    scores = [
        await MA(symbol),
        await BE_BIG(symbol),
        await MACD(symbol),
        await RSI(symbol),
        await THREE(symbol),
        await BREAK_OUT(symbol),
        await KDJ(symbol)
    ]
    total_score = sum(s * w for s, w in zip(scores, weights))
    current_price = await get_current_price(symbol)
    atr = await ATR(symbol)

    # 幣名簡化
    short = symbol.split("-")[0]
    emoji = emoji_map.get(short, "")
    triggered_indicators = [name for name, score in zip(indicators, scores) if score != 0]

    # 轉成字串（用逗號分隔）
    indicators_str = ", ".join(triggered_indicators) if triggered_indicators else "無"

    # 判斷進場方向
    if total_score >= 6:
        direction = "📈 **看漲進場**"
    elif total_score <= -6:
        direction = "📉 **看跌進場**"
    elif total_score >= 9:
        direction = "📈!!!!看漲強力進場!!!!"
    elif total_score <= -9:
        direction = "📉!!!!看跌強力進場!!!!"
    else:
        return 0
    skip_counts[symbol] = 2
    
    # 處理ATR顯示
    atr_info = f"📏 ATR: {atr:,.3f}  " \
               f"1.5: {atr*1.5:,.3f}  " \
               f"3: {atr*3:,.3f}\n" if atr is not None else "📏 ATR: 無法計算\n"

    # 組合訊息
    message = (
        f"{emoji} `{symbol}`\n"
        f"💰 現價：${current_price:,.2f}\n"
        f"📊 總分：{total_score}\n"
        f"{atr_info}"
        f"{direction}\n"
        f"📌 進場依據：{indicators_str}"
    )

    await send_to_discord(message)

async def run_loop_forever():
    await send_to_discord("💡 搜幣程式啟動！")
    while True:
        for sym in symbols:
            await evaluate_symbol(sym)
            await asyncio.sleep(0.2)  # 每次發完訊息後等待0.2秒，避免限速
        print("等待 20 分鐘後重新判斷...\n")
        await asyncio.sleep(1200)  # 非同步等待20分鐘

if __name__ == "__main__":
    asyncio.run(run_loop_forever())
