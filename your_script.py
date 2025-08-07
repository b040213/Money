import asyncio
import pandas as pd
import numpy as np
from bingx_py import BingXAsyncClient
import time
import requests
import aiohttp
import datetime

api_key = "L9ywGJGME1uqTkIRd1Od08IvXyWCCyA2YKGwMPnde8BWOmm8gAC5xCdGAZdXFWZMt1euiT574cgAvQdQTw"
api_secret = "NYY1OfADXhu26a6F4Tw67RbHDvJcQ2bGOcQWOI1vXccWRoutdIdfsvxyxVtdLxZAGFYn9eYZN6RX7w2fQ"


async def MA(symbol, interval):  # ma平均移動線*3，交叉+排列加分版
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
            limit=500
        )

        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()

        df.dropna(subset=['MA5', 'MA10', 'MA20'], inplace=True)
        if len(df) < 3:
            return 0  # 資料不足

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

        # 綜合兩根訊號（你原本的邏輯）
        if signal1 == signal2 and signal1 != 0:
            signal = signal1
        elif signal1 != 0 and signal2 == 0:
            signal = signal1
        elif signal2 != 0 and signal1 == 0:
            signal = signal2
        else:
            signal = 0

        # 加入排列加成邏輯（在最後一根判斷）
        if signal == 1 and last['MA5'] > last['MA10'] > last['MA20']:
            signal += 0.5
        elif signal == -1 and last['MA5'] < last['MA10'] < last['MA20']:
            signal -= 0.5

        return signal



async def BE_BIG(symbol, interval):
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
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
        df['vol_ratio'] = df['volume'] / df['vol_mean20']
        df['price_up'] = df['close'] > df['close'].shift(1)
        df['price_down'] = df['close'] < df['close'].shift(1)

        conditions = [
            (df['vol_ratio'] >= 3) & df['price_up'],
            (df['vol_ratio'] >= 2.5) & df['price_up'],
            (df['vol_ratio'] >= 2) & df['price_up'],
            (df['vol_ratio'] >= 1.5) & df['price_up'],
            (df['vol_ratio'] >= 3) & df['price_down'],
            (df['vol_ratio'] >= 2.5) & df['price_down'],
            (df['vol_ratio'] >= 2) & df['price_down'],
            (df['vol_ratio'] >= 1.5) & df['price_down']
        ]

        choices = [2, 1.5, 1, 0.5, -2, -1.5, -1, -0.5]

        df['signal'] = np.select(conditions, choices, default=0)

        # 取最後兩根K棒的訊號
        last_two = df['signal'].iloc[-2:]
        signals = set(last_two.values)

        if signals == {2} or signals == {1.5} or signals == {1} or signals == {0.5}:
            # 如果都是正信號，回傳最大正訊號
            return max(signals)
        elif signals == {-2} or signals == {-1.5} or signals == {-1} or signals == {-0.5}:
            # 如果都是負信號，回傳最大負訊號（絕對值最大，但要回負）
            return min(signals)
        else:
            # 混合或無訊號回傳0
            return 0
        
        
        
async def MACD(symbol, interval):  # MACD 判斷強度版，±0 → ±2
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
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

        df['signal'] = 0.0

        # 判斷交叉（黃金 / 死亡）條件
        cond_sell = (
            (df['DIF'] < df['DEA']) &
            (df['DIF'].shift(1) >= df['DEA'].shift(1)) &
            (df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) >= 0)
        )
        cond_buy = (
            (df['DIF'] > df['DEA']) &
            (df['DIF'].shift(1) <= df['DEA'].shift(1)) &
            (df['MACD_hist'] > 0) & (df['MACD_hist'].shift(1) <= 0)
        )

        df.loc[cond_buy, 'signal'] = 1
        df.loc[cond_sell, 'signal'] = -1

        last2 = df.iloc[-2]
        last1 = df.iloc[-1]

        signal = 0

        # 最新兩根是否有交叉訊號
        if last2['signal'] == 1 or last1['signal'] == 1:
            signal = 1
        elif last2['signal'] == -1 or last1['signal'] == -1:
            signal = -1

        # 如果剛交叉，檢查 DIF-DEA 是否擴大、柱狀體是否放大
        if signal != 0:
            dif_gap_prev = abs(last2['DIF'] - last2['DEA'])
            dif_gap_now = abs(last1['DIF'] - last1['DEA'])
            hist_prev = abs(last2['MACD_hist'])
            hist_now = abs(last1['MACD_hist'])

            # 動能擴大
            if dif_gap_now > dif_gap_prev:
                signal += 0.5 if signal > 0 else -0.5

            # 柱體放大
            if hist_now > hist_prev:
                signal += 0.5 if signal > 0 else -0.5

        return signal




async def RSI(symbol,interval): #RSI*2
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
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

        df['signal'] = 0.0

        # 優先判斷穿越 20/80
        df.loc[(df['RSI'] > 20) & (df['RSI'].shift(1) <= 20), 'signal'] = 1.5
        df.loc[(df['RSI'] < 80) & (df['RSI'].shift(1) >= 80), 'signal'] = -1.5

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
                last_signal += 0.5
            elif last_signal < 0:
                last_signal -= 0.5

        return last_signal


async def THREE(symbol,interval): #三根陰陽線*1
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
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




async def BREAK_OUT(symbol, interval):
    try:
        async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
            res = await client.swap.kline_candlestick_data(
                symbol=symbol,
                interval=interval,
                limit=30
            )
        
        # 整理 K 線資料
        data = [kline.__dict__ for kline in res.data]
        df = pd.DataFrame(data).sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # 計算過去 20 根（不含當前）高低點
        df['max_20'] = df['high'].shift(1).rolling(window=20).max()
        df['min_20'] = df['low'].shift(1).rolling(window=20).min()

        last_close = df['close'].iloc[-1]
        resistance = df['max_20'].iloc[-1]
        support = df['min_20'].iloc[-1]

        # 取得 ATR 值
        atr_value = await ATR(symbol, 14)
        if atr_value is None:
            return 0  # 計算ATR失敗則跳過

        signal = 0

        # 上漲突破判斷
        if last_close > resistance:
            breakout_amt = last_close - resistance
            if breakout_amt >= 1.5 * atr_value:
                signal = 2
            elif breakout_amt >= 1.0 * atr_value:
                signal = 1.5
            elif breakout_amt > 0 :
                signal = 1

        # 下跌跌破判斷
        elif last_close < support:
            breakdown_amt = support - last_close
            if breakdown_amt >= 1.5 * atr_value:
                signal = -2
            elif breakdown_amt >= 1.0 * atr_value:
                signal = -1.5
            elif breakdown_amt >0 :
                signal = -1

        return signal

    except Exception as e:
        print(f"❗ 計算 {symbol} 突破訊號時發生錯誤：{e}")
        return 0
    
def calculate_kdj(df, n=9, k_period=3, d_period=3):
    low_min = df['low'].rolling(window=n).min()
    high_max = df['high'].rolling(window=n).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=(k_period - 1), adjust=False).mean()
    df['D'] = df['K'].ewm(com=(d_period - 1), adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df


async def KDJ(symbol, interval):
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
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

        signal1 = 0
        signal2 = 0

        # 交叉訊號判斷，倒數第二根
        if (df['K'].iloc[-3] <= df['D'].iloc[-3]) and (df['K'].iloc[-2] > df['D'].iloc[-2]):
            signal1 = 1
        elif (df['K'].iloc[-3] >= df['D'].iloc[-3]) and (df['K'].iloc[-2] < df['D'].iloc[-2]):
            signal1 = -1

        # 交叉訊號判斷，倒數第一根
        if (df['K'].iloc[-2] <= df['D'].iloc[-2]) and (df['K'].iloc[-1] > df['D'].iloc[-1]):
            signal2 = 1
        elif (df['K'].iloc[-2] >= df['D'].iloc[-2]) and (df['K'].iloc[-1] < df['D'].iloc[-1]):
            signal2 = -1

        # 綜合交叉訊號
        if signal1 == signal2 and signal1 != 0:
            signal = signal1
        elif signal1 != 0 and signal2 == 0:
            signal = signal1
        elif signal2 != 0 and signal1 == 0:
            signal = signal2
        else:
            signal = 0

        # 強度加成判斷（只在非零訊號時加強）
        if signal != 0:
            last_K = df['K'].iloc[-1]
            last_D = df['D'].iloc[-1]
            last_J = df['J'].iloc[-1]

            # 超賣區且 J 明顯高於 K、D（強多）
            if signal > 0 and last_K < 20 and last_D < 20 and last_J > last_K and last_J > last_D:
                signal += 0.5

            # 超買區且 J 明顯低於 K、D（強空）
            if signal < 0 and last_K > 80 and last_D > 80 and last_J < last_K and last_J < last_D:
                signal -= 0.5

            # 連續兩根訊號（signal1 與 signal2 都相同且非零）再加強 0.5
            if signal1 == signal2 and signal1 != 0:
                signal += 0.5 if signal > 0 else -0.5

        return signal




async def BOLL(symbol, interval, period=20, std_mult=2):
    async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
        res = await client.swap.kline_candlestick_data(
            symbol=symbol,
            interval=interval,
            limit=period + 50
        )
    data = [kline.__dict__ for kline in res.data]
    df = pd.DataFrame(data)
    df = df.sort_values('time')
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)

    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)

    # 計算中軸與上下軌
    df['MA20'] = df['close'].rolling(window=period).mean()
    df['STD'] = df['close'].rolling(window=period).std()
    df['Upper'] = df['MA20'] + std_mult * df['STD']
    df['Lower'] = df['MA20'] - std_mult * df['STD']

    # 布林帶寬度
    df['Width'] = df['Upper'] - df['Lower']
    width_mean = df['Width'].rolling(window=period).mean()

    # 訊號欄位
    df['signal'] = 0.0

    # 取最後兩根K線判斷
    last_two = df.iloc[-2:]

    # 判斷價格突破上下軌（貼近也算）
    for idx, row in last_two.iterrows():
        sig = 0
        if row['close'] >= row['Upper']:
            sig += 1
        elif row['close'] <= row['Lower']:
            sig -= 1

        # 判斷寬度收縮：當前寬度 < 過去平均寬度的70%
        if row['Width'] < 0.7 * width_mean.loc[idx]:
            if sig > 0:
                sig += 0.5  # 多頭收縮強化
            elif sig < 0:
                sig -= 0.5  # 空頭收縮強化
            else:
                # 如果沒突破，但寬度收縮仍判定弱訊號
                # 根據你說的，只要寬度收縮也要回傳信號，
                # 我這邊用 ±0.5 作區分
                # 多頭趨勢不明，先不回正訊號，這邊可調整
                pass

        df.at[idx, 'signal'] = sig

    # 最後取倒數兩根信號總和，限幅 ±1.5
    total_signal = df['signal'].iloc[-2:].sum()
    if total_signal > 1.5:
        total_signal = 1.5
    elif total_signal < -1.5:
        total_signal = -1.5

    # 取整數或半分數，方便後續整合
    if abs(total_signal) == 1.5:
        return 1.5 if total_signal > 0 else -1.5
    elif abs(total_signal) >= 1.0:
        return 1.0 if total_signal > 0 else -1.0
    elif abs(total_signal) >= 0.5:
        return 0.5 if total_signal > 0 else -0.5
    else:
        return 0

def format_price(price: float) -> str:
    if price >= 1:
        return f"{price:,.2f}"
    elif price >= 0.1:
        return f"{price:,.4f}"
    elif price >= 0.01:
        return f"{price:,.5f}"
    elif price >= 0.001:
        return f"{price:,.6f}"
    else:
        return f"{price:,.7f}"
    
def calculate_trade_parameters_1h(symbol: str, current_price: float, direction: str, atr ,intensity: str = "normal"):
    """
    計算進場點位、槓桿倍率、止損、止盈
    direction: "bull" (看漲), "bear" (看跌)
    intensity: "normal" (一般), "strong" (強力)
    """

    base_coin = symbol.split("-")[0].upper()

    # 進場點位百分比（BTC用0.4%，其他0.8%）
    entry_pct = 0.5 * atr

    # 進場點位計算
    if direction == "bull":
        entry_price = current_price  - entry_pct
    elif direction == "bear":
        entry_price = current_price  + entry_pct
    else:
        # 不明方向
        return None

    # 槓桿倍率設定
    if intensity == "normal":
        if base_coin == "BTC":
            leverage = 10
        elif base_coin in ("ETH", "BNB"):
            leverage = 5
        else:
            leverage = 3
    elif intensity == "strong":
        if base_coin == "BTC":
            leverage = 15
        elif base_coin in ("ETH", "BNB"):
            leverage = 8
        else:
            leverage = 5
    else:
        leverage = 3  # 預設


    # 止損點位計算
    if direction == "bull":
        stop_loss_price = entry_price -2*atr
    else:  # bear
        stop_loss_price = entry_price +2*atr

    # 止盈百分比及分批出場（40% 40% 20%）
   
    tp1_pct, tp2_pct, tp3_pct = 2*atr, 4*atr, 8*atr
    

    # 止盈點位計算（看漲看跌反向計算）
    if direction == "bull":
        tp1 = entry_price + tp1_pct
        tp2 = entry_price + tp2_pct
        tp3 = entry_price + tp3_pct
    else:
        tp1 = entry_price - tp1_pct
        tp2 = entry_price - tp2_pct
        tp3 = entry_price - tp3_pct

    take_profit = [
        (tp1, 0.4),
        (tp2, 0.4),
        (tp3, 0.2),
    ]

    return {
        "entry_price": round(entry_price, 4),
        "leverage": leverage,
        "stop_loss": round(stop_loss_price, 4),
        "take_profit": [(round(p, 4), ratio) for p, ratio in take_profit]
    }


def calculate_trade_parameters_15m(symbol: str, current_price: float, direction: str, atr, intensity: str = "normal"):
    """
    計算進場點位、槓桿倍率、止損、止盈
    direction: "bull" (看漲), "bear" (看跌)
    intensity: "normal" (一般), "strong" (強力)
    """

    base_coin = symbol.split("-")[0].upper()

    # 進場點位百分比（BTC用0.5%，其他1%）
    entry_pct = 0.5 * atr

    # 進場點位計算
    if direction == "bull":
        entry_price = current_price  - entry_pct
    elif direction == "bear":
        entry_price = current_price  + entry_pct
    else:
        # 不明方向
        return None

    # 槓桿倍率設定
    if intensity == "normal":
        if base_coin == "BTC":
            leverage = 10
        elif base_coin in ("ETH", "BNB"):
            leverage = 5
        else:
            leverage = 3
    elif intensity == "strong":
        if base_coin == "BTC":
            leverage = 15
        elif base_coin in ("ETH", "BNB"):
            leverage = 8
        else:
            leverage = 5
    else:
        leverage = 3  # 預設

    
    if direction == "bull":
        stop_loss_price = entry_price - 2*atr
    else:
        stop_loss_price = entry_price + 2*atr

    if direction == "bull":
        tp1 = entry_price + 2*atr
        tp2 = entry_price + 4*atr
        tp3 = entry_price + 8*atr
    else:
        tp1 = entry_price - 2*atr
        tp2 = entry_price - 4*atr
        tp3 = entry_price - 8*atr

    take_profit = [
        (tp1, 0.4),
        (tp2, 0.4),
        (tp3, 0.2),
    ]

    return {
        "entry_price": round(entry_price, 4),
        "leverage": leverage,
        "stop_loss": round(stop_loss_price, 4),
        "take_profit": [(round(p, 4), ratio) for p, ratio in take_profit]
    }



async def ADX(symbol, interval, period=14):
    try:
        async with BingXAsyncClient(api_key=api_key, api_secret=api_secret) as client:
            res = await client.swap.kline_candlestick_data(
                symbol=symbol,
                interval=interval,
                limit=period + 20
            )

        data = [k.__dict__ for k in res.data]
        df = pd.DataFrame(data)
        df = df.sort_values('time')
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)

        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        # True Range 計算
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

        # +DM / -DM 計算
        df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                             np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                             np.maximum(df['low'].shift(1) - df['low'], 0), 0)

        # Wilder smoothing（指數平滑移動平均）
        tr14 = df['TR'].ewm(span=period, adjust=False).mean()
        plus_dm14 = df['+DM'].ewm(span=period, adjust=False).mean()
        minus_dm14 = df['-DM'].ewm(span=period, adjust=False).mean()

        # DI與DX
        plus_di14 = 100 * (plus_dm14 / tr14)
        minus_di14 = 100 * (minus_dm14 / tr14)
        dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14).replace(0, np.nan)

        # ADX = DX 的 EMA
        adx = dx.ewm(span=period, adjust=False).mean()

        last_adx = adx.iloc[-1]

        
        if interval == "1h":
            if last_adx < 25:
                return 0
            elif last_adx > 90:
                return 1.8
            else:
                return round(0.8 + (last_adx - 25) / 65, 2)

        elif interval == "15m":
            if last_adx < 20:
                return 0
            elif last_adx > 60:
                return 1.8
            else:
                return round(0.8 + (last_adx - 20) * (1.8 - 0.8) / (60 - 20), 2)

        else:
            return 1.00
    except Exception as e:
        print(f"❗ 計算 {symbol} ADX 時發生錯誤：{e}")
        return 1.00  # 安全預設值

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
    "BTC-USDT",      "ETH-USDT",      "DOT-USDT",      "SOL-USDT",       "XRP-USDT",
    "AAVE-USDT",     "INJ-USDT",      "CRV-USDT",      "LINK-USDT",      "OM-USDT",
    "CHZ-USDT",      "THETA-USDT",    "NEAR-USDT",     "VET-USDT",       "AVAX-USDT",
    "FIL-USDT",      "ICP-USDT",      "BNB-USDT",      "ALGO-USDT",      "GRT-USDT",
    "OP-USDT",       "HBAR-USDT",     "ARB-USDT",      "MANA-USDT",      "APT-USDT",
    "GALA-USDT",     "LDO-USDT",      "SAND-USDT",     "ATOM-USDT",      "XLM-USDT",
    "ADA-USDT",      "TRX-USDT",      "UNI-USDT",      "MKR-USDT",       "SNX-USDT",
    "DYDX-USDT",     "API3-USDT",     "RUNE-USDT",     "1000PEPE-USDT",  "DOGE-USDT",
    "SHIB-USDT",     "LUNC-USDT",     "WOO-USDT",      "BCH-USDT",       "ETC-USDT",
    "COTI-USDT",     "BLUR-USDT",     "1INCH-USDT",    "SUI-USDT",       "KAVA-USDT",
    "TRB-USDT",      "SFP-USDT",      "GMT-USDT",      "YGG-USDT",       "FLOW-USDT",
    "TWT-USDT",      "KSM-USDT",      "BAT-USDT",      "CFX-USDT",       "RVN-USDT",
    "FXS-USDT",      "STORJ-USDT",    "JOE-USDT",      "HIGH-USDT",      "ID-USDT",
    "SSV-USDT",      "HOOK-USDT",     "RDNT-USDT",     "RENDER-USDT",    "TONCOIN-USDT",
    "SKL-USDT",      "PHA-USDT",      "MASK-USDT",     "CELO-USDT",      "ACH-USDT",
    "PERP-USDT",     "CVC-USDT",      "CELR-USDT",     "COMP-USDT",      "ZIL-USDT",
    "ENJ-USDT",      "ANKR-USDT",     "GLM-USDT",      "DEGO-USDT",      "ASTR-USDT",
    "NEO-USDT",      "MTL-USDT",      "TRU-USDT",      "BNT-USDT",       "ENA-USDT",
    "TROLLSOL-USDT", "PI-USDT",       "VINE-USDT",     "AGT-USDT",       "PUMP-USDT",
    "IP-USDT",       "TIA-USDT",      "PENGU-USDT",    "OL-USDT"
]

# 權重列表，對應指標順序：MA, BE_BIG, MACD, RSI, THREE, BREAK_OUT, KDJ, BOLL
weights = [3, 4.5, 2, 2, 1, 1.5, 1,2]

skip_counts_1h = {}  # 全域字典，記錄幣種跳過次數
skip_counts_15m = {}


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

async def ATR(symbol, period=14,timeframe="1h"):
    try:
        kline_data = await get_kline_data(symbol, period + 1,timeframe)
        if not kline_data or len(kline_data) < period + 1:
            return None

        tr_list = []
        for i in range(1,len(kline_data)):
            high = float(kline_data[i]['high'])
            low = float(kline_data[i]['low'])
            close_prev = float(kline_data[i-1]['close'])
            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
            tr_list.append(tr)
        
        atr = np.mean(tr_list[-period:])
        return atr
    except Exception as e:
        print(f"❗ 計算 {symbol} ATR時發生錯誤：{e}")
        return None

emoji_map = {
    "BTC": "•",        "ETH": "•",        "DOT": "•",        "SOL": "•",        "XRP": "•",
    "AAVE": "•",       "INJ": "•",        "CRV": "•",        "LINK": "•",       "OM": "•",
    "CHZ": "•",        "THETA": "•",      "NEAR": "•",       "VET": "•",        "AVAX": "•",
    "FIL": "•",        "ICP": "•",        "BNB": "•",        "ALGO": "•",       "GRT": "•",
    "OP": "•",         "HBAR": "•",       "ARB": "•",        "MANA": "•",       "APT": "•",
    "GALA": "•",       "LDO": "•",        "SAND": "•",       "ATOM": "•",       "XLM": "•",
    "ADA": "•",        "TRX": "•",        "UNI": "•",        "MKR": "•",        "SNX": "•",
    "DYDX": "•",       "API3": "•",       "RUNE": "•",       "1000PEPE": "•",   "DOGE": "•",
    "SHIB": "•",       "LUNC": "•",       "WOO": "•",        "BCH": "•",        "ETC": "•",
    "COTI": "•",       "BLUR": "•",       "1INCH": "•",      "SUI": "•",        "KAVA": "•",
    "TRB": "•",        "SFP": "•",        "GMT": "•",        "YGG": "•",        "FLOW": "•",
    "TWT": "•",        "KSM": "•",        "BAT": "•",        "CFX": "•",        "RVN": "•",
    "FXS": "•",        "STORJ": "•",      "JOE": "•",        "HIGH": "•",       "ID": "•",
    "SSV": "•",        "HOOK": "•",       "RDNT": "•",       "RENDER": "•",     "TONCOIN": "•",
    "SKL": "•",        "PHA": "•",        "MASK": "•",       "CELO": "•",       "ACH": "•",
    "PERP": "•",       "CVC": "•",        "CELR": "•",       "COMP": "•",       "ZIL": "•",
    "ENJ": "•",        "ANKR": "•",       "GLM": "•",        "DEGO": "•",       "ASTR": "•",
    "NEO": "•",        "MTL": "•",        "TRU": "•",        "BNT": "•",        "ENA": "•",
    "TROLLSOL": "•",   "PI": "•",         "VINE": "•",       "AGT": "•",        "PUMP": "•",
    "IP": "•",         "TIA": "•",        "PENGU": "•",      "OL": "•"
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

async def evaluate_symbol_1h(symbol):

    # 如果該幣跳過計數 > 0，直接跳過並扣減一次
    if skip_counts_1h.get(symbol, 0) > 0:
        skip_counts_1h[symbol] -= 1
        print(f"跳過 {symbol} 偵測，剩餘跳過次數：{skip_counts_1h[symbol]}")
        return  # 不做評估
    indicators = ['MA', 'BE_BIG', 'MACD', 'RSI', 'THREE', 'BREAK_OUT', 'KDJ','BOLL']
    scores = [
        await MA(symbol,interval="1h"),
        await BE_BIG(symbol,interval="1h"),
        await MACD(symbol,interval="1h"),
        await RSI(symbol,interval="1h"),
        await THREE(symbol,interval="1h"),
        await BREAK_OUT(symbol,interval="1h"),
        await KDJ(symbol,interval="1h"),
        await BOLL(symbol,interval="1h")
        
    ]
    total_score = sum(s * w for s, w in zip(scores, weights))
    adx = await ADX(symbol,interval="1h")
    total_score = total_score*adx
    current_price = await get_current_price(symbol)
    atr2 = await ATR(symbol)
    atr=format_price(atr2)
    # 幣名簡化
    short = symbol.split("-")[0]
    emoji = emoji_map.get(short, "")
    triggered_indicators = [name for name, score in zip(indicators, scores) if score != 0]

    # 轉成字串（用逗號分隔）
    indicators_str = ", ".join(triggered_indicators) if triggered_indicators else "無"

    # 判斷進場方向
    if total_score >= 18:
        direction_text = "🔥🔥 📉 **強力進多** 🔥🔥"
        direction = "bull"
        intensity = "strong"
    elif total_score >= 13:
        direction_text = "📈 **看漲進場**"
        direction = "bull"
        intensity = "normal"
    elif total_score <= -18:
        direction_text = "🔥🔥 📈 **強力進空** 🔥🔥"
        direction = "bear"
        intensity = "strong"
    elif total_score <= -13:
        direction_text = "📉 **看跌進場**"
        direction = "bear"
        intensity = "normal"
    else:
        return 0
    skip_counts_1h[symbol] = 8
    
    # 處理ATR顯示
    '''atr_info = f"📏 ATR: {atr:,.3f}  " \
               f"1.5: {atr*1.5:,.3f}  " \
               f"3: {atr*3:,.3f}\n" if atr is not None else "📏 ATR: 無法計算\n"'''
    trade_params = calculate_trade_parameters_1h(symbol, current_price, direction,atr2, intensity)
    if trade_params is None:
        return 0

    entry_price = trade_params["entry_price"]
    leverage = trade_params["leverage"]
    stop_loss = trade_params["stop_loss"]
    take_profit = trade_params["take_profit"]

    # 將進場點、槓桿、止損、止盈加入訊息
    bingx_ratios = [40, 66, 100]  # 對應三段出場

    tp_str = "\n".join([
        f"止盈{int(ratio*100)}%：${format_price(price)}   🔸拉 {bingx}%"
        for (price, ratio), bingx in zip(take_profit, bingx_ratios)
    ])
    extra_info = (
        f"🚀 進場點位: ${entry_price}\n"
        f"🎯 槓桿倍率: {leverage}倍\n"
        f"🛑 止損: ${format_price(stop_loss)}\n"
        f"{tp_str}\n"
    )
    # 組合訊息
    message = (
        f"!!🚨注意🚨!! 🕐時區為1H🕐!!\n"
        f"{emoji} `{symbol}`\n"
        f"💰 現價：${format_price(current_price)}\n"
        f"📊 總分：{total_score:.2f}\n"
        f"{direction_text}\n"
        f"{extra_info}"
        f"📏 ATR: {atr}\n"
        f"📌 進場依據：{indicators_str}"
    )

    await send_to_discord(message)


async def evaluate_symbol_15m(symbol):

    # 如果該幣跳過計數 > 0，直接跳過並扣減一次
    if skip_counts_15m.get(symbol, 0) > 0:
        skip_counts_15m[symbol] -= 1
        print(f"跳過 {symbol} 偵測，剩餘跳過次數：{skip_counts_15m[symbol]}")
        return  # 不做評估
    indicators = ['MA', 'BE_BIG', 'MACD', 'RSI', 'THREE', 'BREAK_OUT', 'KDJ','BOLL']
    scores = [
        await MA(symbol,interval="15m"),
        await BE_BIG(symbol,interval="15m"),
        await MACD(symbol,interval="15m"),
        await RSI(symbol,interval="15m"),
        await THREE(symbol,interval="15m"),
        await BREAK_OUT(symbol,interval="15m"),
        await KDJ(symbol,interval="15m"),
        await BOLL(symbol,interval="15m")
        ]
    total_score = sum(s * w for s, w in zip(scores, weights))
    adx = await ADX(symbol,interval="15m")
    total_score = total_score*adx
    current_price = await get_current_price(symbol)
    atr2 = await ATR(symbol,period=14, timeframe="15m")
    atr=format_price(atr2)
    # 幣名簡化
    short = symbol.split("-")[0]
    emoji = emoji_map.get(short, "")
    triggered_indicators = [name for name, score in zip(indicators, scores) if score != 0]

    # 轉成字串（用逗號分隔）
    indicators_str = ", ".join(triggered_indicators) if triggered_indicators else "無"

    # 判斷進場方向
    if total_score >= 18:
        direction_text = "🔥🔥 📉 **強力進多** 🔥🔥"
        direction = "bull"
        intensity = "strong"
    elif total_score >= 13:
        direction_text =  "📈 **看漲進場**"
        direction = "bull"
        intensity = "normal"
    elif total_score <= -18:
        direction_text = "🔥🔥 📈 **強力進空** 🔥🔥"
        direction = "bear"
        intensity = "strong"
    elif total_score <= -13:
        direction_text = "📉 **看跌進場**"
        direction = "bear"
        intensity = "normal"
    else:
        return 0
    skip_counts_15m[symbol] = 8
    
    # 處理ATR顯示
    '''atr_info = f"📏 ATR: {atr:,.3f}  " \
               f"1.5: {atr*1.5:,.3f}  " \
               f"3: {atr*3:,.3f}\n" if atr is not None else "📏 ATR: 無法計算\n"'''
    
    trade_params = calculate_trade_parameters_15m(symbol, current_price, direction, atr2, intensity)
    if trade_params is None:
        return 0

    entry_price = trade_params["entry_price"]
    leverage = trade_params["leverage"]
    stop_loss = trade_params["stop_loss"]
    take_profit = trade_params["take_profit"]

    # 將進場點、槓桿、止損、止盈加入訊息
    bingx_ratios = [40, 66, 100]  # 對應三段出場

    tp_str = "\n".join([
        f"止盈{int(ratio*100)}%：${format_price(price)}   🔸拉 {bingx}%"
        for (price, ratio), bingx in zip(take_profit, bingx_ratios)
    ])
    extra_info = (
        f"🚀 進場點位: ${entry_price}\n"
        f"🎯 槓桿倍率: {leverage}倍\n"
        f"🛑 止損: ${format_price(stop_loss)}\n"
        f"{tp_str}\n"
    )
    # 組合訊息
    message = (
        f"!!🚨注意🚨!!🕐時區為15m🕐!!\n"
        f"{emoji} `{symbol}`\n"
        f"💰 現價：${format_price(current_price)}\n"
        f"📊 總分：{total_score:.2f}\n"
        f"{direction_text}\n"
        f"{extra_info}"
        f"📏 ATR: {atr}\n"
        f"📌 進場依據：{indicators_str}"
    )

    await send_to_discord(message)


async def get_fgi():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url)
        data = response.json()
        value = data['data'][0]['value']
        value_classification = data['data'][0]['value_classification']
        timestamp = data['data'][0]['timestamp']
        readable_time = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
        return value, value_classification, readable_time
    except Exception as e:
        return None, None, None

async def job():
    value, classification, time_str = await get_fgi()
    value = float(value)
    if value>=75:
        msg = f"🔥 Fear & Greed Index: {value} 🔥注意風險🔥 ({classification})\n時間: {time_str}🔥注意風險🔥"
    elif value<=25:
        msg = f"🧊 Fear & Greed Index: {value} 🧊注意風險🧊 ({classification})\n時間: {time_str}🧊注意風險🧊"
    elif value>25 and value<75:
        msg = f"📊 Fear & Greed Index: {value} ({classification})\n時間: {time_str}"
    else:
        msg = "❌ 取得 Fear & Greed Index 失敗"
    
    await send_to_discord(msg)

async def scheduler():
    while True:
        now = datetime.datetime.now()
        target = now.replace(hour=8, minute=8, second=0, microsecond=0)
        if now > target:
            target += datetime.timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        await job()

async def run_loop_1h():
    await send_to_discord("💡 搜幣程式啟動！")
    while True:

        for sym in symbols:
            await evaluate_symbol_1h(sym)
            await asyncio.sleep(5)

        
        '''print("等待 12 分鐘後重新判斷...\n")
        await asyncio.sleep(720)  # 12分鐘'''

async def run_loop_15m():
    while True:

        for sym in symbols:
            await evaluate_symbol_15m(sym)
            await asyncio.sleep(0.75)

        
        '''print("等待 3 分鐘後重新判斷...\n")
        await asyncio.sleep(180)  # 3分鐘'''
        
async def run_loop_forever():
    await asyncio.gather(
        run_loop_1h(),
        run_loop_15m(),
        scheduler(),
    )        

if __name__ == "__main__":
    asyncio.run(run_loop_forever())



