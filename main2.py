import numpy as np
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import ccxt
from datetime import datetime, timedelta


def fetch_data(exchange, symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def prepare_features(df):
    df['returns'] = df['close'].pct_change()
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, accuracy


def predict_next_7_days(model, scaler, latest_data):
    predictions = []
    for _ in range(7):
        latest_data_scaled = scaler.transform(latest_data)
        prediction = model.predict(latest_data_scaled)[0]
        predictions.append(prediction)
        # 模拟下一天的数据，这里简单地使用最后一天的数据
        latest_data = latest_data.copy()
    return predictions


def place_order(exchange, symbol, side, amount):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        print(f"订单已下: {order}")
    except Exception as e:
        print(f"下单失败: {e}")


def main():
    exchange = ccxt.gate({
        'apiKey': 'YOUR_API_KEY',
        'secret': 'YOUR_SECRET_KEY',
        'enableRateLimit': True,
    })
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 1000

    df = fetch_data(exchange, symbol, timeframe, limit)
    df = prepare_features(df)

    features = ['returns', 'sma_10', 'sma_30', 'rsi']
    X = df[features]
    y = df['target']

    model, scaler, accuracy = train_model(X, y)
    print(f"模型准确率: {accuracy:.2f}")

    latest_data = X.iloc[-1].values.reshape(1, -1)
    predictions = predict_next_7_days(model, scaler, latest_data)

    print("未来7天预测:")
    for i, pred in enumerate(predictions):
        date = datetime.now() + timedelta(days=i+1)
        print(f"{date.date()}: {'上涨' if pred == 1 else '下跌或不变'}")

    # 根据第一天的预测进行交易
    if predictions[0] == 1:
        place_order(exchange, symbol, 'buy', 0.001)  # 买入0.001 BTC
    else:
        place_order(exchange, symbol, 'sell', 0.001)  # 卖出0.001 BTC


if __name__ == "__main__":
    main()
