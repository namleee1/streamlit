#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# Tải file lên Streamlit
uploaded_file = st.file_uploader("Chọn file dữ liệu Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Đọc file Excel
    df = pd.read_excel(uploaded_file, sheet_name="Data- refinitiv")
    df1 = pd.read_excel(uploaded_file, sheet_name="Sheet2")

    df["Date"] = pd.to_datetime(df["Date"])
    df1["Date"] = pd.to_datetime(df1["Date"])
    df1["VND"] = df1["VND"].astype(str).str.replace(" ", "").astype(float)
    df = df.sort_values("Date")

    # Hiển thị biểu đồ
    for column in ["FEDRATE", "DXY", "VND", "OMOrate", "SBVcentralrate"]:
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df[column], marker='+', color='b')
        ax.set_title(f"{column} CHANGES OVER TIME")
        ax.set_xlabel("DATE")
        ax.set_ylabel(column)
        st.pyplot(fig)

    # Hiển thị ma trận tương quan
    correlation_matrix = df.corr()
    st.write("### Ma trận tương quan:")
    st.write(correlation_matrix)

    # Người dùng nhập chỉ số và số ngày
    name = st.text_input("Nhập chỉ số muốn xem dữ liệu lịch sử:")
    n_weeks = st.number_input("Nhập số tuần muốn xem:", min_value=1, step=1)
    n_days = n_weeks * 7

    if name in df.columns:
        st.write(df[["Date", name]].iloc[:n_days])

    # Chuẩn bị dữ liệu huấn luyện mô hình XGBoost
    train = df[df["Date"].dt.year < 2024].drop(columns=["Date"])
    test = df[df["Date"].dt.year == 2024].drop(columns=["Date"])
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    num_features = train.shape[1]

    def xgb_model(train_scaled, test_scaled, scaler, num_features):
        X_train = train_scaled[:, 1:]  
        y_train = train_scaled[:, 2]  # Cột VND
        X_test = test_scaled[:, 1:]

        model = XGBRegressor(n_estimators=200, learning_rate=0.06, max_depth= 8)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        temp = np.zeros((len(pred), num_features))  
        temp[:, 2] = pred  # Cột thứ 3 (index 2) chứa VND
        return scaler.inverse_transform(temp)[:, 2]

    xgb_pred = xgb_model(train_scaled, test_scaled, scaler, num_features)

        # Hiển thị dự báo
    forecast_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')
    st.write("### 📊 Kết quả dự báo XGBoost:")
    st.write("Ngày       | Dự báo   | Xu hướng | % Thay đổi")
    st.write("-------------------------------------------")

    for i in range(1, n_days):
            date_str = forecast_dates[i].strftime("%d-%m-%Y")
            prev_value = xgb_pred[i - 1]
            curr_value = xgb_pred[i]
            change_percent = ((curr_value - prev_value) / prev_value) * 100
            trend = "📈 Up" if curr_value > prev_value else "📉 Down"
            st.write(f"{date_str} | {curr_value:.2f} | {trend} | {change_percent:.2f}%")

    # Vẽ biểu đồ dự báo
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, df1["VND"][:n_days].values, label="Thực tế", color="blue")
    plt.plot(forecast_dates, xgb_pred[:n_days], label="XGBoost", linestyle="dashed", color="purple")
    plt.xlabel("Ngày")
    plt.ylabel("Tỷ giá VND-USD")
    plt.title("Dự báo tỷ giá VND-USD")
    plt.legend()
    st.pyplot(plt)

    EMAIL_SENDER = "namltmta@gmail.com"
    EMAIL_PASSWORD = "jlxk sqlk gckc eqzz"  # Không nên lưu mật khẩu trực tiếp, hãy dùng biến môi trường
    EMAIL_RECEIVER = st.text_input('Nhập email người nhận:')

    if EMAIL_RECEIVER:
            email_content = "<h2>Dự báo tỷ giá VND-USD tuần tới</h2><table border='1' cellpadding='5'>"
            email_content += "<tr><th>Ngày</th><th>Dự đoán giá</th></tr>"
            for date, price in zip(forecast_dates, xgb_pred):
                email_content += f"<tr><td>{date}</td><td>{price:.2f} USD</td></tr>"
            email_content += "</table>"

            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECEIVER
            msg['Subject'] = "Báo cáo dự báo tỷ giá VND-USD tuần tới"
            msg.attach(MIMEText(email_content, 'html'))

            try:
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
                server.quit()
                st.success("✅ Email đã được gửi thành công!")
            except Exception as e:
                st.error(f"❌ Lỗi khi gửi email: {str(e)}")

