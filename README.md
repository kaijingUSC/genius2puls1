# genius2puls1 STOCK PREDICTION

## Final Prodcut is in directory 'Flask_app'

## Detailed implementation information

Close price from 2020-4-18 to 2021-4-15

1. Basic analysis and visualization

    stock_basic_visualization.py

  - Moving average chart
  - Rate of return chart
  - Income distribution scatter chart
  - Correlation heat map
  - Fast scatter plot of stock risk and return

2. Model selection

    kaggle_test1.py

  - LSTM (vanila + encoder+decoder)
  - SVM
  - Arima
    Evaluation: RSME; MAPE; 
    
3. News text crawler

    news_crawler.py

    Crawling company's news from financial post website

4. News data preparation and sentiment analysis

    news_data_preparation_and_sentiment_analysis.py
    
5. Merge stock price data with sentiment signals

    data_merge.py

5. LSTM model

    1. LSTM_model.py 

    2. LSTM_model_with_news.py
