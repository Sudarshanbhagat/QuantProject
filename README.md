# IV-Alpha Stock Selection Pipeline

## What This Is

A machine learning model that picks stocks likely to outperform over the next 5 days, specifically targeting those with interesting IV (implied volatility) behavior. Think of it as a pattern detector built to catch when options markets are saying something different than stock prices.

Bottom line: Train a LightGBM model on 1.5+ years of stock data, make probability predictions, and find trades where the odds are in your favor.

## Why I Built This

I noticed options market data moves before stock prices sometimes. IV skew, term structure, rank changes - these things seemed to predict price moves. Instead of guessing which patterns matter, I let the data tell me by building a gradient boosting classifier.

## The Real Story

Starting point: Raw OHLCV data for 82 US stocks from April 2023 to January 2026. About 40k records total.

The problem: 40k rows is not a lot for machine learning. Easy to fool yourself into thinking you found a signal when it's just noise. Had to be careful.

What I did:
1. Built 5 custom features around IV behavior (rank net IV, IV skew ranking, IV momentum, price-IV divergence, term premium)
2. Combined them with basic OHLCV + market data (28 features total)
3. Split data by time - train on everything before Dec 31, 2024, validate on 2025+ data
4. Trained a LightGBM classifier to predict "next 5-day return positive or negative?"
5. Tested it against proper baselines (shuffle test, 1-bar lag test, VIX rule comparison)
6. Generated 12,421 predictions on the validation set

What worked:
- stock_iv_7d was the strongest predictor (not surprising, but good to confirm)
- Time-series split caught real issues early (helped me avoid data leakage)
- Class weighting (3.72x for minority class) actually improved the model
- Threshold tuning to 0.30 instead of 0.50 made predictions more useful

What didn't:
- Pure accuracy (51.28%) was misleading with imbalanced classes
- Had to focus on Recall (70%) and AUC (0.6072) instead to understand if the model was real
- Shuffle test showed model wasn't just memorizing market timing (good news)

## What The Numbers Mean

```
Model Performance on Validation Set (Jan 2025 - Jan 2026):
- AUC: 0.6072 (can rank-order predictions better than random)
- Recall: 70.02% (catches 70% of actual opportunities)  
- Precision: 26.13% (about 1 in 4 trades profitable)
- Generated: 8,705 trade signals across 12,421 predictions
```

Is this good? Depends on your expectations. It's better than flipping a coin (0.50 AUC). Lag test showed it'd work with 1-bar execution lag. Baseline comparison with VIX rule was... different, not clearly better. Which is honest - it's a complicated optimizer, not magic.

## Files You Get

- Alpha_Pipeline.ipynb - The whole thing. Raw code, training, testing, exports
- validation_predictions.csv - 12,421 predictions with timestamps and probabilities  
- IV_Alpha_Pipeline_Results_V2.png - Chart showing feature importance + equity curve
- final_data_with_targets.csv - The data itself (40,869 records)

## How To Use It

1. Just want to see if it works?
   Run the notebook. 17 cells handle everything end-to-end. Takes a few minutes.

2. Want to deploy it?
   Take validation_predictions.csv into your trading system. Pick a threshold (I used 0.30). Size positions small - this is not a guaranteed money printer, it's a probabilistic edge. Start paper trading, monitor monthly.

3. Want to retrain on new data?
   Load new OHLCV + IV data into final_data_with_targets.csv, run the same notebook. It uses time-series validation split automatically, so no data leakage risk with rolling windows.

4. Want to understand it?
   Cells 3-4 show the feature engineering. Cells 5-6 show training. Cells 7-8 show testing against baselines. Cells 16-20 show verification - how I actually checked that the code and model are correct.

## Check Your Work (Verification)

I added 4 verification methods to the notebook. Actually run them - they're the difference between thinking something works and knowing it works:

- Data Quality: Count records, check for duplicates, verify no weird missing values
- Code Logic: Confirm features are in valid ranges, confirm time-series split has zero leakage
- Model Performance: Check predictions are in [0,1], confusion matrix adds up, metrics make sense
- Robustness: Run the model on first 10 stocks separately, check monthly performance, look at feature variance

If these all pass, the code is correct. If they don't, something's wrong before you trade a penny.

## The Honest Assessment

What this actually is: A classifier that learned patterns from 1.5 years of historical data that might happen again. Recall is high (catches opportunities) but precision is low (lots of false alarms). Works in backtesting on data it's never seen ($val set).

What this might not be: A reliable source of alpha after fees and slippage. The 26% precision means you're losing money on 3/4 trades on average. The 70% recall means you're catching 70% of real moves. Whether that combines to a net edge depends on your position sizing and execution quality.

Why I'm sharing this honestly: If you're going to risk money on this, you need to know what you're buying. It's a probabilistic edge detector, not a money machine. But honestly - that's more than most people have.

## Next Steps If You Run This

1. Paper trade it on a few stocks for a month
2. Compare your realized returns to the prediction distribution
3. Track feature drift - recompute engineered features monthly, see if they're changing
4. If it actually makes money in simulation, size real positions tiny at first (0.5% risk per trade max)
5. Retrain quarterly using the time-series split logic (always train on older data, validate on newer)

## Questions I Know You'll Have

"Can I trust this?"  
Only to the extent of the verification cells passing. Run them. Don't skip them. They're not pretty but they're honest.

"What if markets change?"  
They will. Feature drift is real. This model learned patterns from 2023-2025. If market structure changes, this breaks. That's why it needs quarterly retraining, not annual.

"Is 70% recall actually good?"  
Depends. If you have 100 real opportunities and catch 70 of them, you're doing better than random. But if each false alarm costs you 50 bps in execution, then 26% precision hurts. You do the math for your costs.

"Should I go all-in?"  
No. Absolutely not. This is data-driven but not proven in live trading. Start with paper, then micro positions.

## The Code

It's all in the notebook. 20 cells:
- 1-4: Data loading and feature engineering (this is where the magic begins)
- 5-6: Validation split and model training (here's where we build the classifier)
- 7-8: Backtesting and risk analysis (does it actually work?)
- 9-12: Testing framework (passes 10 tests or fails them)
- 13-20: Verification methods (how we know it's not broken)

No dependencies except standard stuff - pandas, scikit-learn, lightgbm, numpy, matplotlib. Nothing exotic.

## Real Talk

This took work. Not because the code is complicated - it's pretty straightforward - but because good machine learning on financial data is mostly about validation. Making sure you didn't fool yourself. Knowing the difference between a real edge and getting lucky.

The verification cells are the real value here. Anyone can build a model that works on historical data. Not everyone checks whether they actually prevented data leakage or whether their predictions are miscalibrated.

If you run this and it works, great. If it doesn't, the verification cells will tell you why. That's the whole point.

---

Last updated: January 2026  
Model:LightGBM Binary Classifier  
Data:40,869 records, 82 stocks, April 2023 - January 2026  
Status:Tested, documented, ready to validate in your own trading environment
