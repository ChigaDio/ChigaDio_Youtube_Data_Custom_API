#!/usr/bin/env python3
"""
改善版YouTube配信予測システム

主な改善点:
1. データの問題を修正（Shortsを除外し、実際の配信のみを対象）
2. 時系列データとして正しく扱う（データリーケージを防ぐ）
3. クラス不均衡問題に対処
4. より高度な特徴量エンジニアリング
5. ハイパーパラメータの最適化
6. アンサンブル学習
7. 配信時間の予測
8. 視聴者数の予測
9. 詳細な評価とバリデーション
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 設定
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

class AdvancedStreamPredictionSystem:
    """改善版配信予測システム"""
    
    def __init__(self):
        self.stream_classifier = None
        self.time_regressor = None
        self.viewers_regressor = None
        self.scaler = None
        self.feature_names = None
        
    def load_and_analyze_data(self, filepath):
        """データの読み込みと詳細分析"""
        print("=" * 80)
        print("データ読み込みと分析")
        print("=" * 80)
        
        df = pd.read_csv(filepath)
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        # 実際の配信かどうかの判定
        # actual_start_timeがあり、duration_secondsが一定以上のものを「配信」とする
        df['is_stream'] = (
            (df["is_short"] == False) &
            (df['actual_start_time'].notna()) & 
            (df['duration_seconds'].fillna(0) > 600)  # 10分以上
        ).astype(int)
        
        # Shortsを除外
        df['is_short'] = df['title'].str.contains('#Shorts', case=False, na=False)
        
        print(f"\n【データ統計】")
        print(f"総レコード数: {len(df)}")
        print(f"Shorts: {df['is_short'].sum()}")
        print(f"配信（10分以上）: {df['is_stream'].sum()}")
        print(f"その他: {len(df) - df['is_short'].sum() - df['is_stream'].sum()}")
        
        # 配信データのみ抽出
        stream_df = df[df['is_stream'] == 1].copy()
        print(f"\n【配信データ分析】")
        print(f"配信数: {len(stream_df)}")
        
        # 配信時間の分析
        stream_df['start_time'] = pd.to_datetime(stream_df['actual_start_time'])
        stream_df['start_hour'] = stream_df['start_time'].dt.hour
        stream_df['start_weekday'] = stream_df['start_time'].dt.dayofweek
        
        print(f"\n配信時間帯の分布:")
        hour_dist = stream_df['start_hour'].value_counts().sort_index()
        for hour, count in hour_dist.head(24).items():
            print(f"  {hour:2d}時台: {count:3d}回 {'█' * int(count/5)}")
        
        print(f"\n曜日別配信数:")
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']
        weekday_dist = stream_df['start_weekday'].value_counts().sort_index()
        for wd, count in weekday_dist.items():
            print(f"  {weekday_names[wd]}曜日: {count:3d}回 {'█' * int(count/5)}")
        
        # 視聴者数の分析
        if 'concurrent_viewers' in stream_df.columns:
            viewers = stream_df['concurrent_viewers'].dropna()
            if len(viewers) > 0:
                print(f"\n同時視聴者数:")
                print(f"  平均: {viewers.mean():.0f}人")
                print(f"  中央値: {viewers.median():.0f}人")
                print(f"  最大: {viewers.max():.0f}人")
        
        return df, stream_df
    
    def create_timeseries_dataset(self, df):
        """時系列データセットの作成"""
        print("\n" + "=" * 80)
        print("時系列データセットの作成")
        print("=" * 80)
        
        # 日付でソート
        df = df.sort_values('published_at').reset_index(drop=True)
        
        # 全ての日付を取得（配信があった日もなかった日も）
        date_range = pd.date_range(
            start=df['published_at'].min().date(),
            end=df['published_at'].max().date(),
            freq='D'
        )
        
        # 各日付のデータを作成
        daily_data = []
        
        for date in date_range:
            # その日の配信を取得
            day_streams = df[df['published_at'].dt.date == date.date()]
            
            # 基本情報
            row = {
                'date': date,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'day_of_week': date.dayofweek,
                'day_of_year': date.timetuple().tm_yday,
                'week_of_year': date.isocalendar()[1],
                'is_weekend': 1 if date.dayofweek >= 5 else 0,
                'is_month_start': 1 if date.day <= 7 else 0,
                'is_month_end': 1 if date.day >= 24 else 0,
            }
            
            # 配信があったかどうか
            has_stream = len(day_streams[day_streams['is_stream'] == 1]) > 0
            row['has_stream'] = 1 if has_stream else 0
            
            # 配信があった場合の情報
            if has_stream:
                stream = day_streams[day_streams['is_stream'] == 1].iloc[0]
                start_time = pd.to_datetime(stream['actual_start_time'])
                row['stream_hour'] = start_time.hour
                row['stream_duration'] = stream['duration_seconds']
                row['concurrent_viewers'] = stream.get('concurrent_viewers', np.nan)
            else:
                row['stream_hour'] = np.nan
                row['stream_duration'] = np.nan
                row['concurrent_viewers'] = np.nan
            
            daily_data.append(row)
        
        daily_df = pd.DataFrame(daily_data)
        
        print(f"作成された日次データ: {len(daily_df)}日分")
        print(f"配信あり: {daily_df['has_stream'].sum()}日")
        print(f"配信なし: {(1-daily_df['has_stream']).sum()}日")
        print(f"配信率: {daily_df['has_stream'].mean()*100:.1f}%")
        
        return daily_df
    
    def create_advanced_features(self, daily_df):
        """高度な特徴量エンジニアリング"""
        print("\n" + "=" * 80)
        print("高度な特徴量エンジニアリング")
        print("=" * 80)
        
        features_df = daily_df.copy()
        
        # 1. 周期的特徴量
        print("1. 周期的特徴量を作成...")
        features_df['day_of_week_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_of_week_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        features_df['day_of_year_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['day_of_year_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
        
        # 2. 移動平均特徴量（過去の配信パターン）
        print("2. 移動平均特徴量を作成...")
        for window in [3, 7, 14, 30]:
            features_df[f'stream_ma_{window}d'] = (
                features_df['has_stream']
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)  # 未来のデータを使わない
            )
        
        # 3. ラグ特徴量（過去N日の配信状況）
        print("3. ラグ特徴量を作成...")
        for lag in [1, 2, 3, 7, 14]:
            features_df[f'stream_lag_{lag}d'] = features_df['has_stream'].shift(lag)
        
        # 4. 連続配信・非配信日数
        print("4. 連続日数特徴量を作成...")
        features_df['consecutive_stream_days'] = 0
        features_df['consecutive_no_stream_days'] = 0
        features_df['days_since_last_stream'] = 0
        
        stream_count = 0
        no_stream_count = 0
        days_since = 0
        
        for i in range(len(features_df)):
            if i > 0:
                if features_df.loc[i-1, 'has_stream'] == 1:
                    stream_count += 1
                    no_stream_count = 0
                    days_since = 0
                else:
                    stream_count = 0
                    no_stream_count += 1
                    days_since += 1
            
            features_df.loc[i, 'consecutive_stream_days'] = stream_count
            features_df.loc[i, 'consecutive_no_stream_days'] = no_stream_count
            features_df.loc[i, 'days_since_last_stream'] = days_since
        
        # 5. 曜日・月別の歴史的配信率
        print("5. 統計的特徴量を作成...")
        # Expanding mean（その時点までの累積平均）を使う
        for col in ['day_of_week', 'month']:
            grouped = features_df.groupby(col)['has_stream'].expanding().mean()
            grouped = grouped.reset_index(level=0, drop=True)
            features_df[f'{col}_stream_rate'] = grouped.shift(1).fillna(0.5)
        
        # 6. 特別な日（月初、月末、週末）の組み合わせ
        print("6. 複合特徴量を作成...")
        features_df['weekend_month_end'] = (
            features_df['is_weekend'] * features_df['is_month_end']
        )
        
        # 7. トレンド特徴量
        print("7. トレンド特徴量を作成...")
        features_df['time_index'] = np.arange(len(features_df))
        
        print(f"\n作成された特徴量: {len(features_df.columns)}個")
        
        return features_df
    
    def prepare_ml_data(self, features_df):
        """機械学習用データの準備"""
        print("\n" + "=" * 80)
        print("機械学習用データの準備")
        print("=" * 80)
        
        # 特徴量の選択
        feature_cols = [
            # 時間関連
            'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year',
            'is_weekend', 'is_month_start', 'is_month_end',
            
            # 周期的特徴量
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'day_of_year_sin', 'day_of_year_cos',
            
            # 移動平均
            'stream_ma_3d', 'stream_ma_7d', 'stream_ma_14d', 'stream_ma_30d',
            
            # ラグ特徴量
            'stream_lag_1d', 'stream_lag_2d', 'stream_lag_3d', 
            'stream_lag_7d', 'stream_lag_14d',
            
            # 連続日数
            'consecutive_stream_days', 'consecutive_no_stream_days',
            'days_since_last_stream',
            
            # 統計的特徴量
            'day_of_week_stream_rate', 'month_stream_rate',
            
            # 複合特徴量
            'weekend_month_end',
            
            # トレンド
            'time_index'
        ]
        
        # 実際に存在する列のみ選択
        feature_cols = [col for col in feature_cols if col in features_df.columns]
        
        # 欠損値を除去（最初の数日分はラグ特徴量が欠損）
        df_clean = features_df.dropna(subset=feature_cols).copy()
        
        X = df_clean[feature_cols].copy()
        y = df_clean['has_stream'].copy()
        
        # 配信時間と視聴者数（回帰用）
        stream_hour = df_clean['stream_hour'].copy()
        concurrent_viewers = df_clean['concurrent_viewers'].copy()
        
        # 日付情報を保持
        dates = df_clean['date'].copy()
        
        print(f"特徴量数: {len(feature_cols)}")
        print(f"サンプル数: {len(X)}")
        print(f"配信あり: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"配信なし: {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")
        
        self.feature_names = feature_cols
        
        return X, y, stream_hour, concurrent_viewers, dates
    
    def train_stream_classifier(self, X, y, dates):
        """配信有無の分類モデルを学習"""
        print("\n" + "=" * 80)
        print("配信有無分類モデルの学習")
        print("=" * 80)
        
        # 時系列分割（未来のデータで検証）
        tscv = TimeSeriesSplit(n_splits=5)
        
        # データのスケーリング
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ベースモデルの定義
        models = {
            'ロジスティック回帰': LogisticRegression(
                max_iter=5000, 
                class_weight="balanced",
                random_state=42,
                
            ),
            'ランダムフォレスト': RandomForestClassifier(
                n_estimators=4000,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            '勾配ブースティング': GradientBoostingClassifier(
                n_estimators=4000,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }
        
        # 各モデルの評価
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*60}")
            print(f"学習中: {name}")
            print(f"{'='*60}")
            
            # 時系列クロスバリデーション
            cv_scores = []
            cv_precisions = []
            cv_recalls = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 学習
                if name == 'ロジスティック回帰':
                    model.fit(X_train, y_train)
                else:
                    # ランダムフォレストと勾配ブースティングは元のデータを使用
                    model.fit(X.iloc[train_idx], y_train)
                    X_val = X.iloc[val_idx]
                
                # 予測
                y_pred = model.predict(X_val)
                
                # 評価
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)
                
                cv_scores.append(acc)
                cv_precisions.append(prec)
                cv_recalls.append(rec)
            
            avg_acc = np.mean(cv_scores)
            avg_prec = np.mean(cv_precisions)
            avg_rec = np.mean(cv_recalls)
            
            print(f"時系列CV (5分割) 結果:")
            print(f"  平均精度: {avg_acc:.4f} (±{np.std(cv_scores):.4f})")
            print(f"  平均適合率: {avg_prec:.4f}")
            print(f"  平均再現率: {avg_rec:.4f}")
            print(f"  F1スコア: {2*avg_prec*avg_rec/(avg_prec+avg_rec+1e-10):.4f}")
            
            results[name] = {
                'model': model,
                'accuracy': avg_acc,
                'precision': avg_prec,
                'recall': avg_rec,
                'f1': 2*avg_prec*avg_rec/(avg_prec+avg_rec+1e-10)
            }
        
        # 最良モデルの選択（F1スコアで評価）
        best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
        print(f"\n{'='*80}")
        print(f"最良モデル: {best_model_name}")
        print(f"F1スコア: {results[best_model_name]['f1']:.4f}")
        print(f"{'='*80}")
        
        # アンサンブルモデルの作成
        print(f"\n{'='*60}")
        print("アンサンブルモデルを作成中...")
        print(f"{'='*60}")
        
        ensemble = VotingClassifier(
            estimators=[
                ('lr', models['ロジスティック回帰']),
                ('rf', models['ランダムフォレスト']),
                ('gb', models['勾配ブースティング'])
            ],
            voting='soft'
        )
        
        # 全データで最終学習
        ensemble.fit(X_scaled, y)
        
        self.stream_classifier = ensemble
        self.scaler = scaler
        
        # 特徴量の重要度（ランダムフォレストから取得）
        rf_model = models['ランダムフォレスト']
        rf_model.fit(X, y)
        
        print(f"\n特徴量の重要度 (Top 15):")
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. {self.feature_names[idx]:30s} : {importances[idx]:.4f}")
        
        return results
    
    def train_time_regressor(self, X, y, stream_hour):
        """配信時間予測モデルの学習"""
        print("\n" + "=" * 80)
        print("配信時間予測モデルの学習")
        print("=" * 80)
        
        # 配信があった日のデータのみ使用
        stream_mask = (y == 1) & (stream_hour.notna())
        X_stream = X[stream_mask]
        y_hour = stream_hour[stream_mask]
        
        if len(X_stream) < 10:
            print("配信時間のデータが不足しています。")
            return None
        
        print(f"学習データ数: {len(X_stream)}")
        print(f"配信時間の分布:")
        print(f"  平均: {y_hour.mean():.1f}時")
        print(f"  中央値: {y_hour.median():.1f}時")
        print(f"  標準偏差: {y_hour.std():.1f}時間")
        
        # ランダムフォレスト回帰
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # 時系列分割で評価
        tscv = TimeSeriesSplit(n_splits=3)
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X_stream):
            X_train, X_val = X_stream.iloc[train_idx], X_stream.iloc[val_idx]
            y_train, y_val = y_hour.iloc[train_idx], y_hour.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)
        
        print(f"\n時系列CV結果:")
        print(f"  平均絶対誤差(MAE): {np.mean(mae_scores):.2f}時間")
        
        # 全データで学習
        model.fit(X_stream, y_hour)
        self.time_regressor = model
        
        return model
    
    def train_viewers_regressor(self, X, y, concurrent_viewers):
        """視聴者数予測モデルの学習"""
        print("\n" + "=" * 80)
        print("視聴者数予測モデルの学習")
        print("=" * 80)
        
        # 配信があった日のデータのみ使用
        stream_mask = (y == 1) & (concurrent_viewers.notna())
        X_stream = X[stream_mask]
        y_viewers = concurrent_viewers[stream_mask]
        
        if len(X_stream) < 10:
            print("視聴者数のデータが不足しています。")
            return None
        
        print(f"学習データ数: {len(X_stream)}")
        print(f"視聴者数の分布:")
        print(f"  平均: {y_viewers.mean():.0f}人")
        print(f"  中央値: {y_viewers.median():.0f}人")
        print(f"  標準偏差: {y_viewers.std():.0f}人")
        
        # ログ変換（視聴者数は対数正規分布に従うことが多い）
        y_viewers_log = np.log1p(y_viewers)
        
        # ランダムフォレスト回帰
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # 時系列分割で評価
        tscv = TimeSeriesSplit(n_splits=3)
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X_stream):
            X_train, X_val = X_stream.iloc[train_idx], X_stream.iloc[val_idx]
            y_train, y_val = y_viewers_log.iloc[train_idx], y_viewers.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)  # 逆変換
            
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)
        
        print(f"\n時系列CV結果:")
        print(f"  平均絶対誤差(MAE): {np.mean(mae_scores):.0f}人")
        
        # 全データで学習
        model.fit(X_stream, y_viewers_log)
        self.viewers_regressor = model
        
        return model
    
    def save_models(self):
        """モデルの保存"""
        print(f"\n{'='*80}")
        print("モデルを保存中...")
        print(f"{'='*80}")
        
        model_data = {
            'stream_classifier': self.stream_classifier,
            'time_regressor': self.time_regressor,
            'viewers_regressor': self.viewers_regressor,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        model_path = MODEL_DIR / "advanced_models.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ モデル保存完了: {model_path}")
    
    def load_models(self):
        """モデルの読み込み"""
        model_path = MODEL_DIR / "advanced_models.pkl"
        
        if not model_path.exists():
            return False
        
        print(f"{'='*80}")
        print("保存されたモデルを読み込み中...")
        print(f"{'='*80}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.stream_classifier = model_data['stream_classifier']
        self.time_regressor = model_data['time_regressor']
        self.viewers_regressor = model_data['viewers_regressor']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        print(f"✓ モデル読み込み完了")
        return True
    
    def predict_future(self, features_df, days_ahead=7):
        """将来の配信を予測"""
        print(f"\n{'='*80}")
        print(f"今後{days_ahead}日間の予測")
        print(f"{'='*80}")
        
        import pytz
        jst = pytz.timezone('Asia/Tokyo')
        today = datetime.now(jst).date()
        
        predictions = []
        
        # 最新のデータを基に特徴量を作成
        last_data = features_df.iloc[-1].copy()
        
        for i in range(days_ahead):
            future_date = today + timedelta(days=i)
            
            # 特徴量を作成
            features = self.create_features_for_date(future_date, features_df)
            
            # 予測
            X = pd.DataFrame([features])[self.feature_names]
            X_scaled = self.scaler.transform(X)
            
            # 配信確率
            prob = self.stream_classifier.predict_proba(X_scaled)[0, 1]
            pred = 1 if prob >= 0.5 else 0
            
            # 配信時間の予測
            if self.time_regressor and pred == 1:
                hour_pred = self.time_regressor.predict(X)[0]
            else:
                hour_pred = None
            
            # 視聴者数の予測
            if self.viewers_regressor and pred == 1:
                viewers_pred_log = self.viewers_regressor.predict(X)[0]
                viewers_pred = np.expm1(viewers_pred_log)
            else:
                viewers_pred = None
            
            predictions.append({
                'date': future_date,
                'probability': prob,
                'prediction': pred,
                'hour': hour_pred,
                'viewers': viewers_pred
            })
        
        # 結果表示
        weekday_names = ['月', '火', '水', '木', '金', '土', '日']
        
        print(f"\n{'日付':<14} {'曜日':<6} {'確率':<8} {'予測':<8} {'時間':<8} {'視聴者数':<12}")
        print("-" * 80)
        
        for pred in predictions:
            date_str = pred['date'].strftime('%Y/%m/%d')
            weekday = weekday_names[pred['date'].weekday()]
            prob_str = f"{pred['probability']*100:.1f}%"
            pred_str = "✓ あり" if pred['prediction'] == 1 else "✗ なし"
            
            hour_str = f"{pred['hour']:.0f}時頃" if pred['hour'] else "-"
            viewers_str = f"{pred['viewers']:.0f}人" if pred['viewers'] else "-"
            
            print(f"{date_str:<14} {weekday:>4}  {prob_str:>6}  {pred_str:>8}  {hour_str:>8}  {viewers_str:>10}")
        
        return predictions
    
    def create_features_for_date(self, target_date, historical_df):
        """指定日の特徴量を作成"""
        features = {}
        
        # 基本的な時間特徴量
        features['month'] = target_date.month
        features['day'] = target_date.day
        features['day_of_week'] = target_date.weekday()
        features['day_of_year'] = target_date.timetuple().tm_yday
        features['week_of_year'] = target_date.isocalendar()[1]
        features['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
        features['is_month_start'] = 1 if target_date.day <= 7 else 0
        features['is_month_end'] = 1 if target_date.day >= 24 else 0
        
        # 周期的特徴量
        features['day_of_week_sin'] = np.sin(2 * np.pi * target_date.weekday() / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * target_date.weekday() / 7)
        features['month_sin'] = np.sin(2 * np.pi * target_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * target_date.month / 12)
        features['day_of_year_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['day_of_year_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
        
        # 過去のデータから計算
        # 移動平均
        for window in [3, 7, 14, 30]:
            recent = historical_df['has_stream'].tail(window).mean()
            features[f'stream_ma_{window}d'] = recent
        
        # ラグ特徴量
        for lag in [1, 2, 3, 7, 14]:
            if len(historical_df) >= lag:
                features[f'stream_lag_{lag}d'] = historical_df['has_stream'].iloc[-lag]
            else:
                features[f'stream_lag_{lag}d'] = 0
        
        # 連続日数
        features['consecutive_stream_days'] = 0
        features['consecutive_no_stream_days'] = 0
        features['days_since_last_stream'] = 0
        
        # 最近のデータから計算
        if len(historical_df) > 0:
            for i in range(min(30, len(historical_df))):
                idx = -(i+1)
                if historical_df['has_stream'].iloc[idx] == 1:
                    features['days_since_last_stream'] = i
                    break
        
        # 統計的特徴量
        if len(historical_df) > 0:
            weekday_rate = historical_df.groupby('day_of_week')['has_stream'].mean()
            features['day_of_week_stream_rate'] = weekday_rate.get(target_date.weekday(), 0.5)
            
            month_rate = historical_df.groupby('month')['has_stream'].mean()
            features['month_stream_rate'] = month_rate.get(target_date.month, 0.5)
        else:
            features['day_of_week_stream_rate'] = 0.5
            features['month_stream_rate'] = 0.5
        
        # 複合特徴量
        features['weekend_month_end'] = features['is_weekend'] * features['is_month_end']
        
        # トレンド
        features['time_index'] = len(historical_df)
        
        return features


def main():
    """メイン処理"""
    print("\n" + "=" * 80)
    print("改善版YouTube配信予測システム")
    print("=" * 80)
    
    system = AdvancedStreamPredictionSystem()
    
    # データの読み込みと分析
    filepath = "youtube_UCHVXbQzkl3rDfsXWo8xi2qw_20260206_005751.csv"
    df, stream_df = system.load_and_analyze_data(filepath)
    
    # 時系列データセットの作成
    daily_df = system.create_timeseries_dataset(df)
    
    # 特徴量エンジニアリング
    features_df = system.create_advanced_features(daily_df)
    
    # データ準備
    X, y, stream_hour, concurrent_viewers, dates = system.prepare_ml_data(features_df)
    
    # モデルの学習
    print("\n【配信有無の分類モデル】")
    results = system.train_stream_classifier(X, y, dates)
    
    print("\n【配信時間の予測モデル】")
    system.train_time_regressor(X, y, stream_hour)
    
    print("\n【視聴者数の予測モデル】")
    system.train_viewers_regressor(X, y, concurrent_viewers)
    
    # モデルの保存
    system.save_models()
    
    # 将来の予測
    system.predict_future(features_df, days_ahead=7)
    
    print(f"\n{'='*80}")
    print("処理が完了しました！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()