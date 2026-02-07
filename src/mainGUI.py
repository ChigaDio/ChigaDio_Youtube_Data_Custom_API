import threading
import webbrowser
import requests
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
from io import BytesIO

from youtubedataapi import YoutubeContentType, YoutubeDataFind, get_youtube_data

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class YouTubeProTool(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("YouTube Channel Analytics Dashboard")
        self.geometry("1400x900")

        self.master_df = pd.DataFrame()
        
        # --- 全体レイアウト ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1) # メインエリア
        self.grid_rowconfigure(2, weight=0) # ステータスバー

        self.setup_header()
        self.setup_sidebar()
        self.setup_main_area()
        self.setup_status_bar()

    def setup_header(self):
        self.header_frame = ctk.CTkFrame(self, height=100, corner_radius=0)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.header_frame.grid_columnconfigure(2, weight=1)

        # 入力エリア
        input_sub = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        input_sub.grid(row=0, column=0, padx=20, pady=10)

        self.ent_api = ctk.CTkEntry(input_sub, show="*", width=200, placeholder_text="API KEY (※)")
        self.ent_api.grid(row=0, column=0, padx=5)
        
        self.ent_channel = ctk.CTkEntry(input_sub, width=200, placeholder_text="CHANNEL ID")
        self.ent_channel.grid(row=0, column=1, padx=5)

        self.btn_fetch = ctk.CTkButton(self.header_frame, text="データ取得開始", width=120, command=self.start_fetching)
        self.btn_fetch.grid(row=0, column=1, padx=10)

        # チャンネル情報
        self.info_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.info_frame.grid(row=0, column=2, padx=20, sticky="e")
        
        self.icon_label = ctk.CTkLabel(self.info_frame, text="", width=50, height=50)
        self.icon_label.pack(side="left", padx=10)
        self.name_label = ctk.CTkLabel(self.info_frame, text="---", font=ctk.CTkFont(size=16, weight="bold"))
        self.name_label.pack(side="left")

    def setup_sidebar(self):
        self.sidebar = ctk.CTkScrollableFrame(self, width=250, label_text="詳細フィルター")
        self.sidebar.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # フィルター項目
        self.add_filter_label("タイトル検索")
        self.f_title = ctk.CTkEntry(self.sidebar)
        self.f_title.pack(fill="x", padx=10, pady=5)

        self.add_filter_label("コンテンツ種別")
        self.f_type = ctk.CTkOptionMenu(self.sidebar, values=["すべて", "ライブ配信", "動画", "ショート"])
        self.f_type.pack(fill="x", padx=10, pady=5)

        self.add_filter_label("最低視聴回数")
        self.f_view_min = ctk.CTkEntry(self.sidebar, placeholder_text="0")
        self.f_view_min.pack(fill="x", padx=10, pady=5)

        self.btn_apply = ctk.CTkButton(self.sidebar, text="フィルター適用", command=self.apply_filters)
        self.btn_apply.pack(pady=20, padx=10, fill="x")

        self.btn_save = ctk.CTkButton(self.sidebar, text="CSV保存", fg_color="#28a745", command=self.save_csv)
        self.btn_save.pack(pady=5, padx=10, fill="x")

    def add_filter_label(self, text):
        ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=11, weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))

    def setup_main_area(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.tab_data = self.tabview.add("データグリッド")
        self.tab_stats = self.tabview.add("詳細統計ダッシュボード")

        # --- DataGrid Tab ---
        self.setup_datagrid()
        
        # --- Statistics Tab (Dashboard) ---
        self.stats_container = ctk.CTkScrollableFrame(self.tab_stats)
        self.stats_container.pack(fill="both", expand=True)
        self.setup_stats_ui()

    def setup_datagrid(self):
        # 列の定義
        self.cols = ("title", "view_count", "like_count", "comment_count", "live_status", "published_at", "duration_seconds", "url")
        self.tree_frame = ctk.CTkFrame(self.tab_data)
        self.tree_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(self.tree_frame, columns=self.cols, show="headings")
        for col in self.cols:
            self.tree.heading(col, text=col.upper(), command=lambda c=col: self.sort_treeview(c, False))
            self.tree.column(col, width=120, anchor="center")
        self.tree.column("title", width=350, anchor="w")

        y_scroll = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        x_scroll = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscroll=y_scroll.set, xscrollcommand=x_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)

        # 右クリックメニュー
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="ブラウザで開く", command=self.open_url)
        self.tree.bind("<Button-3>", lambda e: self.menu.post(e.x_root, e.y_root) if self.tree.identify_row(e.y) else None)

    def setup_stats_ui(self):
        # 1. サマリーカード (最上部)
        self.card_frame = ctk.CTkFrame(self.stats_container, fg_color="transparent")
        self.card_frame.pack(fill="x", padx=10, pady=10)
        self.cards = {}
        titles = ["総コンテンツ", "通常動画", "ライブ配信", "ショート"]
        for i, title in enumerate(titles):
            f = ctk.CTkFrame(self.card_frame, corner_radius=10)
            f.grid(row=0, column=i, padx=10, sticky="nsew")
            ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(10, 0))
            val = ctk.CTkLabel(f, text="0", font=ctk.CTkFont(size=28, weight="bold"), text_color="#3b8ed0")
            val.pack(pady=10)
            self.cards[title] = val
        self.card_frame.grid_columnconfigure((0,1,2,3), weight=1)

        # 2. 詳細統計エリア (2列グリッド)
        self.detail_frame = ctk.CTkFrame(self.stats_container, fg_color="transparent")
        self.detail_frame.pack(fill="both", expand=True, padx=10)
        self.detail_frame.grid_columnconfigure((0,1), weight=1)

        # 各統計ボックスのコンテナを作成
        self.box_yearly = self.create_stat_box(self.detail_frame, "年別内訳", 0, 0)
        self.box_monthly = self.create_stat_box(self.detail_frame, "直近12ヶ月の内訳", 0, 1)
        self.box_hourly = self.create_stat_box(self.detail_frame, "投稿時間帯分布", 1, 0)
        self.box_dayofweek = self.create_stat_box(self.detail_frame, "曜日別分布", 1, 1)
        
        # 3. 配信詳細情報 (最下部)
        self.live_info_box = ctk.CTkFrame(self.stats_container)
        self.live_info_box.pack(fill="x", padx=20, pady=10)
        self.live_info_label = ctk.CTkLabel(self.live_info_box, text="配信データ詳細: 待機中...", font=ctk.CTkFont(size=14))
        self.live_info_label.pack(pady=15)

    def create_stat_box(self, parent, title, r, c):
        """統計ボックスの枠とスクロール可能なリスト領域を作成"""
        container = ctk.CTkFrame(parent)
        container.grid(row=r, column=c, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(container, text=title, font=ctk.CTkFont(size=15, weight="bold"), text_color="#aaaaaa").pack(pady=5)
        
        # ヘッダー (項目名 | 動画 | 配信 | 短)
        header = ctk.CTkFrame(container, fg_color="#333333", height=25)
        header.pack(fill="x", padx=5)
        ctk.CTkLabel(header, text="区分", width=60, font=ctk.CTkFont(size=11)).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="動", width=40, font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="配", width=40, font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        ctk.CTkLabel(header, text="短", width=40, font=ctk.CTkFont(size=11)).pack(side="left", padx=5)

        # データ表示用フレーム
        content_frame = ctk.CTkFrame(container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        return content_frame  # ここに後でラベルを流し込む

    def setup_status_bar(self):
        self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        self.prog_bar = ctk.CTkProgressBar(self.status_frame, width=200)
        self.prog_bar.pack(side="left", padx=20, pady=5)
        self.prog_bar.set(0)
        
        self.status_lbl = ctk.CTkLabel(self.status_frame, text="待機中")
        self.status_lbl.pack(side="left", padx=10)

    # --- ロジック ---

    def start_fetching(self):
        self.btn_fetch.configure(state="disabled")
        threading.Thread(target=self.fetch_process, daemon=True).start()

    def fetch_process(self):
        api, cid = self.ent_api.get().strip(), self.ent_channel.get().strip()
        if not api or not cid:
            self.update_status("エラー: キーとIDを入力してください", 0)
            self.btn_fetch.configure(state="normal")
            return

        try:
            self.master_df = pd.DataFrame()
            self.update_status("チャンネル情報取得中...", 0.2)
            self.get_channel_meta(api, cid)
            
            self.update_status("動画リスト取得中...", 0.4)
            find = YoutubeDataFind(Api=api, ChannelId=cid, MaxResults=0)
            results = get_youtube_data(find)

            self.update_status("データ変換中...", 0.7)
            data = []
            for v in results:
                # ライブ判定の修正: live_statusが'none'以外、または実際の開始時間がある場合
                is_live = v.live_status in ['live', 'completed', 'upcoming']
                
                if "コトブキヤ" in v.title:
                    print(v.title, v.published_at, v.scheduled_start_time, v.actual_start_time)
                
                row = {
                    "title": v.title,
                    "content_category": v.content_category,
                    "published_at": v.published_at.isoformat() if v.published_at else "",
                    "view_count": v.view_count,
                    "like_count": v.like_count,
                    "comment_count": v.comment_count,
                    "is_live": is_live,
                    "is_live_now": v.is_live_now,
                    "live_status": v.live_status,
                    "scheduled_start_time": v.scheduled_start_time.isoformat() if v.scheduled_start_time else v.published_at.isoformat(),
                    "actual_start_time": v.actual_start_time.isoformat() if v.actual_start_time else v.published_at.isoformat(),
                    "actual_end_time": v.actual_end_time.isoformat() if v.actual_end_time else v.published_at.isoformat(),
                    "concurrent_viewers": v.concurrent_viewers,
                    "duration_seconds": v.duration.total_seconds() if v.duration else None,
                    "was_broadcast_yesterday": v.was_broadcast_yesterday,
                    "weekday": v.weekday.name if v.weekday else "",
                    "is_holiday": v.is_holiday,
                    "consecutive_broadcast_days": v.consecutive_broadcast_days,
                    "same_day_broadcast_count": v.same_day_broadcast_count,
                    "days_since_last_broadcast": v.days_since_last_broadcast,
                    "url": v.url
                }
                data.append(row)

            self.master_df = pd.DataFrame(data)
            self.after(0, self.finish_fetch)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.update_status("取得失敗", 0))
        finally:
            self.after(0, lambda: self.btn_fetch.configure(state="normal"))

    def finish_fetch(self):
        self.apply_filters()
        self.update_dashboard()
        self.update_status(f"完了: {len(self.master_df)}件取得", 1.0)

    def get_channel_meta(self, api, cid):
        url = f"https://www.googleapis.com/youtube/v3/channels?part=snippet&id={cid}&key={api}"
        res = requests.get(url).json()
        if "items" in res:
            snip = res["items"][0]["snippet"]
            name = snip["title"]
            thumb = snip["thumbnails"]["default"]["url"]
            img = Image.open(BytesIO(requests.get(thumb).content))
            ctk_img = ctk.CTkImage(img, size=(50, 50))
            self.after(0, lambda: self.icon_label.configure(image=ctk_img))
            self.after(0, lambda: self.name_label.configure(text=name))

    def update_status(self, text, val):
        self.status_lbl.configure(text=text)
        self.prog_bar.set(val)

    def apply_filters(self):
        if self.master_df.empty: return
        df = self.master_df.copy()
        
        # タイトル検索
        q = self.f_title.get()
        if q: df = df[df['title'].str.contains('|'.join(q.split()), case=False)]
        
        # 種別
        t = self.f_type.get()
        if t == "ライブ配信": df = df[df['content_category'] == YoutubeContentType.LIVE]
        elif t == "動画": df = df[df['content_category'] == YoutubeContentType.NORMAL_VIDEO]
        elif t == "ショート": df = df[df['content_category'] == YoutubeContentType.SHORTS]

        # 表示更新
        for i in self.tree.get_children(): self.tree.delete(i)
        for _, r in df.iterrows():
            self.tree.insert("", "end", values=[r[c] for c in self.cols])

    def update_dashboard(self):
        df = self.master_df.copy()
        if df.empty: return
        
        # --- 基準を published_at に変更し、日本時間に変換 ---
        # published_at を datetime オブジェクトに変換 (UTCとして読み込み)
        publish_time = pd.to_datetime(df['scheduled_start_time'])

        

        # 統計用の列を生成
        df['year'] = publish_time.dt.year
        df['month_val'] = publish_time.dt.strftime('%Y-%m')
        df['hour'] = publish_time.dt.hour
        wd_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
        df['weekday_jp'] = publish_time.dt.weekday.map(wd_map)

        # --- 修正ポイント2: 判定ロジックの整理 ---
        # type_label の割り当てをより確実に
        df['type_label'] = 'live'
        df.loc[df['content_category'] == YoutubeContentType.NORMAL_VIDEO, 'type_label'] = 'video'
        df.loc[df['content_category'] == YoutubeContentType.SHORTS, 'type_label'] = 'short'

        # 1. サマリーカード更新
        self.cards["総コンテンツ"].configure(text=str(len(df)))
        self.cards["通常動画"].configure(text=str(len(df[df['type_label']=='video'])))
        self.cards["ライブ配信"].configure(text=str(len(df[df['type_label']=='live'])))
        self.cards["ショート"].configure(text=str(len(df[df['type_label']=='short'])))

        # 2. 各詳細ボックスの更新
        # NaNを除去して集計
        df_clean = df.dropna(subset=['year'])
        
        self.fill_stat_rows(self.box_yearly, df_clean, 'year')
        self.fill_stat_rows(self.box_monthly, df_clean, 'month_val', limit=12)
        self.fill_stat_rows(self.box_hourly, df_clean, 'hour')
        
        # 曜日順に並べるための処理
        wd_order = ['月', '火', '水', '木', '金', '土', '日']
        self.fill_stat_rows(self.box_dayofweek, df_clean, 'weekday_jp', sort_idx=False)

        # 3. 配信詳細
        live_df = df[df['content_category'] == YoutubeContentType.LIVE]
        if not live_df.empty:
            avg_sec = live_df['duration_seconds'].fillna(0).mean()
            max_views = live_df['view_count'].max()
            avg_text = f"【配信分析】 平均配信時間: {avg_sec/60:.1f}分 | 最大視聴数: {max_views:,}回"
            self.live_info_label.configure(text=avg_text)
        else:
            self.live_info_label.configure(text="配信データがありません")

    def fill_stat_rows(self, target_frame, df, group_col, limit=None, sort_idx=True):
        """フレーム内に統計行(区分|動|配|短)を生成して配置する"""
        # 前の表示をクリア
        for widget in target_frame.winfo_children():
            widget.destroy()

        # クロス集計 (行: group_col, 列: type_label)
        ct = pd.crosstab(df[group_col], df['type_label'])
        
        # 必要な列が欠けている場合の補完
        for col in ['video', 'live', 'short']:
            if col not in ct.columns: ct[col] = 0
        
        if sort_idx:
            ct = ct.sort_index(ascending=False)
        if limit:
            ct = ct.head(limit)

        for index, row in ct.iterrows():
            row_f = ctk.CTkFrame(target_frame, fg_color="transparent")
            row_f.pack(fill="x", pady=1)
            
            ctk.CTkLabel(row_f, text=str(index), width=70, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
            ctk.CTkLabel(row_f, text=str(row['video']), width=45, text_color="#cccccc").pack(side="left")
            ctk.CTkLabel(row_f, text=str(row['live']), width=45, text_color="#3b8ed0").pack(side="left")
            ctk.CTkLabel(row_f, text=str(row['short']), width=45, text_color="#e74c3c").pack(side="left")

    def sort_treeview(self, col, rev):
        data = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        try:
            data.sort(key=lambda x: float(str(x[0]).replace(",","")), reverse=rev)
        except:
            data.sort(reverse=rev)
        for i, (_, k) in enumerate(data): self.tree.move(k, "", i)
        self.tree.heading(col, command=lambda: self.sort_treeview(col, not rev))

    def open_url(self):
        sel = self.tree.selection()
        if sel: webbrowser.open(self.tree.item(sel[0])['values'][-1])

    def save_csv(self):
        # 規約確認
        if not messagebox.askyesno("規約確認", "YouTube APIの規約に基づき、データは30日以内に破棄する必要があります。同意して保存しますか？"):
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            self.master_df.to_csv(path, index=False, encoding="utf-8-sig")
            messagebox.showinfo("保存", "保存完了しました")

if __name__ == "__main__":
    app = YouTubeProTool()
    # スタイル微調整
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", rowheight=28)
    style.map("Treeview", background=[('selected', '#1f538d')])
    app.mainloop()