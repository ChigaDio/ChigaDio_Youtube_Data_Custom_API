from dataclasses import dataclass
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from enum import Enum
from datetime import datetime, date, timedelta
from typing import List, Optional, Set
from zoneinfo import ZoneInfo
from collections import defaultdict
import requests
import pandas as pd
import io

# ── グローバルキャッシュ（モジュール読み込み時に1回だけ取得） ──
HOLIDAYS_CACHE: Set[date] = set()
_HOLIDAYS_LOADED = False


def load_japanese_holidays_once():
    """内閣府公式CSVを1回だけ取得してキャッシュする"""
    global HOLIDAYS_CACHE, _HOLIDAYS_LOADED

    if _HOLIDAYS_LOADED:
        return

    url = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        # SJISでエンコードされていることが多いので指定
        df = pd.read_csv(io.StringIO(resp.text), encoding="shift-jis")

        # 列名が「国民の祝日・休日月日」「国民の祝日・休日名称」のはず
        date_col = df.columns[0]
        name_col = df.columns[1]

        for _, row in df.iterrows():
            date_str = row[date_col].strip()
            try:
                # 形式: YYYY/M/D または YYYY/MM/DD
                ymd = [int(x) for x in date_str.split("/")]
                if len(ymd) == 3:
                    d = date(ymd[0], ymd[1], ymd[2])
                    HOLIDAYS_CACHE.add(d)
            except:
                pass  # 不正な行は無視

        print(f"内閣府祝日データをロード完了: {len(HOLIDAYS_CACHE)}件 (1955年〜最新年)")
        _HOLIDAYS_LOADED = True

    except Exception as e:
        print(f"祝日CSV取得エラー: {e} → 祝日判定は無効になります")
        _HOLIDAYS_LOADED = True  # 二度と試さない


# モジュール読み込み時に自動でロード（または関数内で呼ぶ）
load_japanese_holidays_once()


class YoutubeOrder(Enum):
    DATE = "date"
    RATING = "rating"
    RELEVANCE = "relevance"
    TITLE = "title"
    VIDEO_COUNT = "videoCount"
    VIEW_COUNT = "viewCount"


class Weekday(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class YoutubeVideoDetail:
    """YouTube動画1本分の詳細情報をまとめたデータクラス（ライブ情報・配信傾向分析用）"""

    # 基本情報
    title: str                          # 動画タイトル
    video_id: str                       # YouTube動画ID（11桁の英数字）
    published_at: Optional[datetime] = None  
    # 動画が公開（アップロード）された日時（JST変換済み）

    # 統計情報（通常動画・ライブ終了後）
    view_count: Optional[int] = None        # 総再生回数
    like_count: Optional[int] = None        # いいね数（公開設定の場合のみ）
    comment_count: Optional[int] = None     # コメント総数（公開設定の場合のみ）

    # ライブ配信の状態・時刻情報
    is_live_now: bool = False               # 現在この動画がライブ配信中かどうか（リアルタイム判定）
    live_status: str = "none"               # ライブの状態："live"（配信中） / "upcoming"（予定） / "none"（通常動画または終了）
    scheduled_start_time: Optional[datetime] = None  
    # 配信予定開始時刻（JST変換済み）→ upcoming の場合に存在
    actual_start_time: Optional[datetime] = None    
    # 実際に配信が開始した時刻（JST変換済み）→ 開始後に出現
    actual_end_time: Optional[datetime] = None      
    # 実際に配信が終了した時刻（JST変換済み）→ 終了後に出現
    concurrent_viewers: Optional[int] = None        
    # 現在同時接続視聴者数（配信中の場合のみリアルタイムで更新）

    # 配信時間（計算値）
    duration: Optional[timedelta] = None            
    # 実際の配信時間（actual_end_time - actual_start_time）→ 終了後に計算

    # 配信傾向分析用の追加フィールド
    was_broadcast_yesterday: bool = False           
    # 前日（この動画の公開日の前日）に配信があったかどうか（連日判定用）

    weekday: Optional[Weekday] = None               
    # 公開日（published_at）の曜日（Enumで管理）

    is_holiday: bool = False                        
    # 公開日が日本の祝日・休日かどうか（内閣府データに基づく）

    consecutive_broadcast_days: int = 1             
    # 連日配信カウント（この日を含む連続配信日数。最新日から遡って計算）

    same_day_broadcast_count: int = 1               
    # 同日内の配信順番（同じ日に複数配信した場合、何本目か。時間昇順で1スタート）

    days_since_last_broadcast: int = 0              
    # 前回の配信（直近の古い日）からの空き日数（連日の場合は0、1日空くと1、など）

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass
class YoutubeDataFind:
    Api: str = ""
    ChannelId: str = ""
    Order: YoutubeOrder = YoutubeOrder.DATE
    MaxResults: int = 200


def get_youtube_data(findData: YoutubeDataFind) -> List[YoutubeVideoDetail]:
    if not findData.Api:
        print("APIキーが設定されていません。")
        return []
    if not findData.ChannelId:
        print("チャンネルIDが設定されていません。")
        return []

    youtube = build('youtube', 'v3', developerKey=findData.Api)
    jst = ZoneInfo("Asia/Tokyo")

    try:
        # 1. アップロードプレイリストID
        channel_resp = youtube.channels().list(
            part="contentDetails",
            id=findData.ChannelId
        ).execute()

        if not channel_resp.get("items"):
            print("チャンネルが見つかりません")
            return []

        uploads_id = channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # 2. 動画ID収集
        video_ids: List[str] = []
        next_page_token: Optional[str] = None
        fetched_count = 0

        while True:
            if findData.MaxResults > 0 and fetched_count >= findData.MaxResults:
                break

            pl_resp = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            for item in pl_resp.get("items", []):
                if findData.MaxResults > 0 and fetched_count >= findData.MaxResults:
                    break
                video_ids.append(item["contentDetails"]["videoId"])
                fetched_count += 1

            next_page_token = pl_resp.get("nextPageToken")
            if not next_page_token:
                break

        if not video_ids:
            return []

        # 3. 詳細バッチ取得
        videos: List[YoutubeVideoDetail] = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            vid_resp = youtube.videos().list(
                part="snippet,statistics,liveStreamingDetails",
                id=",".join(batch)
            ).execute()

            for item in vid_resp.get("items", []):
                snip = item["snippet"]
                stats = item.get("statistics", {})
                live = item.get("liveStreamingDetails", {})

                published_at = None
                if pub_str := snip.get("publishedAt"):
                    dt_utc = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    published_at = dt_utc.astimezone(jst)

                sched_start = act_start = act_end = None
                if "scheduledStartTime" in live:
                    sched_start = datetime.fromisoformat(live["scheduledStartTime"].replace("Z", "+00:00")).astimezone(jst)
                if "actualStartTime" in live:
                    act_start = datetime.fromisoformat(live["actualStartTime"].replace("Z", "+00:00")).astimezone(jst)
                if "actualEndTime" in live:
                    act_end = datetime.fromisoformat(live["actualEndTime"].replace("Z", "+00:00")).astimezone(jst)

                detail = YoutubeVideoDetail(
                    title=snip["title"],
                    video_id=item["id"],
                    published_at=published_at,
                    view_count=int(stats.get("viewCount", 0)) if stats.get("viewCount") else None,
                    like_count=int(stats.get("likeCount", 0)) if stats.get("likeCount") else None,
                    comment_count=int(stats.get("commentCount", 0)) if stats.get("commentCount") else None,
                    is_live_now=bool(live.get("concurrentViewers")),
                    live_status=snip.get("liveBroadcastContent", "none"),
                    scheduled_start_time=sched_start,
                    actual_start_time=act_start,
                    actual_end_time=act_end,
                    concurrent_viewers=int(live["concurrentViewers"]) if live.get("concurrentViewers") else None
                )

                if detail.actual_start_time and detail.actual_end_time:
                    detail.duration = detail.actual_end_time - detail.actual_start_time

                videos.append(detail)

        videos = [v for v in videos if v.published_at]
        videos.sort(key=lambda v: v.published_at, reverse=True)

        if not videos:
            return []

        # 4. 祝日判定（キャッシュ使用 → 超軽量）
        for v in videos:
            if v.published_at:
                day = v.published_at.date()
                v.is_holiday = day in HOLIDAYS_CACHE

        # 5. 日付グループ & 追加計算（以前と同じ）
        daily_groups: defaultdict[date, List[YoutubeVideoDetail]] = defaultdict(list)
        for v in videos:
            day = v.published_at.date()  # type: ignore
            daily_groups[day].append(v)

        sorted_days = sorted(daily_groups.keys(), reverse=True)

        consecutive_counts: dict[date, int] = {}
        prev_day: Optional[date] = None
        streak = 0
        for day in sorted_days:
            if prev_day and (prev_day - day) == timedelta(days=1):
                streak += 1
            else:
                streak = 1
            consecutive_counts[day] = streak
            prev_day = day

        for day, group in daily_groups.items():
            group.sort(key=lambda v: v.published_at)
            for idx, v in enumerate(group, 1):
                v.same_day_broadcast_count = idx
                v.weekday = Weekday(v.published_at.weekday())  # type: ignore
                v.consecutive_broadcast_days = consecutive_counts[day]
                yesterday = day - timedelta(days=1)
                v.was_broadcast_yesterday = yesterday in daily_groups

                cur_idx = sorted_days.index(day)
                if cur_idx + 1 < len(sorted_days):
                    prev = sorted_days[cur_idx + 1]
                    v.days_since_last_broadcast = (day - prev).days - 1
                else:
                    v.days_since_last_broadcast = 0

        print(f"取得完了: {len(videos)} 本（祝日キャッシュ使用）")
        return videos

    except HttpError as e:
        print(f"YouTube APIエラー: {e}")
        return []
    except Exception as e:
        print(f"エラー: {e}")
        return []