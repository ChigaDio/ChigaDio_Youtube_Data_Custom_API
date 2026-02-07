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
import isodate

# ── グローバルキャッシュ（変更なし） ──
HOLIDAYS_CACHE: Set[date] = set()
_HOLIDAYS_LOADED = False

def load_japanese_holidays_once():
    global HOLIDAYS_CACHE, _HOLIDAYS_LOADED
    if _HOLIDAYS_LOADED:
        return
    url = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="shift-jis")
        date_col, name_col = df.columns[0], df.columns[1]
        for _, row in df.iterrows():
            date_str = row[date_col].strip()
            try:
                ymd = [int(x) for x in date_str.split("/")]
                if len(ymd) == 3:
                    HOLIDAYS_CACHE.add(date(ymd[0], ymd[1], ymd[2]))
            except:
                pass
        print(f"内閣府祝日データをロード完了: {len(HOLIDAYS_CACHE)}件")
        _HOLIDAYS_LOADED = True
    except Exception as e:
        print(f"祝日CSV取得エラー: {e}")
        _HOLIDAYS_LOADED = True

load_japanese_holidays_once()


class YoutubeOrder(Enum):
    DATE = "date"
    RATING = "rating"
    RELEVANCE = "relevance"
    TITLE = "title"
    VIDEO_COUNT = "videoCount"
    VIEW_COUNT = "viewCount"


class YoutubeContentType(Enum):
    """動画がどのカテゴリ（プレイリスト）から取得されたか"""
    NORMAL_VIDEO = "normal_video"   # 通常のアップロード（UC...）
    SHORTS       = "shorts"         # Shorts専用（UUSH...）
    LIVE         = "live"           # ライブ専用（UULV...）
    UNKNOWN      = "unknown"


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
    title: str
    video_id: str
    published_at: Optional[datetime] = None

    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None

    is_live_now: bool = False
    live_status: str = "none"
    scheduled_start_time: Optional[datetime] = None
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    concurrent_viewers: Optional[int] = None

    duration: Optional[timedelta] = None
    duration_sec: Optional[float] = None

    # 追加：どのプレイリスト由来かをEnumで管理
    content_category: YoutubeContentType = YoutubeContentType.UNKNOWN

    # 傾向分析フィールド（元のまま）
    was_broadcast_yesterday: bool = False
    weekday: Optional[Weekday] = None
    is_holiday: bool = False
    consecutive_broadcast_days: int = 1
    same_day_broadcast_count: int = 1
    days_since_last_broadcast: int = 0

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass
class YoutubeDataFind:
    Api: str = ""
    ChannelId: str = ""
    ShortID: str = ""
    VideoID: str = ""
    LiveID: str = ""
    Order: YoutubeOrder = YoutubeOrder.DATE
    MaxResults: int = 200


def get_youtube_data(findData: YoutubeDataFind) -> List[YoutubeVideoDetail]:
    if not findData.Api:
        print("APIキーが設定されていません。")
        return []
    if not findData.ChannelId:
        print("チャンネルIDが設定されていません。")
        return []

    # プレイリストID生成（元のロジックそのまま）
    findData.ShortID = findData.ChannelId.replace("UC", "UUSH", 1)
    findData.VideoID = findData.ChannelId.replace("UC", "UULF", 1)
    findData.LiveID  = findData.ChannelId.replace("UC", "UULV", 1)

    youtube = build('youtube', 'v3', developerKey=findData.Api)
    jst = ZoneInfo("Asia/Tokyo")

    try:
        # 1. チャンネル情報取得 → 通常アップロードplaylist
        channel_resp = youtube.channels().list(
            part="contentDetails",
            id=findData.ChannelId
        ).execute()

        if not channel_resp.get("items"):
            print("チャンネルが見つかりません")
            return []

        uploads_normal = channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # 対象プレイリスト（全部取得する方針）
        target_playlists = {
            YoutubeContentType.NORMAL_VIDEO: uploads_normal,
            YoutubeContentType.SHORTS:       findData.ShortID,
            YoutubeContentType.LIVE:         findData.LiveID,
        }

        # 2. 全プレイリストから動画ID収集
        video_ids: List[str] = []
        video_to_category: dict[str, YoutubeContentType] = {}  # videoId → カテゴリ

        fetched_count = 0
        next_page_token: Optional[str] = None

        for category, playlist_id in target_playlists.items():
            next_page_token = None
            while True:
                try:
                    if findData.MaxResults > 0 and fetched_count >= findData.MaxResults:
                        break

                    responce = youtube.playlistItems().list(
                        part="snippet,contentDetails",
                        playlistId=playlist_id,
                        maxResults=50,
                        pageToken=next_page_token
                    )

                    if not responce:
                        print(f"プレイリストの取得に失敗しました: {playlist_id}")
                        break
                    pl_resp = responce.execute()

                    for item in pl_resp.get("items", []):
                        if findData.MaxResults > 0 and fetched_count >= findData.MaxResults:
                            break
                        vid = item["contentDetails"]["videoId"]
                        video_ids.append(vid)
                        video_to_category[vid] = category   # ← ここで紐付け
                        fetched_count += 1

                    next_page_token = pl_resp.get("nextPageToken")
                    if not next_page_token:
                        break
                except HttpError as e:
                    print(f"プレイリスト取得中にエラー発生: {e}")
                    break

        if not video_ids:
            return []

        # 重複除去（稀に同じ動画が複数プレイリストに入る可能性を考慮）
        video_ids = list(dict.fromkeys(video_ids))

        # 3. 詳細バッチ取得（ほぼ元のロジックそのまま）
        videos: List[YoutubeVideoDetail] = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            vid_resp = youtube.videos().list(
                part="snippet,statistics,liveStreamingDetails,contentDetails",
                id=",".join(batch)
            ).execute()

            for item in vid_resp.get("items", []):
                snip = item["snippet"]
                stats = item.get("statistics", {})
                live = item.get("liveStreamingDetails", {})
                content = item.get("contentDetails", {})

                published_at = None
                if pub_str := snip.get("publishedAt"):
                    dt_utc = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    published_at = dt_utc.astimezone(jst)

                duration_sec = 0
                if "duration" in content:
                    try:
                        duration_sec = isodate.parse_duration(content["duration"]).total_seconds()
                    except:
                        pass

                sched_start = act_start = act_end = None
                if "scheduledStartTime" in live:
                    sched_start = datetime.fromisoformat(live["scheduledStartTime"].replace("Z", "+00:00")).astimezone(jst)
                if "actualStartTime" in live:
                    act_start = datetime.fromisoformat(live["actualStartTime"].replace("Z", "+00:00")).astimezone(jst)
                if not act_start:
                    act_start = published_at
                if not sched_start:
                    sched_start = act_start
                if not published_at:
                    published_at = act_start
                if sched_start and act_start and sched_start > act_start:
                    sched_start = sched_start.replace(year=act_start.year)
                if "actualEndTime" in live:
                    act_end = datetime.fromisoformat(live["actualEndTime"].replace("Z", "+00:00")).astimezone(jst)
                    published_at = act_end

                

                # カテゴリをプレイリスト由来で決定
                category = video_to_category.get(item["id"], YoutubeContentType.UNKNOWN)

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
                    concurrent_viewers=int(live["concurrentViewers"]) if live.get("concurrentViewers") else None,
                    duration_sec=duration_sec,
                    content_category=category,   # ← ここでEnumを設定
                )

                if detail.actual_start_time and detail.actual_end_time:
                    detail.duration = detail.actual_end_time - detail.actual_start_time

                videos.append(detail)

        videos = [v for v in videos if v.published_at]
        videos.sort(key=lambda v: v.published_at, reverse=True)

        if not videos:
            return []

        # 4. 祝日判定（変更なし）
        for v in videos:
            if v.published_at:
                v.is_holiday = v.published_at.date() in HOLIDAYS_CACHE

        # 5. 日付グループ & 追加計算（元のロジックそのまま）
        daily_groups: defaultdict[date, List[YoutubeVideoDetail]] = defaultdict(list)
        for v in videos:
            day = v.published_at.date()
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
                v.weekday = Weekday(v.published_at.weekday())
                v.consecutive_broadcast_days = consecutive_counts[day]
                yesterday = day - timedelta(days=1)
                v.was_broadcast_yesterday = yesterday in daily_groups

                cur_idx = sorted_days.index(day)
                if cur_idx + 1 < len(sorted_days):
                    prev = sorted_days[cur_idx + 1]
                    v.days_since_last_broadcast = (day - prev).days - 1
                else:
                    v.days_since_last_broadcast = 0

        print(f"取得完了: {len(videos)} 本（全プレイリスト対象 / 祝日キャッシュ使用）")
        return videos

    except HttpError as e:
        print(f"YouTube APIエラー: {e}")
        return []
    except Exception as e:
        print(f"エラー: {e}")
        return []