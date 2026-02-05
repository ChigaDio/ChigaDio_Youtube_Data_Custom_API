from zoneinfo import ZoneInfo
from youtubedataapi import YoutubeDataFind,YoutubeOrder, get_youtube_data
import os
import pandas as pd
from io import StringIO
def main():

    
    find = YoutubeDataFind(
    Api="YOUTUBE_API_KEY",
    ChannelId="ChANNEL_ID_HERE",
    MaxResults=0
    )

    results = get_youtube_data(find)

    for v in results[:3]:
        print(f"{v.title}")
        print(f"  URL: {v.url}")
        print(f"  公開: {v.published_at}")
        print(f"  再生: {v.view_count:,}  いいね: {v.like_count:,}  コメント: {v.comment_count:,}")
        if v.live_status == "live":
            print(f"  現在ライブ中！ 同時視聴: {v.concurrent_viewers:,}人")
        elif v.live_status == "upcoming":
            print(f"  配信予定: {v.scheduled_start_time}")
        print()
        
    # データフレームに変換
    data = []
    for v in results:
        row = {
            "title": v.title,
            "video_id": v.video_id,
            "published_at": v.published_at.isoformat() if v.published_at else "",
            "view_count": v.view_count,
            "like_count": v.like_count,
            "comment_count": v.comment_count,
            "is_live_now": v.is_live_now,
            "live_status": v.live_status,
            "scheduled_start_time": v.scheduled_start_time.isoformat() if v.scheduled_start_time else "",
            "actual_start_time": v.actual_start_time.isoformat() if v.actual_start_time else "",
            "actual_end_time": v.actual_end_time.isoformat() if v.actual_end_time else "",
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

    df = pd.DataFrame(data)
    
    # 保存先ファイル名（例: チャンネルID + 現在時刻）
    from datetime import datetime
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    filename = f"youtube_{find.ChannelId}_{timestamp}.csv"

    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"CSV保存完了: {filename} ({len(results)}行)")

if __name__ == "__main__":
    main()
