-- Demo queries for Rewind on Tiger Data (TimescaleDB). Run them all with:  python show_history.py
-- Each "-- name:" line starts a query. %(label)s is filled from --label (default: glasses).
-- Times are shown in UK time. Rewind stores only labels and boxes, never camera images.

-- name: Timeline of one item (change it with --label)
SELECT time AT TIME ZONE 'Europe/London' AS at, kind, keyframe
FROM rewind_events
WHERE label = %(label)s
ORDER BY time
LIMIT 25;

-- name: Activity per item every 5 minutes (time_bucket)
SELECT time_bucket('5 minutes', time) AT TIME ZONE 'Europe/London' AS bucket,
       label, count(*) AS events
FROM rewind_events
GROUP BY 1, 2
ORDER BY 1 DESC, 2
LIMIT 20;

-- name: The same, read from the continuous aggregate (Tiger Data keeps it up to date)
SELECT bucket AT TIME ZONE 'Europe/London' AS bucket, label, sum(n) AS events
FROM rewind_events_5min
GROUP BY 1, 2
ORDER BY 1 DESC, 2
LIMIT 20;

-- name: How long each item stayed on the table (appeared -> left)
WITH e AS (
    SELECT time, label, kind,
           LEAD(kind) OVER w AS next_kind,
           LEAD(time) OVER w AS next_time
    FROM rewind_events
    WHERE kind IN ('appeared', 'left')
    WINDOW w AS (PARTITION BY COALESCE(session, 'earlier runs'), label ORDER BY time)
)
SELECT label,
       time AT TIME ZONE 'Europe/London' AS appeared_at,
       date_trunc('second', next_time - time) AS on_the_table
FROM e
WHERE kind = 'appeared' AND next_kind = 'left'
ORDER BY time DESC
LIMIT 15;

-- name: Last known state of every item (last() is a Timescale hyperfunction)
SELECT label,
       max(time) AT TIME ZONE 'Europe/London' AS last_change,
       last(kind, time) AS last_event,
       CASE WHEN last(kind, time) = 'left' THEN 'taken off the table'
            WHEN last(kind, time) = 'picked up' THEN 'with someone'
            ELSE 'on the table' END AS state,
       last(x1, time) AS x, last(y1, time) AS y
FROM rewind_events
GROUP BY label
ORDER BY max(time) DESC;

-- name: Runs of Rewind (one session per run; older rows have none)
SELECT COALESCE(session, 'earlier runs') AS session,
       min(time) AT TIME ZONE 'Europe/London' AS first_event,
       max(time) AT TIME ZONE 'Europe/London' AS last_event,
       count(*) AS events
FROM rewind_events
GROUP BY 1
ORDER BY min(time) DESC
LIMIT 10;
