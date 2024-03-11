# Video events
[Documentation](https://edx.readthedocs.io/projects/devdata/en/latest/internal_data_formats/tracking_logs/student_event_types.html#video-interaction-events)

## Play

### Example event
{"username":"","event_source":"browser","name":"play_video","accept_language":"en-us","time":"2018-05-15T19:35:54.180026+00:00","agent":"User-Agent","page":"","host":"","session":"session-id","referer":"","context":{"user_id":"user-id","org_id":"org-id","course_id":"course-id","path":"","client":"","host":"","ip":"","username":""},"ip":"","event":{"duration":478.56,"code":"video-id","id":"event-id","currentTime":136},"event_type":"play_video"}

event_type: play_video

### Variables
Play (P) -> If currentTime > 0

## Pause
{"username":"","event_source":"browser","name":"pause_video","accept_language":"en-us","time":"2018-05-15T19:34:53.531565+00:00","agent":"User-Agent","page":"","host":"","session":"session-id","referer":"","context":{"user_id":"user-id","org_id":"org-id","course_id":"course-id","path":"","client":"","host":"","ip":"","username":""},"ip":"","event":{"duration":478.56,"code":"video-id","id":"event-id","currentTime":154.0546},"event_type":"pause_video"}

event_type: pause_video

## Seek

### Example event
{"username":"","event_source":"browser","name":"seek_video","accept_language":"en-us","time":"2018-05-15T19:35:53.192979+00:00","agent":"User-Agent","page":"","host":"","session":"session-id","referer":"","context":{"user_id":"user-id","org_id":"org-id","course_id":"course-id","path":"","client":"","host":"","ip":"","username":""},"ip":"","event":{"code":"video-id","new_time":136,"old_time":2.1026,"duration":478.56,"type":"onSlideSeek","id":"event-id"},"event_type":"seek_video"}

### Variables
SeekFw (Sf) -> If new_time > old_time
SeekBw (Sb) -> If new_time < old_time
ScrollFw (SSf) -> If new_time > old_time and two seeks within one second
ScrollBw (SSb) -> If new_time < old_time and two seeks within one second

event_type: seek_video

## Load

### Example event
{"username":"","context":{"component":"videoplayer","received_at":"2018-05-15T19:18:22.244000+00:00","course_id":"course-id","path":"","user_id":"user-id","org_id":"org-id","application":{"version":"2.13.2","name":"edx.mobileapp.android"},"client":"","host":"","ip":"","username":""},"event_source":"mobile","name":"edx.video.loaded","ip":"","agent":"User-Agent","event":{"code":"video-id","id":"event-id"},"host":"","session":"","referer":"","accept_language":"","time":"2018-05-15T19:18:22.244000+00:00","page":"","event_type":"load_video"}

event_type: load_video

### Variables
-

## Stop

### Example event
{"username":"","event_source":"browser","name":"stop_video","accept_language":"de-DE,de;q=0.5","time":"2018-05-15T20:06:50.716240+00:00","agent":"User-Agent","page":"","host":"","session":"session-id","referer":"","context":{"user_id":"user-id","org_id":"org-id","course_id":"course-id","path":"","client":"","host":"","ip":"","username":""},"ip":"","event":{"duration":304.09,"code":"video-id","id":"event-id","currentTime":304.3948},"event_type":"stop_video"}

event_type: stop_video

Emitted when video is streamed until the end in a browser, together with a pause_event.
### Variables
-

## Speed change

### Example event
{"username":"","event_source":"browser","name":"speed_change_video","accept_language":"nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7","time":"2018-05-15T12:25:05.211418+00:00","agent":"User-Agent","page":"","host":"","session":"session-id","referer":"","context":{"user_id":"user-id","org_id":"org-id","course_id":"course-id","path":"","client":"","host":"","ip":"","username":""},"ip":"","event":{"current_time":25.548,"old_speed":"1.0","code":"video-id","new_speed":"1.25","duration":113.45,"id":"event-id"},"event_type":"speed_change_video"}

event_type: speed_change_video

### Variables
RatechangeFast (Rf) -> If new_speed > old_speed
RatechangeSlow (Rs) -> If new_speed < old_speed