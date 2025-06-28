# מילון מונחים - תרשימי זרימת נתונים (DFD)
## מערכת Right Motion לעקיבת כושר

---

## 4.6.3.2 מילון מונחים

### ישויות חיצוניות (External Entities):

**משתמש (E1)** - איש הקשר העיקרי עם המערכת, מבצע רישום, התחברות ואימונים. מספק נתוני רישום (registration_data), פרטי התחברות (login_credentials), בקשות תרגיל (exercise_requests) ובקשות היסטוריה (history_requests). מקבל מצב אימות (authentication_status), משוב תרגיל (exercise_feedback), היסטוריית סשנים (session_history) והשמעת וידאו (video_playback).

**חומרת מצלמה (E2-Camera Hardware)** - חומרה חיצונית המספקת זרם וידאו בזמן אמת לזיהוי תנוחות. מספקת הזנת מצלמה (camera_feed), פריימי וידאו (video_frames) וזרם וידאו בזמן אמת (real_time_video_stream).

**אחסון מערכת קבצים (E3-File System Storage)** - אחסון חיצוני לקבצי וידאו מעובדים ונתוני מולטימדיה. מקבלת זרמי וידאו מעובדים (processed_video_streams) וקבצים זמניים (temporary_files), ומספקת קבצי וידאו מאוחסנים (stored_video_files) ומטא-דאטה של וידאו (video_metadata).

---

### טריגרים (Triggers):

**לחיצה על כפתור התחלת אימון (T1)** - מפעיל את תהליך סשן אימון חי (P2). מקבל סוג תרגיל (exercise_type), מזהה משתמש (user_id) ואפשרויות סשן (session_options), ומתחיל זיהוי תנוחות בזמן אמת וניתוח צורה.

**העלאת קובץ וידאו (T2)** - מפעיל את תהליך ניתוח וידאו מוקלט (P3). מקבל קובץ וידאו (video_file), סוג תרגיל (exercise_type) ומזהה משתמש (user_id), ומבצע ניתוח פריים אחר פריים ויצירת משוב.

**לחיצה על היסטוריה (T3)** - מפעיל את תהליך הצגת היסטוריה (P4). מקבל מזהה משתמש (user_id) וקריטריוני סינון (filter_criteria), ומבצע אחזור והצגת היסטוריית סשנים.

**שליחת טופס רישום (T4)** - מפעיל את תהליך רישום משתמש חדש (P1.1). מקבל נתוני רישום (registration_data) ופרטי פרופיל (profile_information), ויוצר חשבון משתמש חדש.

**שליחת נתוני התחברות (T5)** - מפעיל את תהליך אימות משתמשים (P1.2). מקבל פרטי התחברות (login_credentials: email/username, password), ומבצע אימות משתמש והקמת סשן.

**זיהוי תנוחה שגויה (T6)** - מפעיל תהליך משוב בזמן אמת. מקבל נקודות ציון תנוחה (pose_landmarks) ותוצאת ניתוח צורה (form_analysis_result), ומספק משוב תיקון מיידי.

**סיום סשן אימון (T7)** - מפעיל תהליך שמירת נתוני סשן (P2.5). מקבל סיכום סשן (session_summary) וסטטיסטיקות סופיות (final_statistics), ושומר נתוני סשן מלאים למסד הנתונים.

---

### תהליכים עיקריים (Main Processes):

#### **ניהול משתמשים (P1)**
טיפול ברישום/התחברות. מקבל פרטי גישה (credentials) ומספק מצב אימות (auth_status).

- **רישום משתמש (P1.1)** - יצירת חשבון משתמש חדש. מקבל נתוני רישום (registration_data) ומספק רשומת משתמש (user_record)
- **התחברות משתמש (P1.2)** - אימות התחברות. מקבל פרטי התחברות (login_credentials) ומספק תוצאת אימות (auth_result)  
- **מנהל מצב סשן (P1.3)** - ניהול מצב הפעלת המשתמש במערכת

#### **סשן תרגיל (P2)**
מעקב תרגיל חי. מקבל בקשת תרגיל (exercise_request) והזנת מצלמה (camera_feed), מספק נתוני סשן (session_data).

- **אתחול סשן (P2.1)** - אתחול סשן תרגיל. מקבל מזהה תרגיל (exercise_id) מ-DS2
- **זיהוי תנוחות (P2.2)** - זיהוי תנוחות MediaPipe. מקבל הזנת מצלמה ומספק נקודות ציון תנוחה (pose_landmarks)
- **ניתוח צורה (P2.3)** - ניתוח צורת התרגיל. מקבל נקודות ציון ומספק ספירת חזרות ומשוב
- **ספירת חזרות (P2.4)** - ספירת חזרות התרגיל. מקבל חישובי זוויות ומספק ספירת חזרות
- **אחסון סשן (P2.5)** - שמירת נתוני סשן. מקבל פרטי סשן ומספק מזהה סשן (session_id)
- **הקלטת וידאו (P2.6)** - הקלטת וידאו של הסשן. מספק קובץ וידאו (video_file) ל-DS6

#### **ניתוח וידאו (P3)**
ניתוח וידאו מוקלט. מקבל קובץ וידאו (video_file) ומספק וידאו מנותח (analyzed_video).

- **העלאת וידאו (P3.1)** - טיפול בהעלאת קובץ וידאו. מקבל כללי תרגיל (exercise_rules) מ-DS2
- **עיבוד פריים אחר פריים (P3.2)** - עיבוד פריימים בנפרד. מספק נקודות ציון לכל פריים
- **יצירת משוב עם חותמת זמן (P3.3)** - יצירת משוב מתוזמן
- **אחסון וידאו מעובד (P3.4)** - שמירת וידאו מנותח. מספק נתיב וידאו (video_path) ונתוני סשן ל-DS3

#### **הצגת היסטוריה (P4)**
הצגת היסטוריית משתמש. מקבל בקשת היסטוריה (history_request) ומספק היסטוריה מעוצבת (formatted_history).

- **אחזור סשנים (P4.1)** - שליפת סשנים של משתמש. מקבל סשנים של משתמש (user_sessions) מ-DS3
- **העשרת פרטים (P4.2)** - העשרת פרטי סשן. מקבל פרטי סשן (session_details) מ-DS4  
- **שילוב וידאו (P4.3)** - שילוב קבצי וידאו. מקבל קבצי וידאו (video_files) מ-DS6
- **תצוגת ממשק משתמש (P4.4)** - עיצוב תצוגת ההיסטוריה

#### **זרימת וידאו (P5)**
הגשת קבצי וידאו. מקבל בקשת וידאו (video_request) ומספק זרם וידאו (video_stream).

- **זרימה בזמן אמת (P5.1)** - זרימת וידאו חי. מקבל קבצי וידאו מ-DS6 ומספק זרם חי
- **הגשת וידאו מעובד (P5.2)** - הגשת וידאו מנותח. מקבל נתיב וידאו ומספק תגובת וידאו

---

### מאגרי נתונים (Data Stores):

#### **מאגר משתמשים (DS1 - User Store)**
אחסון פרטי משתמשים, פרופילים, נתוני רישום ואימות.
- **טבלאות**: User
- **נתונים**: user_id, email, username, password_hash, profile_data, user_type, registration_data, authentication_data
- **פעולות**: יצירה, קריאה, עדכון (אימות, עדכוני פרופיל)

#### **מאגר תרגילים (DS2 - Exercise Store)**
הגדרות תרגילים, כללי ניתוח וסטנדרטים לביצוע.
- **טבלאות**: Exercise, Workout  
- **נתונים**: exercise_id, exercise_name, target_muscles, instructions, workout_plans, analysis_rules, performance_standards
- **פעולות**: קריאה (הגדרות תרגילים, תבניות אימון, קריטריוני ניתוח)

#### **מאגר סשנים (DS3 - Session Store)**
נתוני אימונים, תזמון ומטא-דאטה של סשנים.
- **טבלאות**: Session
- **נתונים**: session_id, user_id, exercise_id, workout_id, timestamps, duration, session_status, video_path, metadata
- **פעולות**: יצירה, קריאה (מעקב סשן, אחזור היסטוריה)

#### **מאגר פרטי סשן (DS4 - Session Details Store)**  
נתוני חזרות, דיוק ביצוע וניתוח מפורט.
- **טבלאות**: SessionDetails
- **נתונים**: detail_id, session_id, rep_number, timestamp, features_json, is_correct_form, incorrect_duration, detailed_analysis
- **פעולות**: יצירה, קריאה (מעקב חזרה אחר חזרה, ניתוח ביצועים מפורט)

#### **מאגר משוב מערכתי (DS5 - System Feedback Store)**
הודעות תיקון, המלצות והנחיות שניתנו במהלך האימונים.
- **טבלאות**: SystemFeedback
- **נתונים**: feedback_id, session_id, message, feedback_type, related_rep, timestamp, correction_messages, recommendations
- **פעולות**: יצירה, קריאה (משוב בזמן אמת, הודעות הנחיה)

#### **מאגר וידאו (DS6 - Video Store)**
אחסון קבצי וידאו, נתיבי קבצים ומטא-דאטה.
- **מיקום**: מערכת קבצים (תיקיית /videos)
- **נתונים**: video_files, file_paths, metadata, processed_videos, temporary_files, video_metadata
- **פעולות**: יצירה, קריאה (הקלטת וידאו, השמעה, ניהול קבצים)

---

### תהליכי ניתוח צורה מפורטים (Form Analysis Process 2.3):

#### **בודק תרגיל (2.3.1 - Exercise Checker)**
בחירת בודק התרגיל המתאים ואחזור כללי תרגיל (exercise_rules) מ-DS2. מקבל נקודות ציון תנוחה (pose_landmarks) מתהליך 2.2 ומספק חישובי זוויות (angle_calculations).

#### **מונה חזרות (2.3.2 - Rep Counter)**  
ספירת חזרות על בסיס חישובי זוויות ותנועה. מקבל חישובי זוויות ומספק ספירת חזרות (rep_count).

#### **מחולל משוב (2.3.3 - Feedback Generator)**
יצירת הודעות משוב והנחיות תיקון. מקבל ספירת חזרות ומספק הודעות משוב (feedback_messages).

#### **פרטי סשן (2.3.4 - Session Details)**
איסוף ועדכון פרטי ביצוע מפורטים לכל חזרה. מקבל הודעות משוב ומספק פרטי סשן (session_details) לתהליך 2.4.

---

### זרמי נתונים עיקריים (Data Flows):

| זרם נתונים | תיאור | הרכב |
|------------|-------|------|
| user_credentials | פרטי התחברות | אימייל/שם משתמש, סיסמה |
| exercise_request | בקשת תחילת סשן | סוג תרגיל, מזהה משתמש, אפשרויות |
| pose_data | נקודות ציון MediaPipe | קואורדינטות x,y,z, נראות |
| session_data | מידע סשן מלא | חותמות זמן, חזרות, משך, מצב |
| feedback_data | תיקוני צורה | הודעות, חותמות זמן, קישור לחזרה |
| processed_video | סשן מוקלט | קובץ וידאו, שכבות תנוחה, מטא-דאטה |
| history_request | שאילתת היסטוריית משתמש | מזהה משתמש, מסננים, עימוד |

---

### הגדרות תהליכים מפורטות:

| תהליך | תיאור | קלטים | פלטים |
|-------|-------|-------|-------|
| P1 ניהול משתמשים | טיפול ברישום/התחברות | credentials | auth_status |
| P1.1 רישום משתמש | יצירת חשבון משתמש חדש | registration_data | user_record |
| P1.2 אימות משתמש | אימות התחברות | login_credentials | auth_result |
| P1.3 ניהול פרופיל | ניהול פרופיל משתמש | profile_updates | updated_profile |
| P2 סשן תרגיל | מעקב תרגיל חי | exercise_request, camera_feed | session_data |
| P2.1 אתחול סשן | אתחול סשן תרגיל | exercise_type, user_id | session_state |
| P2.2 זיהוי תנוחות | ניתוח תנוחות MediaPipe | camera_feed | pose_landmarks |
| P2.3 ניתוח צורה | בדיקת צורת תרגיל | pose_landmarks | rep_count, feedback |
| P2.4 ספירת חזרות | ספירת חזרות תרגיל | angle_calculations | rep_count |
| P2.5 אחסון סשן | שמירת נתוני סשן | session_details | session_id |
| P3 ניתוח וידאו | ניתוח וידאו מוקלט | video_file | analyzed_video |
| P3.1 העלאת וידאו | טיפול בהעלאת וידאו | video_file, exercise_type | uploaded_video |
| P3.2 עיבוד פריימים | ניתוח פריים אחר פריים | video_frames | pose_landmarks_per_frame |
| P3.3 משוב מתוזמן | יצירת משוב עם זמן | analysis_results | timestamped_feedback |
| P3.4 אחסון וידאו מעובד | שמירת וידאו מנותח | processed_video | video_path |
| P4 הצגת היסטוריה | הצגת היסטוריית משתמש | history_request | formatted_history |
| P4.1 אחזור סשנים | שליפת סשנים של משתמש | user_id | session_list |
| P4.2 הצגת רשימת אימונים | הצגת רשימת אימונים | sessions | workout_display |
| P4.3 הצגת וידאו וסטטיסטיקות | הצגת וידאו ונתונים | session_data | enriched_display |
| P5 זרימת וידאו | הגשת קבצי וידאו | video_request | video_stream |
| P5.1 זרימה בזמן אמת | זרימת וידאו חי | video_files | live_stream |
| P5.2 הגשת וידאו מעובד | הגשת וידאו מנותח | video_path | video_response |